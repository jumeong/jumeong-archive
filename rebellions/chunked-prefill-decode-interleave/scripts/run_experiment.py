"""Run paired BF16 policy experiments and save all client traces locally."""
import asyncio
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from types import SimpleNamespace
import urllib.request

from transformers import AutoTokenizer
import measure

ROOT = Path(__file__).resolve().parents[1]
SOURCE = '/home/juwon/git/hello-npu-vllm-rbln'
PYTHON = '/home/juwon/rbln-env/bin/python'
MODEL = ROOT / 'models/compiled-b4-s4096'
URL = 'http://127.0.0.1:8124'
SERVER_LOG = None


def save_run(name, policy, workload):
    out = ROOT / 'data/results' / name / policy
    out.mkdir(parents=True, exist_ok=False)
    args = SimpleNamespace(model='experiment', url=URL, timeout=1800)
    log_offset = SERVER_LOG.stat().st_size
    rows = asyncio.run(measure.replay(args, workload))
    (out / 'raw.json').write_text(json.dumps(rows))
    (out / 'summary.json').write_text(json.dumps(measure.summarize(rows), indent=2))
    (out / 'perfetto.json').write_text(json.dumps(measure.trace(rows, policy)))
    if any(r['error'] for r in rows):
        raise RuntimeError(f'Failed requests in {out}')
    with SERVER_LOG.open('rb') as f:
        f.seek(log_offset)
        compiled_during_run = b'Compile(#' in f.read()
    (out / 'validation.json').write_text(json.dumps(
        {'compiled_during_run': compiled_during_run}, indent=2))
    if compiled_during_run and not name.startswith('warmup'):
        raise RuntimeError(f'Compilation contaminated timing in {out}')
    print('Completed', name, policy, flush=True)
    return rows


def start_server(policy, label):
    global SERVER_LOG
    SERVER_LOG = ROOT / f'logs/server-{label}.log'
    env = {**os.environ, 'PYTHONPATH': SOURCE, 'VLLM_RBLN_USE_VLLM_MODEL': '0',
           'VLLM_RBLN_INTERLEAVE_POLICY': policy, 'VLLM_RBLN_INTERLEAVE_K': '1',
           'VLLM_WORKER_MULTIPROC_METHOD': 'spawn', 'TOKENIZERS_PARALLELISM': 'false'}
    cmd = [PYTHON, '-m', 'vllm.entrypoints.openai.api_server', '--model', str(MODEL),
           '--served-model-name', 'experiment', '--host', '127.0.0.1', '--port', '8124',
           '--max-model-len', '4096', '--max-num-seqs', '4',
           '--no-enable-prefix-caching', '--disable-log-requests']
    (ROOT / f'metadata/server-{label}.json').write_text(json.dumps(dict(command=cmd,
        overrides={k: env[k] for k in ('PYTHONPATH', 'VLLM_RBLN_USE_VLLM_MODEL',
                    'VLLM_RBLN_INTERLEAVE_POLICY', 'VLLM_RBLN_INTERLEAVE_K')}), indent=2))
    with SERVER_LOG.open('w') as log:
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
    for _ in range(300):
        if proc.poll() is not None:
            raise RuntimeError(f'Server failed; see server-{label}.log')
        try:
            with urllib.request.urlopen(URL + '/health', timeout=2) as response:
                if response.status == 200:
                    return proc
        except Exception:
            time.sleep(2)
    stop_server(proc)
    raise TimeoutError('Server startup')


def stop_server(proc):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    time.sleep(3)


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    base = tok.encode('Explain how a computer processes information. Give a detailed technical discussion. ',
                      add_special_tokens=False)
    def spec(idx, n, out, at=0, group='mixed'):
        prefix = tok.encode(f'Request {idx}. ', add_special_tokens=False)
        prompt = (prefix + base * (n // len(base) + 2))[:n]
        return dict(id=str(idx), prompt=prompt, max_tokens=out, at_s=at, group=group)

    def warmup(policy, label):
        for count in range(1,5):
            save_run(f'warmup-{label}-batch{count}',policy,
                     [spec(f'w{i}',128,32) for i in range(count)])
        save_run(f'warmup-{label}-interleave',policy,
                 [spec(f'bg{i}',128,96) for i in range(3)] +
                 [spec('long',1024,32,1)])

    proc = start_server('none', 'pilot')
    try:
        warmup('none','pilot')
        pilot = save_run('pilot', 'none', [spec('short',128,64), spec('long',2048,64,10)])
    finally:
        stop_server(proc)
    short = pilot[0]
    itl = measure.percentile([b['at_s']-a['at_s'] for a,b in
                             zip(short['events'],short['events'][1:])],50)
    ttft = short['events'][0]['at_s']-short['sent_s']
    injection = max(1., 3*ttft + 8*itl)
    metadata = dict(itl_s=itl, ttft_s=ttft, injection_s=injection)
    (ROOT / 'metadata/pilot-settings.json').write_text(json.dumps(metadata,indent=2))

    for seed in range(3):
        workloads = {}
        for length in [1024,2048,3072]:
            workloads[f'injection-{length}-seed{seed}'] = [
                spec(f'bg0-{seed}',128,512,group='background'),
                spec(f'bg1-{seed}',128,512,group='background'),
                spec(f'long-{seed}',length,64,injection,'injected')]
        for name, workload in workloads.items():
            (ROOT / f'data/workloads/{name}-workload.json').write_text(json.dumps(workload))
        for policy in (['none','fixed'] if seed%2==0 else ['fixed','none']):
            label=f'{policy}-seed{seed}'
            proc = start_server(policy,label)
            try:
                warmup(policy,label)
                for name, workload in workloads.items():
                    rows=save_run(name,policy,workload)
                    if name.startswith('injection'):
                        injected=rows[2]['sent_s']
                        if not all(r['events'][0]['at_s'] < injected < r['events'][-1]['at_s']
                                   for r in rows[:2]):
                            raise RuntimeError('Injection did not overlap both background decodes')
            finally:
                stop_server(proc)


if __name__ == '__main__':
    main()

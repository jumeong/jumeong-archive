import csv
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]


def main():
    records = []
    correctness = []
    for experiment in sorted((ROOT / 'data/results').iterdir()):
        if not experiment.name.startswith('injection-'):
            continue
        raw = {}
        summaries = {}
        for policy in ['none', 'fixed']:
            raw[policy] = json.loads((experiment / policy / 'raw.json').read_text())
            summaries[policy] = json.loads((experiment / policy / 'summary.json').read_text())
        tokens = {p: {r['id']: [token for e in r['events'] for token in e['ids']]
                      for r in raw[p]} for p in raw}
        matches = sum(tokens['none'][key] == tokens['fixed'][key] for key in tokens['none'])
        correctness.append(f'{experiment.name}: {matches}/{len(tokens["none"])} request outputs identical')
        combined = []
        for pid, policy in enumerate(['none', 'fixed'], 1):
            trace = json.loads((experiment / policy / 'perfetto.json').read_text())
            for event in trace['traceEvents']:
                event['pid'] = pid
                combined.append(event)
        (ROOT / 'perfetto' / 'injection' / (experiment.name + '.json')).write_text(json.dumps(
            {'traceEvents':combined,'displayTimeUnit':'ms'}))
        for group in summaries['none']:
            for metric in ['ttft_ms','itl_ms','request_max_gap_ms']:
                for quantile in ['p50','p95','p99','p99.9','max']:
                    values = [summaries[p][group][metric] for p in ['none','fixed']]
                    if any(v is None or v[quantile] is None for v in values):
                        continue
                    a,b = [v[quantile] for v in values]
                    records.append(dict(experiment=experiment.name, group=group, metric=metric,
                        quantile=quantile, none_ms=a, fixed_ms=b, delta_ms=b-a,
                        delta_percent=(b/a-1)*100 if a else None,
                        none_samples=values[0]['n'], fixed_samples=values[1]['n']))
    with (ROOT / 'reports/comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]))
        writer.writeheader();writer.writerows(records)
    lines=['# Qwen3-4B original BF16: none vs fixed K=1','',
           'Client-observed timing. Same PR commit, model and fixed decode batch 4.',
           'Three-run medians of per-run statistics; percentage is ratio of these medians.',
           'Traces align independent runs at client start; tracks are not accelerator kernels.', '',
           '| Experiment | Group | Metric | none (ms) | fixed (ms) | Change |',
           '|---|---|---|---:|---:|---:|']
    keys = sorted({(r['experiment'].split('-seed')[0], r['group'], r['metric'], r['quantile'])
                   for r in records if r['quantile'] in ['p50','p99','max']})
    for name,group,metric,q in keys:
        rows=[r for r in records if r['experiment'].split('-seed')[0]==name
              and (r['group'],r['metric'],r['quantile'])==(group,metric,q)]
        a=median(r['none_ms'] for r in rows);b=median(r['fixed_ms'] for r in rows)
        change=f'{(b/a-1)*100:+.1f}%' if a else 'N/A'
        lines.append(f'| {name} | {group} | {metric} {q} | {a:.2f} | {b:.2f} | {change} |')
    lines += ['', '## Output token comparison', '', *correctness]
    lines += ['', '## Interpretation limits', '',
              'Synthetic repeated-text prompts and forced output lengths; not representative production traffic.',
              'Three requests per injection run; p99.9 and maximum are descriptive, not stable tail estimates.',
              'Warmup excluded. Both policies disable prefix caching. BF16 refers to source/export dtype;',
              'accelerator internal arithmetic is controlled by the compiler.',
              'Open a perfetto/injection/*.json file in https://ui.perfetto.dev.']
    (ROOT / 'reports/REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()

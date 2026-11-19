"""Replay a fixed open-loop workload; preserve SSE token timestamps for Perfetto.

Workload JSON: [{"id": "r0", "at_s": 0, "prompt": [token IDs],
                 "max_tokens": 128, "group": "short"}, ...]
Run once per policy against a warm, freshly started server.
"""
import argparse
import asyncio
import json
import time
from pathlib import Path

import aiohttp


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    x = (len(values) - 1) * p / 100
    lo = int(x)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (x - lo)


def stats(values):
    return {"n": len(values), **{f"p{p}": percentile(values, p)
                               for p in (50, 95, 99, 99.9)},
            "max": max(values, default=None)}


async def replay(args, workload):
    origin = time.perf_counter()
    def now():
        return time.perf_counter() - origin

    async def request(session, spec):
        await asyncio.sleep(max(0, spec["at_s"] - now()))
        row = {**spec, "sent_s": now(), "events": [], "error": None}
        payload = dict(model=args.model, prompt=spec["prompt"],
                       max_tokens=spec["max_tokens"], temperature=0,
                       ignore_eos=True, stream=True, return_token_ids=True,
                       stream_options={"include_usage": True})
        done = False
        try:
            async with session.post(args.url.rstrip('/') + '/v1/completions',
                                    json=payload) as response:
                response.raise_for_status()
                # readline handles transport fragmentation without treating TCP
                # packets as token events. vLLM emits one JSON per SSE data line.
                async for line in response.content:
                    if not line.startswith(b'data:'):
                        continue
                    data = line[5:].strip()
                    received = now()
                    if data == b'[DONE]':
                        done = True
                        break
                    event = json.loads(data)
                    if 'error' in event:
                        raise RuntimeError(str(event['error']))
                    if event.get('usage'):
                        row['usage'] = event['usage']
                    for choice in event.get('choices', []):
                        ids = choice.get('token_ids') or []
                        if ids:
                            row['events'].append({'at_s': received, 'ids': ids})
            count = sum(len(e['ids']) for e in row['events'])
            if not done or count != spec['max_tokens']:
                raise RuntimeError(f'incomplete stream: done={done}, tokens={count}')
        except Exception as exc:
            row['error'] = repr(exc)
        row['end_s'] = now()
        return row

    timeout = aiohttp.ClientTimeout(total=args.timeout)
    async with aiohttp.ClientSession(timeout=timeout,
                                    connector=aiohttp.TCPConnector(limit=0)) as session:
        return await asyncio.gather(*(request(session, spec) for spec in workload))


def summarize(rows):
    groups = {}
    for group in ['all'] + sorted({r.get('group', 'default') for r in rows} - {'all'}):
        subset = [r for r in rows if group == 'all' or r.get('group', 'default') == group]
        valid = [r for r in subset if not r['error']]
        # If a server batches token emissions, individual client ITL is unknown.
        # Preserve events, report chunk gaps, and do not invent zero token gaps.
        single = all(len(e['ids']) == 1 for r in valid for e in r['events'])
        gaps = [1000 * (b['at_s'] - a['at_s']) for r in valid
                for a, b in zip(r['events'], r['events'][1:])]
        groups[group] = {
            'requests': len(subset), 'successful': len(valid),
            'ttft_ms': stats([1000 * (r['events'][0]['at_s'] - r['sent_s']) for r in valid]),
            'client_schedule_lag_ms': stats([1000 * (r['sent_s'] - r['at_s']) for r in subset]),
            'itl_ms': stats(gaps) if single else None,
            'stream_chunk_gap_ms': stats(gaps),
            'all_events_single_token': single,
            'request_max_gap_ms': stats([max(
                [1000 * (b['at_s'] - a['at_s']) for a, b in zip(r['events'], r['events'][1:])],
                default=0) for r in valid]),
        }
    return groups


def trace(rows, policy):
    events = [{'ph': 'M', 'pid': 1, 'tid': 0, 'name': 'process_name',
               'args': {'name': f'{policy}: client observations (not NPU execution)'}}]
    for tid, row in enumerate(rows, 1):
        events.append({'ph': 'M', 'pid': 1, 'tid': tid, 'name': 'thread_name',
                       'args': {'name': f"{row['id']} ({row.get('group', 'default')})"}})
        previous = row['sent_s']
        for i, event in enumerate(row['events']):
            events.append({'ph': 'X', 'pid': 1, 'tid': tid,
                           'name': 'TTFT' if i == 0 else 'stream gap',
                           'ts': previous * 1e6, 'dur': (event['at_s'] - previous) * 1e6,
                           'args': {'token_ids': event['ids'], 'event_index': i}})
            previous = event['at_s']
        if row['error']:
            events.append({'ph': 'i', 's': 't', 'pid': 1, 'tid': tid,
                           'name': 'ERROR', 'ts': row['end_s'] * 1e6,
                           'args': {'error': row['error']}})
    return {'traceEvents': events, 'displayTimeUnit': 'ms'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--model', required=True)
    parser.add_argument('--policy', choices=['none', 'fixed'], required=True)
    parser.add_argument('--workload', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=1800)
    args = parser.parse_args()
    workload = json.loads(args.workload.read_text())
    assert workload and len({r['id'] for r in workload}) == len(workload)
    assert all(r['at_s'] >= 0 and r['max_tokens'] > 0 and r['prompt'] for r in workload)
    args.out.mkdir(parents=True, exist_ok=False)
    rows = asyncio.run(replay(args, workload))
    (args.out / 'raw.json').write_text(json.dumps(rows))
    (args.out / 'summary.json').write_text(json.dumps(summarize(rows), indent=2))
    (args.out / 'perfetto.json').write_text(json.dumps(trace(rows, args.policy)))
    (args.out / 'run.json').write_text(json.dumps({**vars(args), 'workload': str(args.workload),
                                                'out': str(args.out)}, indent=2))
    if any(r['error'] for r in rows):
        raise SystemExit('Some requests failed; inspect raw.json before comparing policies.')


if __name__ == '__main__':
    main()

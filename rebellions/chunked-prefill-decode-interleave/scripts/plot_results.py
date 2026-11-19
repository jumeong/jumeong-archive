"""Create static, exportable figures from completed measured runs."""
import json
from pathlib import Path
from statistics import median
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
lengths=[1024,2048,3072]
for ax,group,metric,q,title in zip(axes,
        ['injected','background','background'],
        ['ttft_ms','itl_ms','itl_ms'],['p50','p99','max'],
        ['New request: TTFT','Background: ITL p99','Background: maximum ITL']):
    for policy,color in [('none','#555b65'),('fixed','#137b9e')]:
        runs=[]
        for length in lengths:
            values=[]
            for seed in range(3):
                p=ROOT/'data/results'/f'injection-{length}-seed{seed}'/policy/'summary.json'
                values.append(json.loads(p.read_text())[group][metric][q])
            runs.append(values)
        ys=[median(v) for v in runs]
        errors=[[y-min(v) for y,v in zip(ys,runs)],
                [max(v)-y for y,v in zip(ys,runs)]]
        ax.errorbar(lengths,ys,yerr=errors,marker='o',label=policy if policy=='none' else 'fixed K=1',
                    color=color,capsize=4)
    ax.set(title=title,xlabel='Injected prompt tokens',ylabel='Milliseconds',ylim=(0,None))
    ax.set_xticks(lengths)
    ax.grid(alpha=.2)
axes[0].legend()
fig.suptitle('Qwen3-4B BF16 · CA22 · decode batch 4 · 2 background requests\n'
             'Median of 3 runs; whiskers show min–max; client-observed timing')
fig.savefig(ROOT/'reports/tradeoff.png',dpi=180)
fig.savefig(ROOT/'reports/tradeoff.svg')

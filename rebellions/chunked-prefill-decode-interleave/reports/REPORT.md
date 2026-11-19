# Qwen3-4B original BF16: none vs fixed K=1

Client-observed timing. Same PR commit, model and fixed decode batch 4.
Three-run medians of per-run statistics; percentage is ratio of these medians.
Traces align independent runs at client start; tracks are not accelerator kernels.

| Experiment | Group | Metric | none (ms) | fixed (ms) | Change |
|---|---|---|---:|---:|---:|
| injection-1024 | all | itl_ms max | 535.23 | 122.23 | -77.2% |
| injection-1024 | all | itl_ms p50 | 51.07 | 51.64 | +1.1% |
| injection-1024 | all | itl_ms p99 | 57.74 | 110.29 | +91.0% |
| injection-1024 | all | request_max_gap_ms max | 535.23 | 122.23 | -77.2% |
| injection-1024 | all | request_max_gap_ms p50 | 535.15 | 119.69 | -77.6% |
| injection-1024 | all | request_max_gap_ms p99 | 535.23 | 122.23 | -77.2% |
| injection-1024 | all | ttft_ms max | 516.32 | 886.48 | +71.7% |
| injection-1024 | all | ttft_ms p50 | 138.95 | 192.60 | +38.6% |
| injection-1024 | all | ttft_ms p99 | 508.92 | 872.60 | +71.5% |
| injection-1024 | background | itl_ms max | 535.23 | 122.23 | -77.2% |
| injection-1024 | background | itl_ms p50 | 51.05 | 51.64 | +1.2% |
| injection-1024 | background | itl_ms p99 | 58.52 | 110.29 | +88.5% |
| injection-1024 | background | request_max_gap_ms max | 535.23 | 122.23 | -77.2% |
| injection-1024 | background | request_max_gap_ms p50 | 535.19 | 121.88 | -77.2% |
| injection-1024 | background | request_max_gap_ms p99 | 535.23 | 122.23 | -77.2% |
| injection-1024 | background | ttft_ms max | 138.95 | 192.60 | +38.6% |
| injection-1024 | background | ttft_ms p50 | 105.82 | 134.26 | +26.9% |
| injection-1024 | background | ttft_ms p99 | 138.28 | 191.43 | +38.4% |
| injection-1024 | injected | itl_ms max | 56.94 | 58.46 | +2.7% |
| injection-1024 | injected | itl_ms p50 | 51.96 | 51.81 | -0.3% |
| injection-1024 | injected | itl_ms p99 | 56.53 | 56.37 | -0.3% |
| injection-1024 | injected | request_max_gap_ms max | 56.94 | 58.46 | +2.7% |
| injection-1024 | injected | request_max_gap_ms p50 | 56.94 | 58.46 | +2.7% |
| injection-1024 | injected | request_max_gap_ms p99 | 56.94 | 58.46 | +2.7% |
| injection-1024 | injected | ttft_ms max | 516.32 | 886.48 | +71.7% |
| injection-1024 | injected | ttft_ms p50 | 516.32 | 886.48 | +71.7% |
| injection-1024 | injected | ttft_ms p99 | 516.32 | 886.48 | +71.7% |
| injection-2048 | all | itl_ms max | 1064.19 | 129.10 | -87.9% |
| injection-2048 | all | itl_ms p50 | 51.23 | 51.69 | +0.9% |
| injection-2048 | all | itl_ms p99 | 59.37 | 115.63 | +94.8% |
| injection-2048 | all | request_max_gap_ms max | 1064.19 | 129.10 | -87.9% |
| injection-2048 | all | request_max_gap_ms p50 | 1063.85 | 129.07 | -87.9% |
| injection-2048 | all | request_max_gap_ms p99 | 1064.18 | 129.10 | -87.9% |
| injection-2048 | all | ttft_ms max | 1066.07 | 1827.07 | +71.4% |
| injection-2048 | all | ttft_ms p50 | 144.21 | 209.02 | +44.9% |
| injection-2048 | all | ttft_ms p99 | 1047.56 | 1794.82 | +71.3% |
| injection-2048 | background | itl_ms max | 1064.19 | 129.10 | -87.9% |
| injection-2048 | background | itl_ms p50 | 51.12 | 51.62 | +1.0% |
| injection-2048 | background | itl_ms p99 | 59.79 | 115.65 | +93.4% |
| injection-2048 | background | request_max_gap_ms max | 1064.19 | 129.10 | -87.9% |
| injection-2048 | background | request_max_gap_ms p50 | 1064.02 | 129.08 | -87.9% |
| injection-2048 | background | request_max_gap_ms p99 | 1064.19 | 129.10 | -87.9% |
| injection-2048 | background | ttft_ms max | 144.21 | 209.02 | +44.9% |
| injection-2048 | background | ttft_ms p50 | 111.11 | 143.32 | +29.0% |
| injection-2048 | background | ttft_ms p99 | 143.55 | 207.71 | +44.7% |
| injection-2048 | injected | itl_ms max | 60.03 | 63.89 | +6.4% |
| injection-2048 | injected | itl_ms p50 | 52.35 | 52.54 | +0.4% |
| injection-2048 | injected | itl_ms p99 | 59.26 | 61.31 | +3.5% |
| injection-2048 | injected | request_max_gap_ms max | 60.03 | 63.89 | +6.4% |
| injection-2048 | injected | request_max_gap_ms p50 | 60.03 | 63.89 | +6.4% |
| injection-2048 | injected | request_max_gap_ms p99 | 60.03 | 63.89 | +6.4% |
| injection-2048 | injected | ttft_ms max | 1066.07 | 1827.07 | +71.4% |
| injection-2048 | injected | ttft_ms p50 | 1066.07 | 1827.07 | +71.4% |
| injection-2048 | injected | ttft_ms p99 | 1066.07 | 1827.07 | +71.4% |
| injection-3072 | all | itl_ms max | 1643.90 | 131.59 | -92.0% |
| injection-3072 | all | itl_ms p50 | 52.18 | 51.88 | -0.6% |
| injection-3072 | all | itl_ms p99 | 60.74 | 122.09 | +101.0% |
| injection-3072 | all | request_max_gap_ms max | 1643.90 | 131.59 | -92.0% |
| injection-3072 | all | request_max_gap_ms p50 | 1643.61 | 131.58 | -92.0% |
| injection-3072 | all | request_max_gap_ms p99 | 1643.90 | 131.59 | -92.0% |
| injection-3072 | all | ttft_ms max | 1640.21 | 2804.15 | +71.0% |
| injection-3072 | all | ttft_ms p50 | 142.12 | 183.95 | +29.4% |
| injection-3072 | all | ttft_ms p99 | 1610.33 | 2751.81 | +70.9% |
| injection-3072 | background | itl_ms max | 1643.90 | 131.59 | -92.0% |
| injection-3072 | background | itl_ms p50 | 52.08 | 51.76 | -0.6% |
| injection-3072 | background | itl_ms p99 | 61.00 | 122.55 | +100.9% |
| injection-3072 | background | request_max_gap_ms max | 1643.90 | 131.59 | -92.0% |
| injection-3072 | background | request_max_gap_ms p50 | 1643.76 | 131.58 | -92.0% |
| injection-3072 | background | request_max_gap_ms p99 | 1643.90 | 131.59 | -92.0% |
| injection-3072 | background | ttft_ms max | 142.12 | 183.95 | +29.4% |
| injection-3072 | background | ttft_ms p50 | 109.46 | 128.84 | +17.7% |
| injection-3072 | background | ttft_ms p99 | 141.47 | 182.85 | +29.3% |
| injection-3072 | injected | itl_ms max | 61.15 | 78.10 | +27.7% |
| injection-3072 | injected | itl_ms p50 | 53.85 | 53.11 | -1.4% |
| injection-3072 | injected | itl_ms p99 | 60.57 | 65.65 | +8.4% |
| injection-3072 | injected | request_max_gap_ms max | 61.15 | 78.10 | +27.7% |
| injection-3072 | injected | request_max_gap_ms p50 | 61.15 | 78.10 | +27.7% |
| injection-3072 | injected | request_max_gap_ms p99 | 61.15 | 78.10 | +27.7% |
| injection-3072 | injected | ttft_ms max | 1640.21 | 2804.15 | +71.0% |
| injection-3072 | injected | ttft_ms p50 | 1640.21 | 2804.15 | +71.0% |
| injection-3072 | injected | ttft_ms p99 | 1640.21 | 2804.15 | +71.0% |

## Output token comparison

injection-1024-seed0: 3/3 request outputs identical
injection-1024-seed1: 3/3 request outputs identical
injection-1024-seed2: 3/3 request outputs identical
injection-2048-seed0: 3/3 request outputs identical
injection-2048-seed1: 3/3 request outputs identical
injection-2048-seed2: 3/3 request outputs identical
injection-3072-seed0: 3/3 request outputs identical
injection-3072-seed1: 3/3 request outputs identical
injection-3072-seed2: 3/3 request outputs identical

## Interpretation limits

Synthetic repeated-text prompts and forced output lengths; not representative production traffic.
Three requests per injection run; p99.9 and maximum are descriptive, not stable tail estimates.
Warmup excluded. Both policies disable prefix caching. BF16 refers to source/export dtype;
accelerator internal arithmetic is controlled by the compiler.
Open a perfetto/injection/*.json file in https://ui.perfetto.dev.

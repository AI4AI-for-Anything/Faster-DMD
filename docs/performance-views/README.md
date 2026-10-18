# Performance measurements

`01-h20-benchmark.svg` uses all 2,000 effective-step entries per arm from the completed 39k→41k continuations: baseline 29.497 s, optimized 17.627 s, speedup 1.67×. Hardware: eight H20 GPUs; global batch 2,048 = 8 × 32 × 8. Checkpoint writes are outside these timings. [Raw step entries](../../evaluation/20261017/effective-step-timings.json) and [summary](../../evaluation/20261017/completed_comparison_summary.json) are retained.

`02-history.svg` is a separate earlier four-H20 experiment at global batch 128 = 4 × 32 × 1. [history.json](history.json) retains the selected screening points and their original report paths. Screening speedups use a frozen 3.7348 s reference; the final full check remeasured reference/candidate at 3.7368 / 2.1704 s (1.7218×). The final point includes intervening changes and remeasurement; it is not one isolated optimization's contribution.

The [final historical check](historical-final-check.json) passed the short-trajectory statistical check across three seeds (10, 11 and 12) and seven endpoints. The strict numerical check did not pass: the report records 2,946 unresolved comparisons and eight dtype changes. These results support the reported short-trajectory statistical verdict, but do not establish bitwise equivalence or equivalent quality after full training.

Regenerate SVG/PNG figures using `python docs/performance-views/render.py` with matplotlib.

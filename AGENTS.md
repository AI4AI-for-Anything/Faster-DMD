# Repository instructions

- Preserve global batch 2,048 for the documented SD1.5 continuation. Default to eight GPUs; adapt microbatch and accumulation rather than reducing the effective batch. Keep the 1:10 G/guidance update cadence and scheduler world size 64.
- Distinguish the four-GPU global-128 historical benchmark from the eight-GPU global-2048 continuation. Headline timing excludes checkpoint writes. One seed per arm does not establish quality equivalence or lossless acceleration.
- `evaluation/20261017` is archival evidence. Preserve it and record new experiments separately. Do not restart completed GPU jobs for documentation changes. Respect shared-host GPU ownership and locks.
- The accumulation adapter supports SD1.5 DDP only. Do not claim bitwise full-model validation from the CPU toy oracle.

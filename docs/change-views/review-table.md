# Runtime changes

Four unique runtime files changed relative to [upstream DMD2](https://github.com/tianweiy/DMD2/tree/8d8fa55633d47cfb81bbc7a892e7248f9518763f). The original performance changes are summarized below; the later accumulation adapter was used in the completed eight-H20 continuation. Documentation, launchers and evidence archives are excluded from this count.

| File | Initial optimization | Later eight-GPU adaptation |
|---|---|---|
| [`main/sd_unified_model.py`](../../main/sd_unified_model.py) | Compile generator forward; BF16 no-grad generator path | None |
| [`main/sd_guidance.py`](../../main/sd_guidance.py) | Compile fake UNet; BF16 no-grad fake score; reduce activation recomputation; concatenate real/fake bottleneck processing while retaining two classification-head calls | None |
| [`main/train_sd.py`](../../main/train_sd.py) | cuDNN autotuning, DDP bucket views, batched scalar gathers, zero-grad cleanup | Explicit accumulation dispatch, global-batch validation, scheduler clock, completed-update checkpoint numbering, all-rank RNG saving and optimizer audits |
| [`main/accumulation.py`](../../main/accumulation.py) | — | Separate G/D microbatch phases; divide loss once; synchronize final microbatch; clip/step once; reuse detached pre-update generated images |

The cumulative [history](../performance-views/history.json) does not isolate each file's speed contribution. The measured launch command is retained in [`training/h20/run_optimized.sh`](../../training/h20/run_optimized.sh). See the [accumulation validation](../gradient-accumulation.md) for correctness scope.

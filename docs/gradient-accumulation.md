# SD1.5 DDP gradient accumulation

Both completed comparison arms use the same `main/accumulation.py`. The original-precision arm starts from [upstream DMD2](https://github.com/tianweiy/DMD2/tree/8d8fa55633d47cfb81bbc7a892e7248f9518763f) with orchestration-only adaptation, documented in the [original training verification](../evaluation/20261017/original_training_verified.json). This repository includes the optimized UNet paths and the tested accumulation trainer used in the optimized arm.

## Effective-batch semantics

Eight GPUs × microbatch 32 × accumulation 8 preserves the official 64 × 32 global batch 2,048. Guidance updates once per effective iteration; the generator updates every ten iterations. Equal-size microbatch mean losses are divided by the accumulation count exactly once. Accelerator's own accumulation count is one. DDP synchronizes only the last microbatch of each relevant phase, followed by one gradient clip and one optimizer update.

All generator microbatches run before its optimizer step. Guidance receives detached images from before that update, preserving the upstream phase order without retaining generator autograd graphs. G and guidance have separate no-sync scopes. SDXL, denoising and FSDP accumulation are unsupported.

## Scheduler and checkpoints

`--schedule_world_size 64` preserves the official Accelerate scheduler clock: both schedulers advance 64 internal ticks per effective iteration, including generator scheduler advancement on guidance-only turns. The launcher uses the explicit accumulated path (`GRAD_ACCUM >= 2`). Both runs restart Adam and warmup from checkpoint weights, following the official weights-only resume recipe.

Checkpoint filenames count completed updates. All ranks save RNG states; checkpoint 041000 follows exactly 2,000 guidance and 200 generator updates after 039000. The completed audits verify optimizer states, 128,000 scheduler ticks, eight RNG files, finite tensors and sampled rank-replica equality.

## Precision and validation limits

Upstream `--use_fp16` includes a BF16 teacher and selected autocast paths, a FP16 VAE and TF32. The original arm preserves those choices; it is not all-FP32. The optimized arm adds the changes in the [file review](change-views/review-table.md).

Microbatching changes random-number ordering and floating-point reduction order versus native 64-GPU execution. No bitwise full-model equivalence is claimed.

```bash
PYTHONPATH=. torchrun --standalone --nproc_per_node=2 tests/test_sd_accumulation.py
```

A CPU test compares gradient accumulation against full-batch updates in a small model. It checks accumulation 1/2/8, generator/guidance turns, pre-update image reuse, all pre-clip gradients, clipped parameters, Adam moments/steps and scheduler counts at FP64 rtol/atol 1e-10. [Both ranks passed](../evaluation/20261017/cpu-ddp-test.log). This checks the accumulation and update logic; it does not establish bitwise equivalence for the full SD1.5 model. A real eight-H20 20-step probe and the completed 2,000-step checkpoint audit passed; see [original training verification](../evaluation/20261017/original_training_verified.json) and [optimized audit](../evaluation/20261017/optimized-training-audit.json).

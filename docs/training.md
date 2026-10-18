# Environment, assets and reproduction

## Install the measured runtime

Use Linux, Python 3.10 and an NVIDIA driver compatible with CUDA 12.4. The measured environment used PyTorch 2.4.1+cu124, torchvision 0.19.1+cu124, diffusers 0.28.2, transformers 4.40.2, accelerate 0.23.0 and Triton 3.0.0. The package snapshot is recorded in `training/h20/requirements.lock.txt`. Root `requirements.txt` is the original upstream environment, not the measured optimized runtime.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r training/h20/requirements.lock.txt \
  --extra-index-url https://download.pytorch.org/whl/cu124
```

Ensure a working CUDA toolkit/compiler toolchain for torch.compile. Keep compilation caches on writable local storage. Run a short probe to check compilation and memory use on your machine. Installation on a fresh machine has not been tested.

## Prepare public assets once

Use only public assets from [upstream DMD2](https://github.com/tianweiy/DMD2). The [download script](../scripts/download_sdv15.sh) specifies the LAION captions, COCO captions, real-latent LMDB and COCO val2014 reference images. Review its destination before running it. Extract the latent archive and set `DMD_REAL_LMDB` to the actual LMDB directory.

Place a complete local Stable Diffusion 1.5 Diffusers model at `$DMD_ASSETS/sd15` and the official [39k checkpoint directory](https://huggingface.co/tianweiy/DMD2/tree/main/model/sdv1.5/laion6.25_sd_baseline_8node_guidance1.75_lr1e-5_seed10_dfake10_from_scratch_fid9.28_checkpoint_model_039000) under `$DMD_ASSETS` with its original directory name. The prompt file must be `$DMD_ASSETS/captions_laion_score6.25.pkl`. Downloads must finish before launching: training uses offline Hugging Face and W&B modes.

```bash
export DMD_ASSETS=/path/to/assets
export DMD_REAL_LMDB=/path/to/sd_vae_latents_laion_500k_lmdb
export DMD_OUTPUT=/path/to/new-output
bash scripts/train_faster_sd15.sh
```

The launcher requires a new output directory and defaults to the measured eight-GPU, 2,000-effective-iteration continuation. For a 20-iteration memory and initialization probe, set `TRAIN_ITERS=39020` and use a separate output directory. This is the ending iteration number: training starts at 39,000 and stops at 39,020. Check available GPUs and shared-host locks before launching.

## Memory and reproduction

Keep `NPROC × MICROBATCH × GRAD_ACCUM = 2048`. Defaults: 8 × 32 × 8. The portable launcher uses SD1.5 multi-GPU DDP and explicit accumulation (at least 2); SDXL/FSDP/denoising and single-GPU operation are not validated by this recipe. The launcher sets `--schedule_world_size 64` to preserve the upstream learning-rate schedule. This setting controls scheduler stepping; it does not launch 64 processes.

On smaller-memory cards, try a smaller microbatch with proportionally more accumulation (e.g. eight GPUs × 16 × 16). This is a candidate configuration requiring a real probe, not a measured 80 GB guarantee. The original-precision eight-H20 accumulated run peaked at 66.382 GiB allocated, which does not establish total reserved memory or optimized-run fit. Native eight-GPU batch 256 without accumulation ran out of memory.

Read [accumulation semantics and validation](gradient-accumulation.md). The [archived optimized launcher](../training/h20/run_optimized.sh) records the original H20 command with machine-specific paths replaced by placeholders. The portable launcher makes asset and output paths configurable while retaining the measured training settings. For FID, follow the [matched protocol](h20-results.md#identical-fid-protocol), including epsilon prediction and evaluation batch 10.

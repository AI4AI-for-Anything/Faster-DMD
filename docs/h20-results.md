# Completed H20 continuation and FID comparison

The baseline and optimized runs each continued from the official 39k weights to 41k on eight H20 GPUs, completing on 2026-10-17. The official checkpoints were evaluated as reference points; their training was not timed here. The [summary](../evaluation/20261017/completed_comparison_summary.json) and [full FID results](../evaluation/20261017/fid_comparison.json) retain exact values.

| Checkpoint | FID-30k ↓ | Mean effective step (s) | Logged step time |
|---|---:|---:|---:|
| Official 39k | 9.2758 | — | — |
| Original precision + accumulation, 41k | 8.6536 | 29.497 | 16.39 h |
| Faster-DMD + accumulation, 41k | 8.7030 | 17.627 | 9.79 h |
| Official released 41k | 8.3429 | — | — |

The **1.67×** speedup compares the baseline and optimized continuation runs, using all 2,000 recorded effective iterations per run. Each effective iteration includes one guidance update; every tenth iteration also includes a generator update. These logs exclude checkpoint writes; this is not a ratio of entire end-to-end wall times. The optimized FID is 0.0494 higher than the baseline (lower is better). There is one training seed per arm, so no statistical equivalence claim. Both continuations trail the released official 41k model; the gap cannot be attributed solely to precision.

## Matched training settings

| Setting | Both continuation arms |
|---|---|
| Model / initialization | SD1.5 DMD2, official 039000 weights |
| Hardware | 8 H20 GPUs, approximately 96 GB each |
| Batch | 32 per microbatch × 8 accumulation × 8 GPUs = 2,048 |
| Updates | 2,000 guidance / 200 generator, 39k→41k |
| Seed | 10 |
| Learning rates | G = guidance = 5e-7 |
| Optimizer | AdamW, betas (0.9, 0.999), weight decay 0.01, clip norm 10 |
| Schedule | 500 internal warmup ticks; 64 scheduler ticks per effective iteration |
| Resolution | 512 image / 64 latent |
| Guidance | Real 1.75 / fake 1.0 |
| GAN losses | Generator classification 1e-3 / guidance classification 1e-2; diffusion GAN max timestep 1,000 |
| Public training assets | 3,023,571 LAION prompt records; 480,000 real latent records |
| Resume semantics | Weights-only resume; restart Adam and warmup, as in the official recipe |

[Alignment audit](../evaluation/20261017/comparison_alignment.json) · [original arguments](../training/h20/original-args.json) · [optimized command](../training/h20/run_optimized.sh).

The original model files retain upstream precision and operations, with a training-orchestration adapter. The baseline retains the upstream mixed-precision paths: BF16 teacher/autocast, FP16 VAE and TF32 under `--use_fp16`. Both runs use the same gradient accumulation implementation. [Checkpoint verification](../evaluation/20261017/original_training_verified.json) and [optimized audit](../evaluation/20261017/optimized-training-audit.json) confirm completed update counts, scheduler ticks, RNG files and finite tensors.

## Identical FID protocol

- 30,000 COCO test prompts; seed 10; batch 10; one-step SD1.5 sampling at timestep 999, **epsilon→x0 (`pred_eps=True`)**.
- FP32 / TF32 inference; generate 512-pixel uint8 images; center crop and Lanczos resize to 256 for the bundled cleanFID Inception-v3 implementation.
- 40,504 COCO val2014 reference images. Eight shards of 3,750 samples, with global batch offsets 0, 375, …, 2,625.
- Shared worker and sampler-function hashes and shard offsets were checked. An 80-image serial/distributed check was byte-identical. Historical asset provenance is not a complete per-run cryptographic snapshot of every shared asset.

[Prior three-checkpoint audit](../evaluation/20261017/prior-three-checkpoint-protocol-audit.json) · [original-arm evaluation plan](../evaluation/20261017/original-fid-run-plan.json) · [original-arm sampling parity and result](../evaluation/20261017/original-fid-summary.json).

## Official references

- [Upstream SD1.5 documentation](https://github.com/tianweiy/DMD2/blob/8d8fa55633d47cfb81bbc7a892e7248f9518763f/experiments/sdv1.5/README.md): official reported FID 9.28 at 39k and 8.35 at 41k, with approximately 25 h and a further 2 h on its 64-GPU recipe. Those are upstream reports, not measured H20 times here.
- [Official continuation recipe](../experiments/sdv1.5/laion6.25_sd_baseline_8node_guidance1.75_lr5e-7_seed10_dfake10_diffusion1000_gan1e-3_noode_resume_fixdata.sh): batch, losses, learning rates and weights-only resume.
- [Official 39k checkpoint](https://huggingface.co/tianweiy/DMD2/tree/main/model/sdv1.5/laion6.25_sd_baseline_8node_guidance1.75_lr1e-5_seed10_dfake10_from_scratch_fid9.28_checkpoint_model_039000).
- [Official 41k checkpoint](https://huggingface.co/tianweiy/DMD2/tree/main/model/sdv1.5/laion6.25_sd_baseline_8node_guidance1.75_lr5e-7_seed10_dfake10_diffusion1000_gan1e-3_resume_fid8.35_checkpoint_model_041000).
- [Public asset download script](../scripts/download_sdv15.sh) and [upstream evaluator](../main/test_folder_sd.py). For this comparison use batch 10, 30,000 samples, seed 10 and `pred_eps=True`; upstream defaults alone do not reproduce this protocol.

Machine-specific paths in archived JSON files and shell commands have been replaced with placeholders. They are not portable launchers; use the [training guide](training.md) for a new host.

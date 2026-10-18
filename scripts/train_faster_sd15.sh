#!/usr/bin/env bash
set -euo pipefail
: "${DMD_ASSETS:?Set the public asset directory}"
: "${DMD_REAL_LMDB:?Set the public real-latent LMDB directory}"
: "${DMD_OUTPUT:?Set a new output directory}"
NPROC=${NPROC:-8}; MICROBATCH=${MICROBATCH:-32}; GRAD_ACCUM=${GRAD_ACCUM:-8}
for value in "$NPROC" "$MICROBATCH" "$GRAD_ACCUM"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo 'Batch settings must be positive integers' >&2; exit 2; }
done
(( NPROC >= 2 && GRAD_ACCUM >= 2 && NPROC * MICROBATCH * GRAD_ACCUM == 2048 )) || {
  echo 'Require multi-GPU DDP, accumulation >= 2, and global batch 2048' >&2; exit 2;
}
[[ "$DMD_ASSETS" = /* && "$DMD_REAL_LMDB" = /* && "$DMD_OUTPUT" = /* ]] || {
  echo 'Use absolute asset and output paths' >&2; exit 2;
}
[[ ! -e "$DMD_OUTPUT" ]] || { echo 'Output already exists; choose a new directory' >&2; exit 2; }
for asset in "$DMD_ASSETS/sd15/model_index.json" "$DMD_ASSETS/captions_laion_score6.25.pkl" "$DMD_REAL_LMDB/data.mdb" "$DMD_ASSETS/laion6.25_sd_baseline_8node_guidance1.75_lr1e-5_seed10_dfake10_from_scratch_fid9.28_checkpoint_model_039000/pytorch_model.bin" "$DMD_ASSETS/laion6.25_sd_baseline_8node_guidance1.75_lr1e-5_seed10_dfake10_from_scratch_fid9.28_checkpoint_model_039000/pytorch_model_1.bin"; do
  [[ -f "$asset" ]] || { echo "Missing asset: $asset" >&2; exit 2; }
done
mkdir -p "$DMD_OUTPUT"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR="$DMD_OUTPUT/inductor" TRITON_CACHE_DIR="$DMD_OUTPUT/triton"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python}" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" -m main.train_sd \
 --model_id "$DMD_ASSETS/sd15" --output_path "$DMD_OUTPUT/output" --cache_dir "$DMD_OUTPUT/cache" --log_path "$DMD_OUTPUT/logs" \
 --train_iters "${TRAIN_ITERS:-41000}" --log_iters 500 --batch_size "$MICROBATCH" --gradient_accumulation_steps "$GRAD_ACCUM" --target_global_batch_size 2048 --schedule_world_size 64 \
 --seed 10 --resolution 512 --latent_resolution 64 --initialie_generator --use_fp16 \
 --generator_lr 5e-7 --guidance_lr 5e-7 --max_grad_norm 10 --warmup_step 500 \
 --real_guidance_scale 1.75 --fake_guidance_scale 1 --dfake_gen_update_ratio 10 \
 --gradient_checkpointing --cls_on_clean_image --gen_cls_loss --gen_cls_loss_weight 1e-3 --guidance_cls_loss_weight 1e-2 \
 --diffusion_gan --diffusion_gan_max_timestep 1000 --num_workers 4 --grid_size 2 --log_loss --wandb_iters 50 \
 --wandb_project dmd-validation --wandb_name "faster-dmd-global2048" \
 --ckpt_only_path "$DMD_ASSETS/laion6.25_sd_baseline_8node_guidance1.75_lr1e-5_seed10_dfake10_from_scratch_fid9.28_checkpoint_model_039000" \
 --train_prompt_path "$DMD_ASSETS/captions_laion_score6.25.pkl" \
 --real_image_path "$DMD_REAL_LMDB"

#!/bin/bash
set -euo pipefail
W=/path/to/optimized-run
D=/path/to/optimized-runtime
A=$D/assets
phase=${1:?probe or full}
steps=41000
if [[ $phase == probe ]]; then steps=39020; fi
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
export TORCHINDUCTOR_CACHE_DIR=$D/inductor TRITON_CACHE_DIR=$D/triton
export PYTHONUNBUFFERED=1 PYTHONPATH=$W/source
cd "$W/source"
exec "$D/env/bin/python" -m torch.distributed.run --standalone --nproc_per_node=8 -m main.train_sd \
 --model_id "$A/sd15" --output_path "$W/$phase/output" --cache_dir "$D/$phase/cache" --log_path "$W/$phase/logs" \
 --train_iters "$steps" --log_iters 500 --batch_size 32 --gradient_accumulation_steps 8 --target_global_batch_size 2048 --schedule_world_size 64 \
 --seed 10 --resolution 512 --latent_resolution 64 --initialie_generator --use_fp16 \
 --generator_lr 5e-7 --guidance_lr 5e-7 --max_grad_norm 10 --warmup_step 500 \
 --real_guidance_scale 1.75 --fake_guidance_scale 1 --dfake_gen_update_ratio 10 \
 --gradient_checkpointing --cls_on_clean_image --gen_cls_loss --gen_cls_loss_weight 1e-3 --guidance_cls_loss_weight 1e-2 \
 --diffusion_gan --diffusion_gan_max_timestep 1000 --num_workers 4 --grid_size 2 --log_loss --wandb_iters 50 \
 --wandb_project dmd-validation --wandb_name "h20-8gpu-$phase-global2048" \
 --ckpt_only_path "$A/laion6.25_sd_baseline_8node_guidance1.75_lr1e-5_seed10_dfake10_from_scratch_fid9.28_checkpoint_model_039000" \
 --train_prompt_path "$A/captions_laion_score6.25.pkl" \
 --real_image_path "$A/sd_vae_latents_laion_500k_lmdb"

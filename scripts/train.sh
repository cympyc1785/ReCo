#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ========== Env (W&B) ==========
# WANDB_API_KEY는 ~/.netrc의 로그인 정보 사용
export WANDB_API_KEY=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
export WANDB_PROJECT="QA_GEN_JOINT"
export WANDB_MODE="online"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export GPU_NUM=4
export NUM_NODES=1              # 1 or 2
export TOKENIZERS_PARALLELISM=false  # Linux/macOS

# ========== Config ==========
DIT_PATH="checkpoints/Wan2.1-VACE-1.3B/diffusion_pytorch_model.safetensors,\
checkpoints/Wan2.1-VACE-1.3B/models_t5_umt5-xxl-enc-bf16.pth,\
checkpoints/Wan2.1-VACE-1.3B/Wan2.1_VAE.pth"

# training
world_size=$((GPU_NUM * NUM_NODES))
training_strategy='deepspeed_stage_2'           # or ddp

# logs dir
OUTPUT_PATH="./all_results/mix_data_output"
script_name=$(basename "$0" .sh)
log_dir="./all_results/logs/$script_name"
run_name="train_video_edit"
log_file="$log_dir/$run_name/train.log"
mkdir -p "$log_dir/$run_name"


# ========== Run ==========
MASTER_ADDR="${MASTER_ADDR:-localhost:29500}"
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=$NUM_NODES \
    --nproc_per_node=$GPU_NUM \
    --rdzv_id=distributed_alldata \
    --rdzv_backend=c10d \
    --rdzv-endpoint=$MASTER_ADDR \
    scripts/train.py --train_architecture all_lora \
    --num_nodes $NUM_NODES \
    --training_strategy $training_strategy \
    --every_n_train_steps 250 \
    --log_video_steps 250 \
    --train_batch_size 1 \
    --world_size $world_size \
    --dataloader_num_workers 1 \
    --output_path "$OUTPUT_PATH" \
    --dit_path "$DIT_PATH" \
    --max_epochs 10 \
    --learning_rate 5e-5 \
    --accumulate_grad_batches 2 \
    --log_every_n_steps 1 \
    --use_gradient_checkpointing \
    --lora_rank "128" --lora_alpha "128" \
    --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
    --project_name ReCo \
    --run_name train_run2_contrast_attnscore  2>&1 | tee $log_file
    # 如需加载已有 LoRA：
    # --pretrained_lora_path "all_results/train_runs/.../checkpoints/xxx.ckpt"
    # --resume_ckpt_folder "all_results/train_runs/..../checkpoints/wan_deepspeed_folder-epoch=0-step=7000.ckpt" 2>&1 | tee $log_file



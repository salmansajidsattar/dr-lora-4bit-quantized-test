#!/bin/bash
# OLMoE-1B-7B-0924 Training Script
# Supports two modes: Baseline LoRA / ALOE (Adaptive LoRA with Expert-aware Grow)
#
# Usage:
#   USE_ALOE=false bash train_olmoe.sh   # Baseline LoRA
#   USE_ALOE=true  bash train_olmoe.sh   # ALOE

set -e

# ==================== Training Mode ====================
USE_ALOE=false

echo "======================================================================"
echo "OLMoE-1B-7B-0924 Training"
echo "Mode: $([ "$USE_ALOE" = true ] && echo 'ALOE' || echo 'Baseline LoRA')"
echo "======================================================================"

# ==================== Path Configuration ====================
MODEL_PATH="/path/to/OLMoE-1B-7B-0924"
TRAIN_SCRIPT="/path/to/open-instruct/open_instruct/finetune_colm.py"
OUTPUT_BASE="/path/to/output"
LOG_DIR="/path/to/logs"
METAMATHQA_DATA="/path/to/data/metamathqa_gsm_tulu_gsm8k.jsonl"

export PYTHONPATH=/path/to/open-instruct:$PYTHONPATH
export NCCL_TIMEOUT=1800
export TORCH_NCCL_BLOCKING_WAIT=0

# ==================== Shared Training Hyperparameters ====================
BATCH_SIZE=8
GRAD_ACCUM=8
LR=2e-5
EPOCHS=1
MAX_SEQ_LEN=512

# ── LoRA Common Parameters ──
LORA_ALPHA=16
LORA_SCALING_FACTOR=2
LORA_DROPOUT=0.1
LORA_EXPERTS_ONLY=True
LORA_TOPK_ONLY=True
LORA_TARGET_MLP_PROJS="up_proj down_proj"

# ── Router Schedule ──
# False = unfreeze router after warmup, train together with LoRA
# True  = router frozen throughout, only LoRA is trained
FREEZE_MOE_ROUTER=False

# ── Checkpoint / Logging ──
CHECKPOINTING_STEPS=300
KEEP_LAST_N_CHECKPOINTS=3
LOGGING_STEPS=5

# ==================== ALOE-specific Parameters ====================
LORA_RANK=8             # AdaLoRA initial rank (starting point for grow)
ADALORA_MAX_RANK=32     # Max rank upper bound per module
ADALORA_TARGET_RANK=16  # Target average rank at end of grow
ADALORA_GROW_INTERVAL=50
ADALORA_BETA=0.9
ADALORA_USAGE_DECAY=0.9

# ==================== Baseline-specific Parameters ====================
BASELINE_LORA_RANK=32   # Fixed rank for Baseline, corresponds to ALOE target rank

# ==================== Pre-flight Checks ====================
mkdir -p "$OUTPUT_BASE" "$LOG_DIR"

if [ ! -f "$METAMATHQA_DATA" ]; then
    echo "[Error] Dataset file not found: $METAMATHQA_DATA"
    exit 1
fi
echo "Dataset: $METAMATHQA_DATA  ✓"

# ==================== Common Arguments (shared by both modes) ====================
COMMON_ARGS="
    --model_name_or_path $MODEL_PATH
    --use_flash_attn True
    --max_seq_length $MAX_SEQ_LEN
    --preprocessing_num_workers 16
    --per_device_train_batch_size $BATCH_SIZE
    --gradient_accumulation_steps $GRAD_ACCUM
    --gradient_checkpointing True
    --learning_rate $LR
    --lr_scheduler_type linear
    --warmup_ratio 0.03
    --weight_decay 0.0
    --num_train_epochs $EPOCHS
    --with_tracking True
    --report_to wandb
    --logging_steps $LOGGING_STEPS
    --model_revision main
    --dataset_mixer_list $METAMATHQA_DATA 1.0
    --dataset_mixer_list_splits train
    --dataset_skip_cache True
    --packing False
    --checkpointing_steps $CHECKPOINTING_STEPS
    --keep_last_n_checkpoints $KEEP_LAST_N_CHECKPOINTS
    --push_to_hub False
    --try_launch_beaker_eval_jobs False
    --use_lora True
    --lora_alpha $LORA_ALPHA
    --lora_scaling_factor $LORA_SCALING_FACTOR
    --lora_dropout $LORA_DROPOUT
    --lora_experts_only $LORA_EXPERTS_ONLY
    --lora_target_mlp_projs $LORA_TARGET_MLP_PROJS
    --lora_topk_only $LORA_TOPK_ONLY
    --freeze_moe_router $FREEZE_MOE_ROUTER
    --clean_checkpoints_at_end False
    --chat_template_name tulu
    --add_bos True"  # Required for OLMoE; not needed for Qwen

# ==================== Mode Branch: append mode-specific args ====================
if [ "$USE_ALOE" = true ]; then
    OUTPUT_DIR="${OUTPUT_BASE}/olmoe_gsm8k_metamathqa_aloe_r${LORA_RANK}_target${ADALORA_TARGET_RANK}"
    LOG_FILE="${LOG_DIR}/olmoe_gsm8k_metamathqa_aloe_r${LORA_RANK}_target${ADALORA_TARGET_RANK}.log"

    MODE_ARGS="
    --lora_rank $LORA_RANK
    --use_adalora True
    --adalora_max_rank $ADALORA_MAX_RANK
    --adalora_target_rank $ADALORA_TARGET_RANK
    --adalora_grow_interval $ADALORA_GROW_INTERVAL
    --adalora_beta $ADALORA_BETA
    --adalora_usage_decay $ADALORA_USAGE_DECAY"

else
    OUTPUT_DIR="${OUTPUT_BASE}/olmoe_gsm8k_metamathqa_lora_baseline_r${BASELINE_LORA_RANK}"
    LOG_FILE="${LOG_DIR}/olmoe_gsm8k_metamathqa_lora_baseline_r${BASELINE_LORA_RANK}.log"

    MODE_ARGS="
    --lora_rank $BASELINE_LORA_RANK
    --use_adalora False"
fi

# ==================== Print Configuration Summary ====================
echo ""
echo "Output dir : $OUTPUT_DIR"
echo "Batch size : $BATCH_SIZE x $GRAD_ACCUM = effective $((BATCH_SIZE * GRAD_ACCUM))"
echo "LR         : $LR    Epochs: $EPOCHS    Max seq: $MAX_SEQ_LEN"
echo "Router     : freeze_moe_router=$FREEZE_MOE_ROUTER"
echo "LoRA proj  : $LORA_TARGET_MLP_PROJS"
if [ "$USE_ALOE" = true ]; then
    echo "ALOE rank  : $LORA_RANK → target $ADALORA_TARGET_RANK (max $ADALORA_MAX_RANK), grow every $ADALORA_GROW_INTERVAL steps"
else
    echo "Baseline   : rank=$BASELINE_LORA_RANK (fixed)"
fi
echo ""

read -p "Start training? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

# ==================== Launch Training ====================
echo "======================================================================"
echo "Starting training..."
echo "======================================================================"

python -u -m accelerate.commands.launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes 4 \
    --use_deepspeed \
    --deepspeed_config_file /path/to/configs/ds_configs/stage2_accelerate.conf \
    $TRAIN_SCRIPT \
    --output_dir $OUTPUT_DIR \
    $COMMON_ARGS \
    $MODE_ARGS \
    2>&1 | tee -a "$LOG_FILE"

echo ""
echo "======================================================================"
echo "Training complete! Model saved to: $OUTPUT_DIR"
echo "======================================================================"
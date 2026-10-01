#!/bin/bash
# Short DR-LoRA test run on 4-bit OLMoE-1B-7B, one Kaggle T4.
# Paper settings (rank 8 -> 16, max 32, grow fraction 0.1, beta 0.9,
# lr 2e-5, LoRA on up/down) except: few steps, growth every 5 steps,
# batch 4, 4-bit NF4 + fp16.
# Usage:  !bash dr-lora/kaggle/run_test_4bit.sh
# Longer: !STEPS=300 GROW_EVERY=50 bash dr-lora/kaggle/run_test_4bit.sh
set -euo pipefail

WORK=${WORK:-/kaggle/working}
MODEL=${MODEL:-allenai/OLMoE-1B-7B-0924}
STEPS=${STEPS:-20}
GROW_EVERY=${GROW_EVERY:-5}
N_SAMPLES=${N_SAMPLES:-2000}
DATA=${DATA:-$WORK/data/metamathqa_gsm8k_${N_SAMPLES}.jsonl}
OUT=${OUT:-$WORK/out_drlora_4bit}
LOG=${LOG:-$WORK/train_test.log}
LAUNCH=${LAUNCH:-"--num_processes 1 --mixed_precision fp16"}
OPTIM_8BIT=${OPTIM_8BIT:-True}      # bitsandbytes 8-bit AdamW (GPU only)
SEQ_LEN=${SEQ_LEN:-512}

export HF_HOME=${HF_HOME:-/kaggle/tmp/hf}   # 14 GB model, off /kaggle/working
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$WORK/open-instruct"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
mkdir -p "$HF_HOME"
rm -rf "$OUT"

# GPU memory every 5 s, for the check script
MEM_PID=""
if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 -l 5 \
        > "$WORK/gpu_mem.csv" &
    MEM_PID=$!
fi

cd "$WORK/open-instruct"
accelerate launch $LAUNCH open_instruct/finetune_colm.py \
    --model_name_or_path "$MODEL" --tokenizer_name_or_path "$MODEL" \
    --dataset_mixer_list "$DATA" 1.0 --dataset_mixer_list_splits train \
    --dataset_skip_cache True --chat_template_name tulu --add_bos True \
    --use_flash_attn False --max_seq_length "$SEQ_LEN" --preprocessing_num_workers 2 \
    --per_device_train_batch_size 4 --gradient_accumulation_steps 1 \
    --learning_rate 2e-5 --lr_scheduler_type linear --warmup_ratio 0.1 \
    --weight_decay 0.0 --max_train_steps "$STEPS" --logging_steps "$GROW_EVERY" \
    --seed 42 --low_cpu_mem_usage True --output_dir "$OUT" \
    --clean_checkpoints_at_end False --with_tracking False --push_to_hub False \
    --try_launch_beaker_eval_jobs False --try_auto_save_to_beaker False \
    --add_seed_and_date_to_exp_name False --fused_optimizer False \
    --use_qlora True --use_8bit_optimizer "$OPTIM_8BIT" --gradient_checkpointing True \
    --use_lora True --lora_experts_only True --lora_target_mlp_projs up_proj down_proj \
    --lora_topk_only True --lora_dropout 0.1 --lora_scaling_factor 2 \
    --freeze_moe_router False --use_adalora True --lora_rank 8 \
    --adalora_target_rank 16 --adalora_max_rank 32 \
    --adalora_grow_interval "$GROW_EVERY" --adalora_beta 0.9 \
    --adalora_usage_decay 0.9 --adalora_module_grow_frac 0.1 \
    2>&1 | tee "$LOG"
STATUS=${PIPESTATUS[0]}

[ -n "$MEM_PID" ] && kill "$MEM_PID" 2>/dev/null || true
echo "training exit code $STATUS"
exit "$STATUS"

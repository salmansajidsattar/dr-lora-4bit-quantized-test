# Running DR-LoRA in 4-bit on Kaggle

This copy of the official DR-LoRA code (github.com/gz-d/dr-lora) is changed only so that it trains on a **4-bit** model and runs on Kaggle T4s.

## Changes to `finetune/finetune.py`

Every change is marked `[4-bit patch]`. DR-LoRA's growth method is untouched.

1. **4-bit loading.** `--use_qlora True` now loads the model in 4-bit NF4. Before this patch the flag existed but did nothing.
   - Routers (`mlp.gate`) and `lm_head` stay full precision.
   - `prepare_model_for_kbit_training` is applied.
2. **dtype follows the GPU.** bf16 on Ampere or newer, fp16 on T4, fp32 on CPU. The original hard-coded bf16.
3. **8-bit optimizer.** `--use_8bit_optimizer True` now uses bitsandbytes AdamW8bit (it also did nothing before).
4. **Finding the experts.** Expert layers are found when they are `Linear4bit` too, not only `Linear`.
5. **No final merge.** With `--use_qlora`, the step that merges LoRA into the model is skipped, because merging into 4-bit weights isn't exact. The LoRA adapter is still saved.

## Steps on Kaggle

Notebook settings: Accelerator **GPU T4 x2**, Internet **On**.

```
!git clone https://github.com/<your-username>/<your-repo>.git /kaggle/working/dr-lora
!bash /kaggle/working/dr-lora/kaggle/setup.sh
!bash /kaggle/working/dr-lora/kaggle/run_test_4bit.sh
!python /kaggle/working/dr-lora/kaggle/check_run.py
```

- **`setup.sh`** does four things:
  - installs pinned packages (`kaggle/requirements.txt`);
  - clones open-instruct at commit `d2cf8a90d` (transformers 4.x);
  - copies `finetune.py` into it;
  - writes 2,000 MetaMathQA (GSM8K) examples.
- **`run_test_4bit.sh`** trains OLMoE-1B-7B in 4-bit:
  - 20 steps on one T4;
  - growth every 5 steps, so 3 growth events.
- **`check_run.py`** prints PASS or FAIL for:
  - 4-bit load, routing and rank growth;
  - loss and the saved adapter;
  - GPU memory and time per step.

A longer run reuses the same script with environment variables:

```
!STEPS=300 GROW_EVERY=50 bash /kaggle/working/dr-lora/kaggle/run_test_4bit.sh
!MIN_EVENTS=5 python /kaggle/working/dr-lora/kaggle/check_run.py
```

## Why transformers 4.57.6 (not 5.x)

In transformers 5, OLMoE stores its 64 experts as one combined tensor. DR-LoRA attaches LoRA to each expert's own `up_proj` / `down_proj` layer, so on 5.x it finds no experts. Every version in `requirements.txt` comes from open-instruct's own lockfile at `d2cf8a90d`.

## Known differences between DR-LoRA's code and paper (not changed here)

- **Routing frequency:** the code counts hard top-k picks; the paper uses routing weights.
- **Rank importance:** the code adds the A-side and B-side terms; the paper multiplies them.
- **Growth quota:** the code counts the quota per expert but checks the target per module. Layers therefore reach their target early and can overshoot it slightly.

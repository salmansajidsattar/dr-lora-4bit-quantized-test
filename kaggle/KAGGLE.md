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
5. **No final merge.** With `--use_qlora`, the step that merges LoRA into the model is skipped, because merging into 4-bit weights isn't exact. The LoRA adapter is saved, and the trained router is saved as `router_state_dict.pt`.
6. **Integer GPU index.** `device_map` uses `torch.cuda.current_device()`; accelerate 1.12 crashes on a device with no index.
7. **Resume.** Resuming loads checkpoints non-strictly, because the frozen 4-bit base weights' quantization constants can't be loaded back strictly. LoRA, router, optimizer, scheduler and DR-LoRA state are all restored.

## Bug fixes to the official code (also in `finetune/finetune.py`)

1. **Router top-k (`[fix]`).** The routing hook sits on `mlp.gate`, a plain `Linear` that does not store top-k, so the code fell back to **2**. OLMoE actually routes **top-8** (Qwen1.5-MoE: top-4). This affected:
   - the routing frequency f, which counted only 2 experts per token;
   - which experts' LoRA gradients were kept: some experts that were really used got zero gradient.

   The hook now reads top-k from the MoE block (`mlp.top_k`).
2. **Hidden logs (`[4-bit patch]`).** On Kaggle, some library sets up logging first at WARNING level, so every INFO line was hidden: loss, rank growth and "Training finished". The root logger is now set to INFO.

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

## Full run (about 9–10 h, both T4s) and GSM8K evaluation

**Training: the paper's schedule at batch 8 (4 per GPU, 2 T4s).**
- Same as the paper: 3,750 steps, growth every 200 steps after 3% warm-up (18 events), lr 2e-5, rank 8 → 16, max rank 32.
- Different: batch 8 instead of 48, so it sees 30,000 MetaMathQA examples (one pass) instead of 180,000.
- Set the notebook accelerator to **GPU T4 x2**. `NGPU=1` gives the old 1-GPU run (batch 4, 15,000 examples).
- A checkpoint is saved every 250 steps; running the script again resumes from the latest one.

Test both GPUs first (20 steps, a few minutes); `check_run.py` then also checks "2 GPUs in sync" and prints the time per step:

```
!NGPU=2 bash /kaggle/working/dr-lora-4bit-quantized-test/kaggle/run_test_4bit.sh
!python /kaggle/working/dr-lora-4bit-quantized-test/kaggle/check_run.py
```

The run takes about 9 hours, so use **Save Version → Save & Run All** (runs in the background, up to 12 h). Do not run it in an interactive session: when that session ends, `/kaggle/working` is deleted and the trained adapter is lost. Only a saved version keeps its files (Output tab). Notebook cells:

```
!git clone https://github.com/<you>/<repo>.git /kaggle/working/dr-lora-4bit-quantized-test
!bash /kaggle/working/dr-lora-4bit-quantized-test/kaggle/setup.sh
!bash /kaggle/working/dr-lora-4bit-quantized-test/kaggle/run_full_4bit.sh
!OUT=/kaggle/working/out_drlora_4bit_full LOG=/kaggle/working/train_full.log python /kaggle/working/dr-lora-4bit-quantized-test/kaggle/check_run.py
```

**Evaluation** (`eval_gsm8k_4bit.py`) matches DR-LoRA's `eval/eval_gsm8k.sh`: lm-eval `gsm8k_cot`, 8-shot, greedy, chat template on, batch 8. It loads the 4-bit base, the saved adapter and the trained router. Run it in a new session, with the training version's output added as input:

```
!pip install -q "lm-eval[hf]==0.4.11"
!python .../kaggle/eval_gsm8k_4bit.py --run_dir /kaggle/input/<notebook>/out_drlora_4bit_full --limit 50   # timing test
!python .../kaggle/eval_gsm8k_4bit.py --run_dir /kaggle/input/<notebook>/out_drlora_4bit_full             # all 1,319
!python .../kaggle/eval_gsm8k_4bit.py --no_adapter                                                         # 4-bit base model
```

It prints `strict-match` (requires "The answer is N") and `flexible-extract` (last number). MetaMathQA teaches the format "The answer is: N", with a colon, which strict-match misses, so flexible-extract is the comparable number.

## Why transformers 4.57.6 (not 5.x)

In transformers 5, OLMoE stores its 64 experts as one combined tensor. DR-LoRA attaches LoRA to each expert's own `up_proj` / `down_proj` layer, so on 5.x it finds no experts. Every version in `requirements.txt` comes from open-instruct's own lockfile at `d2cf8a90d`.

## Known differences between DR-LoRA's code and paper (not changed here)

- **Routing frequency:** the code counts hard top-k picks; the paper uses routing weights.
- **Rank importance:** the code adds the A-side and B-side terms; the paper multiplies them.
- **Growth quota:** the code counts the quota per expert but checks the target per module. Layers therefore reach their target early and can overshoot it slightly. In the full run, growth stops after 9 of the 18 planned events (around step 1,712); `check_run.py` passes when every layer reaches its target.

## Two GPUs (DDP)

DDP averages the LoRA gradients, but DR-LoRA's own state is computed on each GPU from its half of the batch. The code (marked `[2-GPU patch]`) combines it so both GPUs make the same growth decision:
- expert usage counts and active experts are added up over both GPUs every step;
- growth scores are averaged over both GPUs before each growth event and before saving;
- after growth, GPU 0's rank masks are copied to GPU 1, and the log reports `[DDP-sync] ... identical` for the masks and the LoRA weights;
- DDP runs with `find_unused_parameters=True` (unused experts get no gradient) and non-reentrant gradient checkpointing.

## Fixed for fp16 (T4)

- **Rank importance under fp16:** fp16 training scales gradients by the GradScaler factor, and the first steps overflow. The gradient hooks saw these values, so layer 0's scores were NaN at the first growth event and its ranks went to experts 0–18 by index. The hooks now divide by the scale and skip non-finite steps (marked `[fix]`).

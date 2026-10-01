# DR-LoRA: Dynamic Rank LoRA for Fine-Tuning Mixture-of-Experts Models

<p align="center">
  <img src="https://img.shields.io/badge/COLM-2026-4c1.svg" alt="COLM 2026">
  <a href="https://arxiv.org/abs/2601.04823"><img src="https://img.shields.io/badge/arXiv-2601.04823-b31b1b.svg" alt="arXiv"></a>
  <a href="https://github.com/gz-d/dr-lora"><img src="https://img.shields.io/badge/GitHub-gz--d%2Fdr--lora-181717.svg" alt="GitHub"></a>
</p>

> **Accepted at the Conference on Language Modeling (COLM) 2026.**

Official research code for our COLM 2026 paper **DR-LoRA: Dynamic Rank LoRA for Fine-Tuning Mixture-of-Experts Models**, which introduces a growth-based dynamic rank allocation method for parameter-efficient fine-tuning of pretrained Mixture-of-Experts (MoE) language models.

Standard LoRA assigns the same rank to every expert, even though pretrained MoE experts exhibit heterogeneous and task-dependent usage. DR-LoRA starts expert LoRA modules from a small active rank, estimates each expert's demand using routing usage and gradient-based rank importance, and periodically grows the ranks of high-saliency experts. The resulting heterogeneous rank allocation concentrates adaptation capacity on experts that matter most for the downstream task.

> **Research-code notice.** This repository contains the training, data-conversion, and evaluation scripts used for the project. Paths in the shell and evaluation scripts are placeholders and must be updated for your environment. The repository does not currently pin an `open-instruct` commit or provide the DeepSpeed configuration used in the experiments.

## Highlights

- **Expert-aware capacity allocation:** allocates LoRA rank at the level of individual pretrained MoE experts.
- **Growth instead of pruning:** begins with a small active rank and expands capacity as reliable task-specific signals emerge.
- **Routing-aware saliency:** prioritizes experts using both routing demand and gradient-based learning signals.
- **Per-layer greedy allocation:** prevents a small set of layers from absorbing the global rank budget.
- **Fixed final budget:** reaches a target average active rank while allowing heterogeneous expert-level ranks.
- **Broad evaluation:** experiments cover mathematical reasoning, code generation, instruction following, medical QA, machine translation, and legal understanding.

## Method Overview

For expert $i$ in MoE layer $\ell$, DR-LoRA conceptually scores the benefit of additional adaptation capacity as

$$
S_{\ell,i}^{(t)} =
\frac{f_{\ell,i}^{(t)}\,g_{\ell,i}^{(t)}}
{\left(r_{\ell,i}^{(t)}+1\right)^\gamma}.
$$

where:

- $f_{\ell,i}$ measures the expert's routing demand on the target task;
- $g_{\ell,i}$ measures gradient-based rank importance while the expert is active;
- $r_{\ell,i}$ is the expert's current active LoRA rank;
- $\gamma$ discourages rank growth from collapsing onto only a few experts.

Training consists of three stages:

1. **Warmup:** all expert LoRA modules use the initial active rank $r_{\text{init}}$; the router is frozen.
2. **Growth window:** every $T_{\text{grow}}$ optimizer steps, a per-layer quota is greedily assigned to the highest-saliency experts.
3. **Fixed stage:** rank allocation stops before training ends, allowing newly activated dimensions to be optimized.

The released implementation records top-k routing decisions with forward hooks, accumulates gradient-weight sensitivity through hooks on expert LoRA matrices, masks inactive rank dimensions, and saves rank snapshots during training.

## Repository Structure

```text
dr-lora/
├── data/
│   ├── convert_codealpaca_to_tulu.py
│   ├── convert_ledgar_to_mc.py
│   ├── convert_medmcqa_to_tulu.py
│   └── convert_metamathqa_to_tulu.py
├── eval/
│   ├── eval_gsm8k.sh
│   ├── eval_ledgar.py
│   ├── eval_medmcqa.py
│   ├── eval_wmt.py
│   ├── evalplus.sh
│   └── run_eval_medmcqa.sh
├── finetune/
│   ├── finetune.py
│   └── finetune.sh
└── README.md
```

## Supported Models and Benchmarks

The paper evaluates:

| Model | MoE configuration |
|---|---|
| OLMoE-1B-7B | Open pretrained MoE model |
| Qwen1.5-MoE-A2.7B | Qwen MoE model |
| LLaMA-MoE-v1-3.5B (4/16) | LLaMA-based MoE model |

| Domain | Training data used in the project | Evaluation benchmark | Metric |
|---|---|---|---|
| Mathematical reasoning | MetaMathQA (60k examples in the paper; the bundled example can filter GSM8K-related records) | GSM8K | Accuracy, 8-shot |
| Code generation | CodeAlpaca-20K (full dataset) | HumanEval | Pass@1, 0-shot |
| Instruction following | OLMoE SFT Mix (60k examples) | IFEval | Accuracy, 0-shot |
| Medical QA | MedMCQA train split (60k examples) | MedMCQA | Accuracy, 0-shot |
| Machine translation | WMT-DA-Human-Evaluation (60k total; 20k each for en-cs, en-de, and en-zh) | WMT23 | COMET, 0-shot |
| Legal understanding | LEDGAR reformatted as multiple choice (60k examples) | LEDGAR | Accuracy, 0-shot |

The current training code assumes that expert modules are named under `.experts.<id>` and that adapted expert MLP projections use suffixes such as `gate_proj`, `up_proj`, and `down_proj`. Routing hooks explicitly support OLMoE, Qwen1.5-MoE, and compatible architectures whose MoE block returns two-dimensional router logits.

## Installation

### 1. Clone this repository

```bash
git clone https://github.com/gz-d/dr-lora.git
cd dr-lora
```

### 2. Install `open-instruct`

The training script is built on the [AllenAI open-instruct](https://github.com/allenai/open-instruct) training stack and imports its dataset, tokenizer, logging, and checkpoint utilities.

```bash
git clone https://github.com/allenai/open-instruct.git
cd open-instruct
pip install -e .
cd ..
```

The repository does not pin an `open-instruct` revision. Use a revision whose APIs provide the imports used at the top of `finetune/finetune.py`.

### 3. Install additional training dependencies

```bash
pip install deepspeed wandb
pip install flash-attn --no-build-isolation
```

Core packages such as PyTorch, Transformers, Accelerate, PEFT, Datasets, Hugging Face Hub, Rich, and tqdm are normally installed by `open-instruct`.

Optional dependencies:

```bash
# QLoRA or 8-bit optimizer support
pip install bitsandbytes

# Evaluation
pip install lm-eval evalplus vllm pandas scipy unbabel-comet
```

### Reference compute environment

The paper experiments use:

- 4 × NVIDIA L40S GPUs with 48 GB memory each;
- bfloat16 mixed precision;
- DeepSpeed ZeRO-2;
- Flash Attention 2.

Other GPU configurations may work after adjusting batch size, gradient accumulation, sequence length, and tensor parallelism.

## Integrating the Training Script with `open-instruct`

`finetune/finetune.py` is intended to run inside an `open-instruct` checkout. A non-destructive setup is:

```bash
cp dr-lora/finetune/finetune.py \
   open-instruct/open_instruct/finetune_colm.py
```

Then export the source checkout:

```bash
export PYTHONPATH=/path/to/open-instruct:$PYTHONPATH
```

The bundled `finetune/finetune.sh` expects:

```text
/path/to/open-instruct/open_instruct/finetune_colm.py
```

Alternatively, replace `open_instruct/finetune.py` as described in the Python file header and update `TRAIN_SCRIPT` accordingly.

## Data Preparation

All training files passed through `--dataset_mixer_list` should use Tulu-style JSONL records:

```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "dataset": "dataset_name"
}
```

### MetaMathQA for GSM8K

```bash
python data/convert_metamathqa_to_tulu.py \
  --gsm8k_only \
  --output_path /path/to/data/metamathqa_gsm8k_tulu.jsonl
```

Useful options are `--max_samples`, `--hf_cache_dir`, and `--split`.

### CodeAlpaca for HumanEval

```bash
python data/convert_codealpaca_to_tulu.py \
  --output_path /path/to/data/codealpaca_tulu.jsonl
```

The converter downloads `sahil2801/CodeAlpaca-20k` through Hugging Face Datasets.

### MedMCQA

Download the original MedMCQA JSONL split, then run:

```bash
python data/convert_medmcqa_to_tulu.py \
  --input_path /path/to/MedMCQA/train.json \
  --output_path /path/to/data/medmcqa_tulu_train.jsonl
```

Add `--no_exp` to omit the explanation field from the assistant response.

### LEDGAR multiple-choice evaluation data

`convert_ledgar_to_mc.py` converts an existing Tulu-style LEDGAR test file into four-choice questions by sampling three distractor labels for each example:

```bash
python data/convert_ledgar_to_mc.py \
  --input_jsonl /path/to/data/LEDGAR/test.jsonl \
  --output_jsonl /path/to/data/LEDGAR/test_mc.jsonl \
  --seed 42
```

The repository does not include conversion scripts for the OLMoE SFT Mix or WMT training files. Prepare them in the same Tulu JSONL format before training.

## Training

### Configure the launcher

Edit the following placeholders in `finetune/finetune.sh`:

```bash
MODEL_PATH="/path/to/model"
TRAIN_SCRIPT="/path/to/open-instruct/open_instruct/finetune_colm.py"
OUTPUT_BASE="/path/to/output"
LOG_DIR="/path/to/logs"
METAMATHQA_DATA="/path/to/data/metamathqa_gsm8k_tulu.jsonl"
```

Also provide a valid Accelerate/DeepSpeed configuration, for example:

```bash
--deepspeed_config_file /path/to/stage2_accelerate.conf
```

### DR-LoRA mode

The current shell script uses the historical internal name `ALOE`, while the Python CLI uses `--use_adalora`. In this repository, both refer to the **DR-LoRA growth mode**, not the pruning-based AdaLoRA baseline from the paper.

Set:

```bash
USE_ALOE=true
```

and launch:

```bash
bash finetune/finetune.sh
```

The script asks for confirmation before starting. For unattended jobs, remove or bypass the `read -p` confirmation block.

### Fixed-rank LoRA baseline

Set:

```bash
USE_ALOE=false
```

then run the same launcher:

```bash
bash finetune/finetune.sh
```

### Direct DR-LoRA launch example

The following command exposes the main settings without relying on the shell wrapper:

```bash
cd /path/to/open-instruct

python -u -m accelerate.commands.launch \
  --mixed_precision bf16 \
  --num_machines 1 \
  --num_processes 4 \
  --use_deepspeed \
  --deepspeed_config_file /path/to/stage2_accelerate.conf \
  open_instruct/finetune_colm.py \
  --model_name_or_path /path/to/OLMoE-1B-7B-0924 \
  --output_dir /path/to/output/olmoe_gsm8k_drlora \
  --dataset_mixer_list /path/to/data/metamathqa_gsm8k_tulu.jsonl 1.0 \
  --dataset_mixer_list_splits train \
  --dataset_skip_cache True \
  --chat_template_name tulu \
  --add_bos True \
  --use_flash_attn True \
  --max_seq_length 512 \
  --preprocessing_num_workers 16 \
  --per_device_train_batch_size 3 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing True \
  --learning_rate 2e-5 \
  --lr_scheduler_type linear \
  --warmup_ratio 0.03 \
  --weight_decay 0.0 \
  --max_train_steps 3750 \
  --with_tracking True \
  --report_to wandb \
  --logging_steps 5 \
  --checkpointing_steps 300 \
  --keep_last_n_checkpoints 3 \
  --push_to_hub False \
  --try_launch_beaker_eval_jobs False \
  --clean_checkpoints_at_end False \
  --use_lora True \
  --lora_experts_only True \
  --lora_target_mlp_projs up_proj down_proj \
  --lora_topk_only True \
  --lora_dropout 0.1 \
  --lora_scaling_factor 2 \
  --freeze_moe_router False \
  --use_adalora True \
  --lora_rank 8 \
  --adalora_target_rank 16 \
  --adalora_max_rank 32 \
  --adalora_grow_interval 200 \
  --adalora_beta 0.9 \
  --adalora_usage_decay 0.9 \
  --adalora_module_grow_frac 0.1
```

For OLMoE, `--add_bos True` is required by the provided launcher. It may not be needed for Qwen models.

## Important Training Arguments

| Argument | Meaning | Paper default |
|---|---|---:|
| `--lora_rank` | Initial active rank $r_{init}$ in DR-LoRA mode | 8 |
| `--adalora_target_rank` | Target average active rank $r_{target}$ | 16 |
| `--adalora_max_rank` | Physically reserved maximum rank $r_{max}$ | 32 |
| `--adalora_grow_interval` | Optimizer steps between growth events | 200 |
| `--adalora_module_grow_frac` | Maximum fraction of initially free rank grown per module at one event | 0.1 |
| `--adalora_beta` | EMA coefficient used by the hook-based gradient importance accumulator | 0.9 |
| `--adalora_usage_decay` | Decay applied to recorded expert routing usage | 0.9 |
| `--lora_experts_only` | Restrict adapters to expert MLP modules | `True` |
| `--lora_target_mlp_projs` | Expert projections receiving LoRA | `up_proj down_proj` in the launcher |
| `--lora_topk_only` | Zero LoRA gradients for experts not selected during the update window | `True` |
| `--freeze_moe_router` | Keep the router frozen throughout training | `False` in the main setting |
| `--lora_scaling_factor` | Sets effective PEFT alpha as `factor × reserved rank` | 2 |

Additional implementation details:

- The rank penalty exponent is currently fixed to `1.2` in `_adalora_grow_once_per_layer`.
- Growth begins after learning-rate warmup and stops one growth interval before the end of training.
- When `--freeze_moe_router False`, the router is frozen during warmup and unfrozen afterward.
- The CLI contains several schedule fields retained from development; the current loop uses the warmup-based schedule described above.
- The bundled shell script sets `ADALORA_GROW_INTERVAL=50`; change it to `200` to match the paper's default configuration.

## Checkpoints and Outputs

A training directory can contain:

```text
output_dir/
├── step_000300/ or step_300/
│   ├── accelerator state files
│   ├── adalora_state.pt
│   ├── hf_model/
│   └── COMPLETED
├── adalora_rank_logs/
│   └── adalora_ranks_step_0000300.json
├── adapter/model files saved by open-instruct
└── final_merged_model/
    ├── config.json
    ├── model*.safetensors
    └── tokenizer files
```

The exact step-directory name is controlled by `open-instruct`/Accelerate utilities; the current script explicitly creates `step_<N>`.

For evaluation, the simplest path is:

```text
<output_dir>/final_merged_model
```

Intermediate `hf_model/` checkpoints are PEFT-formatted and may include a separate `router_state_dict.pt` when the router is trainable. They are not automatically merged into a standalone `merged_model/` directory by the current code.

Rank logs contain the total active rank, per-layer rank totals, and per-module active ranks. `adalora_state.pt` stores masks, importance scores, and routing-usage state for resuming training.

## Evaluation

The paper uses three runs with seeds `42`, `123`, and `456` and sampling temperature `0.2`.

### GSM8K and IFEval with LM Evaluation Harness

Edit `MODEL_CONFIGS`, `CUDA_DEVICE`, and task settings in:

```bash
eval/eval_gsm8k.sh
```

Then run:

```bash
bash eval/eval_gsm8k.sh
```

The default task list includes `gsm8k_cot` with 8-shot prompting and `ifeval` with 0-shot prompting. Results from multiple runs are aggregated into mean and sample standard deviation.

### HumanEval with EvalPlus

Edit `MODEL_CONFIGS` in:

```bash
eval/evalplus.sh
```

Then run:

```bash
bash eval/evalplus.sh
```

`evalplus.sh` expects a helper named `eval/run_evalplus_with_seed.py`. That helper is referenced but is **not included in the current repository snapshot**. Add the wrapper used in your environment or replace the command with the corresponding EvalPlus CLI/API invocation before running this script.

EvalPlus executes generated code. Run it only in a properly isolated environment.

### MedMCQA

Single run:

```bash
python eval/eval_medmcqa.py \
  --model_path /path/to/output/final_merged_model \
  --data_path /path/to/MedMCQA/test.json \
  --output_path /path/to/results/medmcqa/run_1 \
  --gpu 0 \
  --seed 42
```

Three-run wrapper:

```bash
# First edit MODEL_CONFIGS and DATA_PATH.
bash eval/run_eval_medmcqa.sh
```

The evaluator uses vLLM, generates an option letter, and reports overall, single-choice, and multi-choice accuracy.

### WMT23

Edit `MODEL_CONFIGS` and the test path in `eval/eval_wmt.py`, then run:

```bash
python eval/eval_wmt.py \
  --test_jsonl /path/to/data/WMT/test.jsonl \
  --cuda_visible_devices 0 \
  --num_runs 3 \
  --temperature 0.2
```

The script evaluates `en-cs`, `en-de`, and `en-zh` by default, retrieves references from `RicardoRei/wmt-da-human-evaluation`, and scores translations with `Unbabel/wmt22-comet-da`.

### LEDGAR

First create the multiple-choice test set:

```bash
python data/convert_ledgar_to_mc.py \
  --input_jsonl /path/to/data/LEDGAR/test.jsonl \
  --output_jsonl /path/to/data/LEDGAR/test_mc.jsonl
```

Edit `MODEL_CONFIGS` in `eval/eval_ledgar.py`, then run:

```bash
python eval/eval_ledgar.py \
  --test_jsonl /path/to/data/LEDGAR/test_mc.jsonl \
  --cuda_visible_devices 0 \
  --num_runs 3 \
  --temperature 0.2
```

The script supports vLLM and Hugging Face backends, extracts an `A`/`B`/`C`/`D` prediction, and reports aggregate and per-label accuracy.

## Main Results

DR-LoRA achieves the following results in the paper; values are means over three random seeds.

| Model | GSM8K | HumanEval | IFEval | MedMCQA | WMT23 | LEDGAR | Average |
|---|---:|---:|---:|---:|---:|---:|---:|
| OLMoE-1B-7B + DR-LoRA | 28.4 | 16.7 | 26.7 | 43.9 | 66.5 | 82.1 | **44.1** |
| Qwen1.5-MoE-A2.7B + DR-LoRA | 67.2 | 46.5 | 35.1 | 50.1 | 80.7 | 94.8 | **62.4** |
| LLaMA-MoE-3.5B + DR-LoRA | 15.7 | 13.1 | 23.4 | 32.9 | 65.8 | 81.3 | **38.7** |

Under the matched experimental setup, DR-LoRA improves over the strongest reported baseline by `+2.2`, `+1.6`, and `+1.8` average points on OLMoE, Qwen1.5-MoE, and LLaMA-MoE, respectively.

See the paper for baseline tables, ablations, masking analysis, training dynamics, allocation-strategy comparisons, and hyperparameter robustness.

## Reproducing the Paper Configuration

The paper's common settings are:

```text
Initial rank              8
Target average rank      16
Maximum reserved rank    32
Growth interval          200 optimizer steps
Module growth fraction   0.1
Usage EMA coefficient    0.9
Rank penalty exponent    1.2
Learning rate            2e-5
Scheduler                linear
Warmup ratio             0.03
Weight decay             0.0
Optimizer                fused AdamW
LoRA scaling             alpha = 2 × rank
Total steps              3,750
Micro-batch size         3 per GPU
Gradient accumulation    4
Effective batch size     48 on 4 GPUs
Sequence length          512–4096, task-dependent
```

The included `finetune.sh` is an editable example and does not exactly match every value above. In particular, it currently uses a per-device batch size of 8, one epoch, and a growth interval of 50. Update these values when reproducing the paper.

## Troubleshooting

### No expert LoRA modules are found

The script expects expert linear-module names matching:

```text
...experts.<expert_id>.gate_proj
...experts.<expert_id>.up_proj
...experts.<expert_id>.down_proj
```

Inspect `model.named_modules()` and adapt the target-module regex if your architecture uses different names.

### Routing usage remains zero

Look for `[GateHook]`, `[DiagInit]`, or `[collect_active]` messages. The model's routing output may not match either supported hook path. Adapt `_enable_record_routing` to the architecture's router output.

### The router does not train

Use:

```bash
--freeze_moe_router False
```

The current schedule keeps the router frozen during warmup and unfreezes it afterward.

### Out-of-memory errors

Reduce `--per_device_train_batch_size`, increase gradient accumulation, shorten `--max_seq_length`, enable gradient checkpointing, or use a more aggressive DeepSpeed configuration.

### Training cannot import `open_instruct`

Confirm that:

```bash
export PYTHONPATH=/path/to/open-instruct:$PYTHONPATH
```

and that the selected `open-instruct` revision contains all modules imported by `finetune.py`.

### Evaluation scripts cannot find a checkpoint

Use `<output_dir>/final_merged_model` for a standalone Hugging Face model. Paths ending in `step_<N>/merged_model` shown in some example configurations reflect an earlier local workflow and are not created by the current training script.

## Citation

This paper has been accepted at **COLM 2026**. Please cite it as:

```bibtex
@inproceedings{deng2026drlora,
  title     = {DR-LoRA: Dynamic Rank LoRA for Fine-Tuning Mixture-of-Experts Models},
  author    = {Deng, Guanzhi and Li, Bo and Chen, Ronghao and Liu, Xiujin and Han, Zhuo and Wang, Huacan and Wen, Lijie and Song, Linqi},
  booktitle = {Conference on Language Modeling (COLM)},
  year      = {2026}
}
```

## Acknowledgments

The training script is built on [open-instruct](https://github.com/allenai/open-instruct) and uses components from PyTorch, Hugging Face Transformers, Accelerate, PEFT, Datasets, DeepSpeed, Flash Attention, vLLM, LM Evaluation Harness, EvalPlus, and COMET.

## License

This repository snapshot does not include a top-level `LICENSE` file. The training script is adapted from AllenAI's `open-instruct` and retains its Apache License 2.0 header. Add or consult the project-level license before redistributing the complete repository.

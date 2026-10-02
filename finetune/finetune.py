# !/usr/bin/env python
# Copyright 2024 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This training script is built upon the open-instruct framework:
#   https://github.com/allenai/open-instruct
#
# To use this script, replace the following file in your open-instruct installation:
#   /path/to/open-instruct/open_instruct/finetune.py

# isort: off
import contextlib
import os

os.environ["NCCL_CUMEM_ENABLE"] = "0"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

with contextlib.suppress(Exception):
    import deepspeed
# isort: on

import json
import logging  # [4-bit patch]
import math
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Literal

import datasets
import torch
import transformers
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.accelerator import GradientAccumulationPlugin
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs, set_seed
from huggingface_hub import HfApi
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from rich.pretty import pprint
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig, DataCollatorForSeq2Seq, get_scheduler
from transformers.training_args import _convert_str_dict

from open_instruct import logger_utils, utils
from open_instruct.dataset_transformation import (
    INPUT_IDS_KEY,
    TOKENIZED_SFT_DATASET_KEYS,
    TokenizerConfig,
    get_cached_dataset_tulu,
    visualize_token,
)
from open_instruct.model_utils import push_folder_to_hub, save_with_accelerate
from open_instruct.padding_free_collator import TensorDataCollatorWithFlattening
from open_instruct.utils import (
    ArgumentParserPlus,
    clean_last_n_checkpoints,
    get_last_checkpoint_path,
    get_wandb_tags,
    is_beaker_job,
    launch_ai2_evals_on_weka,
    maybe_get_beaker_config,
    maybe_update_beaker_description,
    maybe_use_ai2_hf_entity,
    maybe_use_ai2_wandb_entity,
)

logger = get_logger(__name__)


@dataclass
class FlatArguments:
    _VALID_DICT_FIELDS = ["additional_model_arguments"]
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    do_not_randomize_output_dir: bool = False
    model_name_or_path: str | None = field(
        default=None,
        metadata={"help": "The model checkpoint for weights initialization. Don't set if you want to train from scratch."},
    )
    config_name: str | None = field(default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"})
    use_flash_attn: bool = field(default=True, metadata={"help": "Whether to use flash attention in the model training"})
    model_revision: str | None = field(
        default=None,
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    additional_model_arguments: dict | str | None = field(
        default_factory=dict, metadata={"help": "A dictionary of additional model args used to construct the model."}
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "Create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
                "Setting True reduces LLM loading time and RAM consumption."
            )
        },
    )
    dataset_name: str | None = field(default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."})
    dataset_mixer: dict | None = field(default=None, metadata={"help": "A dictionary of datasets (local or HF) to sample from."})
    dataset_mixer_list: list[str] = field(default_factory=lambda: ["allenai/tulu-3-sft-personas-algebra", "1.0"])
    dataset_mixer_list_splits: list[str] = field(default_factory=lambda: ["train"])
    dataset_transform_fn: list[str] = field(default_factory=lambda: ["sft_tulu_tokenize_and_truncate_v1", "sft_tulu_filter_v1"])
    dataset_target_columns: list[str] = field(default_factory=lambda: TOKENIZED_SFT_DATASET_KEYS)
    dataset_cache_mode: Literal["hf", "local"] = "local"
    dataset_local_cache_dir: str = "local_dataset_cache"
    dataset_config_hash: str | None = None
    dataset_skip_cache: bool = False
    dataset_mix_dir: str | None = field(default=None, metadata={"help": "The directory to save the mixed dataset to disk."})
    dataset_config_name: str | None = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_train_samples: int | None = field(
        default=None,
        metadata={"help": "For debugging or quicker training, truncate the number of training examples to this value if set."},
    )
    preprocessing_num_workers: int | None = field(default=None, metadata={"help": "The number of processes to use for preprocessing."})
    max_seq_length: int | None = field(
        default=None,
        metadata={"help": "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated."},
    )
    overwrite_cache: bool = field(default=False, metadata={"help": "Overwrite the cached training and evaluation sets"})
    clip_grad_norm: float = field(
        default=-1,
        metadata={"help": "Clip gradient norm. Not compatible with deepspeed (use deepspeed config instead)."},
    )
    gradient_accumulation_steps: int = field(
        default=1, metadata={"help": "Number of update steps to accumulate before performing a backward/update pass."}
    )
    learning_rate: float = field(default=2e-5, metadata={"help": "The initial learning rate for AdamW optimizer."})
    logging_steps: int | None = field(default=None, metadata={"help": "Log the training loss and learning rate every logging_steps steps."})
    lora_rank: int = field(default=64, metadata={"help": "The rank of LoRA."})
    lora_alpha: float = field(default=16, metadata={"help": "The alpha parameter of LoRA."})
    lora_dropout: float = field(default=0.1, metadata={"help": "The dropout rate of LoRA modules."})
    lora_scaling_factor: float = field(default=2, metadata={"help": "The scaling factor of LoRA modules."})
    use_adalora: bool = field(default=True, metadata={"help": "If True, use AdaLoRA-style dynamic rank growing on expert LoRA modules."})
    adalora_beta: float = field(default=0.0, metadata={"help": "EMA coefficient for AdaLoRA importance scores. 0 = plain accumulation."})
    adalora_max_rank: int = field(default=256, metadata={"help": "Maximum rank reserved per LoRA module."})
    adalora_target_rank: int = field(default=128, metadata={"help": "Target average rank per module at the end of growing."})
    adalora_grow_start_step: int = field(default=-1, metadata={"help": "Optimizer step at which rank growing is allowed to start."})
    adalora_grow_offset_after_warmup: int = field(default=0, metadata={"help": "When adalora_grow_start_step < 0: grow_start = warmup_end_step + this offset."})
    adalora_grow_interval: int = field(default=500, metadata={"help": "Number of optimizer steps between rank grow events."})
    adalora_usage_decay: float = field(default=0.9, metadata={"help": "EMA decay coefficient for expert routing frequency (0~1, smaller = more recent-biased)."})
    adalora_module_grow_frac: float = field(default=0.1, metadata={"help": "Per grow event, each LoRA module activates at most this fraction of its initial free rank (0~1)."})
    router_unfreeze_step: int = field(default=-1, metadata={"help": "Global optimizer step at which the MoE gate is unfrozen; <0 means auto-unfreeze after warmup."})
    router_refreeze_step: int = field(default=-1, metadata={"help": "Step at which to re-freeze the gate; <0 means never re-freeze."})
    router_refreeze_after_grow: int = field(default=-1, metadata={"help": "Relative mode for router_refreeze_step: when router_refreeze_step < 0, re-freeze router this many steps after AdaLoRA growing ends."})
    lr_scheduler_type: str = field(default="linear", metadata={"help": "The scheduler type to use for learning rate adjustment.", "choices": ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]})
    num_train_epochs: int = field(default=2, metadata={"help": "Total number of training epochs to perform."})
    output_dir: str = field(default="output/", metadata={"help": "The output directory where model predictions and checkpoints will be written."})
    per_device_train_batch_size: int = field(default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for training."})
    use_lora: bool = field(default=False, metadata={"help": "If True, use LoRA (low-rank parameter-efficient training) to train the model."})
    use_qlora: bool = field(default=False, metadata={"help": "Use qLoRA training - initializes model in quantized form. Not compatible with deepspeed."})
    use_8bit_optimizer: bool = field(default=False, metadata={"help": "Use 8-bit optimizer from bitsandbytes. Not compatible with deepspeed."})
    warmup_ratio: float = field(default=0.03, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."})
    final_lr_ratio: float | None = field(default=None, metadata={"help": "Set the final lr to final_lr_ratio * learning_rate at end of training. Only for linear schedulers."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if applied."})
    timeout: int = field(default=3600, metadata={"help": "Timeout for the training process in seconds."})
    resume_from_checkpoint: str | None = field(default=None, metadata={"help": "If training should continue from a checkpoint folder."})
    report_to: str | list[str] = field(default="all", metadata={"help": "Integration(s) to report results and logs to."})
    save_to_hub: str | None = field(default=None, metadata={"help": "Save the model to the Hub under this name."})
    gradient_checkpointing: bool = field(default=False, metadata={"help": "Turn on gradient checkpointing. Saves memory but slows training."})
    use_liger_kernel: bool = field(default=False, metadata={"help": "Whether to use LigerKernel for training."})
    max_train_steps: int | None = field(default=None, metadata={"help": "If set, overrides the number of training steps. Otherwise num_train_epochs is used."})
    seed: int = field(default=42, metadata={"help": "Random seed for initialization and dataset shuffling."})
    checkpointing_steps: str | None = field(default=None, metadata={"help": "Save states every n steps, or 'epoch' for each epoch."})
    keep_last_n_checkpoints: int = field(default=3, metadata={"help": "How many checkpoints to keep in the output directory. -1 for all."})
    fused_optimizer: bool = field(default=True, metadata={"help": "Whether to use fused AdamW."})
    load_balancing_loss: bool = field(default=False, metadata={"help": "Whether to include a load balancing loss (for OLMoE)."})
    load_balancing_weight: float = field(default=0.5, metadata={"help": "Weight for load balancing loss if applicable."})
    clean_checkpoints_at_end: bool = field(default=True, metadata={"help": "Whether to clean up all previous checkpoints at the end of training."})
    freeze_moe_router: bool = field(default=True, metadata={"help": "Freeze the MoE router (gate) weights during SFT."})
    lora_experts_only: bool = field(default=True, metadata={"help": "Attach LoRA only to MoE experts (MLP gate/up/down proj)."})
    lora_target_mlp_projs: list[str] = field(
        default_factory=lambda: ["gate_proj", "up_proj", "down_proj"],
        metadata={"help": (
            "Which expert MLP projections to attach LoRA to. "
            "Choices: any subset of [gate_proj, up_proj, down_proj]. "
            "e.g. --lora_target_mlp_projs up_proj down_proj  (skip gate_proj)"
        )}
    )
    lora_topk_only: bool = field(default=True, metadata={"help": "Per step, update LoRA grads only for experts selected by top-k routing."})
    with_tracking: bool = False
    wandb_project_name: str = "open_instruct_internal"
    wandb_entity: str | None = None
    push_to_hub: bool = True
    hf_entity: str | None = None
    hf_repo_id: str | None = None
    hf_repo_revision: str | None = None
    hf_repo_url: str | None = None
    try_launch_beaker_eval_jobs: bool = True
    hf_metadata_dataset: str | None = "allenai/tulu-3-evals"
    cache_dataset_only: bool = False
    add_seed_and_date_to_exp_name: bool = True
    try_auto_save_to_beaker: bool = True
    gs_bucket_path: str | None = None
    oe_eval_tasks: list[str] | None = None
    oe_eval_max_length: int = 4096
    sync_each_batch: bool = False
    packing: bool = field(default=False, metadata={"help": "Whether to use packing/padding-free collation via TensorDataCollatorWithFlattening"})
    verbose: bool = field(default=False, metadata={"help": "Optionally print additional statistics at each reporting period"})

    def __post_init__(self):
        if self.dataset_name is None and self.dataset_mixer is None and self.dataset_mixer_list is None:
            raise ValueError("Need either a dataset name, dataset mixer, or dataset mixer list.")
        if (self.dataset_name is not None and (
                self.dataset_mixer is not None or self.dataset_mixer_list is not None)) or (
                self.dataset_name is not None) or (
                self.dataset_mixer is not None and self.dataset_mixer_list is not None):
            raise ValueError("Cannot provide two dataset selection mechanisms.")
        if self.try_launch_beaker_eval_jobs and not self.push_to_hub:
            raise ValueError("Cannot launch Beaker evaluation jobs without pushing to the Hub.")
        if self.final_lr_ratio is not None:
            if self.lr_scheduler_type != "linear":
                raise NotImplementedError("final_lr_ratio only currently implemented for linear schedulers")
            if not (1.0 >= self.final_lr_ratio >= 0.0):
                raise ValueError(f"final_lr_ratio must be between 0 and 1, not {self.final_lr_ratio=}")
        for dict_feld in self._VALID_DICT_FIELDS:
            passed_value = getattr(self, dict_feld)
            if isinstance(passed_value, str) and passed_value.startswith("{"):
                loaded_dict = json.loads(passed_value)
                loaded_dict = _convert_str_dict(loaded_dict)
                setattr(self, dict_feld, loaded_dict)


def _iter_expert_lora_modules(model):
    for name, module in model.named_modules():
        if ".experts." in name and hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            yield name, module


def _reinit_lora_weights(model):
    logger.info("[AdaLoRA] Re-initializing LoRA weights before accelerator.prepare()...")
    num_reinitialized = 0
    for name, mod in _iter_expert_lora_modules(model):
        if hasattr(mod, "lora_A") and "default" in mod.lora_A and hasattr(mod, "lora_B") and "default" in mod.lora_B:
            with torch.no_grad():
                torch.nn.init.kaiming_uniform_(mod.lora_A["default"].weight, a=math.sqrt(5))
                torch.nn.init.zeros_(mod.lora_B["default"].weight)
            num_reinitialized += 1
    logger.info(f"[AdaLoRA] Re-initialized A and zeroed B for {num_reinitialized} modules.")


def main(args: FlatArguments, tc: TokenizerConfig):
    def _collect_active_experts(model):
        try:
            base_model = accelerator.unwrap_model(model)
        except Exception:
            base_model = model
        active = {}
        usage_incr = {}

        # Traverse through PeftModel wrapping to find the model object that holds decoder layers.
        # PeftModel structure: PeftModel -> .base_model (LoraModel) -> .model (original model)
        # Use at most 5 steps to prevent accidental loops, checking for 'layers' at each step.
        raw_model = base_model
        for _ in range(5):
            candidate_dec = (raw_model.get_decoder() if hasattr(raw_model, "get_decoder")
                             else raw_model if hasattr(raw_model, "layers") else None)
            if candidate_dec is not None and hasattr(candidate_dec, "layers"):
                break
            if hasattr(raw_model, "model") and hasattr(getattr(raw_model, "model"), "layers"):
                raw_model = raw_model.model
                break
            if hasattr(raw_model, "base_model"):
                next_model = raw_model.base_model
                if next_model is raw_model:
                    break
                raw_model = next_model
            else:
                break

        dec = raw_model.get_decoder() if hasattr(raw_model, "get_decoder") else (
            raw_model if hasattr(raw_model, "layers") else None
        )
        if dec is None or not hasattr(dec, "layers"):
            if accelerator.is_main_process:
                logger.warning(f"[DEBUG][collect_active] cannot find decoder layers, "
                               f"raw_model type={type(raw_model).__name__}")
            return active, usage_incr
        for i, layer in enumerate(dec.layers):
            mlp = layer.mlp
            if getattr(mlp, "_last_selected_experts", None) is not None:
                sel = mlp._last_selected_experts
                try:
                    sel_flat = torch.as_tensor(sel, device=accelerator.device).view(-1)
                    uniq, counts = torch.unique(sel_flat, return_counts=True)
                    eids = [int(x) for x in uniq.tolist()]
                    cnts = [int(c) for c in counts.tolist()]
                    active[i] = set(eids)
                    usage_incr[i] = {eid: cnt for eid, cnt in zip(eids, cnts)}
                except Exception as e:
                    logger.warning(f"Error collecting active experts in layer {i}: {e}")
                    pass
            else:
                if accelerator.is_main_process:
                    logger.warning(f"[collect_active] layer {i}: _last_selected_experts is None, hook not firing?")
        return active, usage_incr

    def _zero_inactive_expert_lora_grads(model, active_dict):
        if not active_dict:
            return
        for n, p in model.named_parameters():
            if p.grad is None or "lora_" not in n:
                continue
            parts = n.split(".")
            if "layers" in parts and "experts" in parts:
                try:
                    li = parts.index("layers")
                    ei = parts.index("experts")
                    layer_id = int(parts[li + 1])
                    expert_id = int(parts[ei + 1])
                    if layer_id not in active_dict or expert_id not in active_dict[layer_id]:
                        p.grad.zero_()
                except (ValueError, IndexError):
                    continue

    def _parse_layer_id_from_name(name: str) -> int:
        parts = name.split(".")
        if "layers" in parts:
            try:
                return int(parts[parts.index("layers") + 1])
            except (ValueError, IndexError):
                return -1
        return -1

    def _parse_expert_id_from_name(name: str) -> int:
        parts = name.split(".")
        if "experts" in parts:
            try:
                return int(parts[parts.index("experts") + 1])
            except (ValueError, IndexError):
                return -1
        return -1

    def _init_adalora_state(model, args):
        logger.info("[AdaLoRA-Grow] Initializing state with hook-based gradient accumulation...")
        baseline_rank = args.lora_rank
        adalora_beta = args.adalora_beta

        example_mod = next(_iter_expert_lora_modules(model), (None, None))[1]
        if example_mod is None:
            logger.warning("[AdaLoRA-Grow] No expert LoRA modules found; state not initialized.")
            return None
        max_rank = example_mod.lora_A["default"].weight.size(0)

        state = {
            "layers": {},
            "baseline_rank": baseline_rank,
            "max_rank": max_rank,
            "expert_usage": {},
            "layer_init_total_rank": {}, "layer_target_total_rank": {},
            "layer_grow_quota_per_step": {}, "num_grow_events": 0,
            "grow_start_step": -1, "grow_end_step": -1, "grow_interval": -1,
        }

        for name, mod in _iter_expert_lora_modules(model):
            layer_id = _parse_layer_id_from_name(name)
            expert_id = _parse_expert_id_from_name(name)
            if layer_id < 0 or expert_id < 0: continue

            A = mod.lora_A["default"].weight
            B = mod.lora_B["default"].weight
            r_max = A.size(0)

            mask = torch.zeros(r_max, dtype=torch.bool, device=A.device)
            active_r = min(baseline_rank, r_max)
            mask[:active_r] = True

            score = torch.zeros(r_max, dtype=torch.float32, device=A.device)
            mod._adalora_active_mask = mask
            mod._adalora_score = score

            def create_hook(score_buf, weight, is_lora_A, beta):
                def grad_hook(grad):
                    if grad is None: return
                    with torch.no_grad():
                        s = (grad * weight).sum(dim=1 if is_lora_A else 0).abs()
                        # [fix] fp16 training multiplies gradients by the GradScaler factor,
                        # and its first steps overflow to inf/NaN (the optimizer skips them).
                        # Undo the factor and ignore non-finite steps, so the scores match
                        # bf16 training. Runs on the GPU without waiting for it.
                        scaler = getattr(accelerator, "scaler", None)
                        if scaler is not None and getattr(scaler, "_scale", None) is not None:
                            s = s / scaler._scale.to(s.device)
                        ok = torch.isfinite(s).all()
                        s = torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
                        if beta > 0:
                            updated = score_buf * beta + (1.0 - beta) * s
                        else:
                            updated = score_buf + s
                        score_buf.copy_(torch.where(ok, updated, score_buf))
                return grad_hook

            A.register_hook(create_hook(score, A, is_lora_A=True, beta=adalora_beta))
            B.register_hook(create_hook(score, B, is_lora_A=False, beta=adalora_beta))

            if layer_id not in state["layers"]:
                state["layers"][layer_id] = {"modules": [], "target_total_rank": 0}

            state["layers"][layer_id]["modules"].append({
                "name": name, "module": mod, "rank": r_max, "layer_id": layer_id,
                "expert_id": expert_id, "initial_free": max(0, r_max - active_r),
            })

        target_avg_rank = min(args.adalora_target_rank, max_rank)
        for lid, info in state["layers"].items():
            num_mods = len(info["modules"])
            init_total = num_mods * baseline_rank
            tgt_total = num_mods * target_avg_rank
            info["target_total_rank"] = tgt_total
            state["layer_init_total_rank"][lid] = init_total
            state["layer_target_total_rank"][lid] = tgt_total

        logger.info(f"[AdaLoRA-Grow] Initialized hook-based state for {len(state['layers'])} layers.")
        return state

    def _adalora_current_total_rank_layer(layer_info):
        return sum(int(entry["module"]._adalora_active_mask.sum().item()) for entry in layer_info["modules"])

    def _plan_layerwise_grow(adalora_state, args, adalora_grow_start_step: int, adalora_grow_end_step: int):
        layers = adalora_state.get("layers", {})
        layer_init_total = adalora_state.get("layer_init_total_rank", {})
        layer_target_total = adalora_state.get("layer_target_total_rank", {})
        if not layers:
            logger.warning("[AdaLoRA-Grow] _plan_layerwise_grow: no layers found, skipping planning.")
            adalora_state["num_grow_events"] = 0
            adalora_state["layer_grow_quota_per_step"] = {}
            return

        start = max(1, adalora_grow_start_step)
        end = min(adalora_grow_end_step, args.max_train_steps) if adalora_grow_end_step > 0 else args.max_train_steps
        if end < start:
            logger.warning(
                f"[AdaLoRA-Grow] grow window invalid: start={start}, end={end}; no grow events will be scheduled.")
            num_events = 0
        else:
            interval = max(1, int(args.adalora_grow_interval))
            num_events = ((end - start) // interval) + 1

        adalora_state["num_grow_events"] = num_events
        adalora_state["grow_start_step"] = start
        adalora_state["grow_end_step"] = end
        adalora_state["grow_interval"] = interval

        layer_quota_per_step = {}
        for lid, info in sorted(layers.items()):
            G_layer_total = max(0, layer_target_total.get(lid, 0) - layer_init_total.get(lid, 0))
            quota = int(math.ceil(G_layer_total / float(num_events))) if num_events > 0 else 0
            layer_quota_per_step[lid] = quota
        adalora_state["layer_grow_quota_per_step"] = layer_quota_per_step
        logger.info(f"[AdaLoRA-Grow] Global grow plan created for {num_events} events across window [{start}, {end}].")

    def _adalora_mask_pruned_grads(adalora_state):
        # Called inside the sync_gradients block (once per optimizer step) to zero out
        # gradients for inactive rank dimensions, rather than after every micro-batch backward.
        if adalora_state is None: return
        for layer_info in adalora_state["layers"].values():
            for entry in layer_info["modules"]:
                mod = entry["module"]
                mask = mod._adalora_active_mask
                if mod.lora_A["default"].weight.grad is not None:
                    mod.lora_A["default"].weight.grad[~mask, :].zero_()
                if mod.lora_B["default"].weight.grad is not None:
                    mod.lora_B["default"].weight.grad[:, ~mask].zero_()

    def _collect_adalora_rank_snapshot(adalora_state):
        if adalora_state is None: return None
        per_module, per_layer, total_rank = {}, defaultdict(int), 0
        for lid, layer_info in adalora_state["layers"].items():
            for entry in layer_info["modules"]:
                name, mod = entry["name"], entry["module"]
                if not hasattr(mod, "_adalora_active_mask"): continue
                active_r = int(mod._adalora_active_mask.sum().item())
                total_rank += active_r
                per_layer[lid] += active_r
                per_module[name] = {"layer_id": lid, "expert_id": entry.get("expert_id", -1), "active_rank": active_r}
        return {"total_rank": total_rank, "per_layer_rank": dict(per_layer), "per_module_rank": per_module}

    def _save_hf_eval_checkpoint(accelerator, model, tokenizer, ckpt_dir: str, freeze_moe_router: bool):
        hf_dir = os.path.join(ckpt_dir, "hf_model")
        unwrapped_model = accelerator.unwrap_model(model)

        # All ranks must participate in get_state_dict for ZeRO Stage 2/3 aggregation.
        # Must be called before the is_main_process guard; otherwise non-rank-0 processes
        # skip participation and cause a deadlock under ZeRO Stage 3.
        # The aggregated full state_dict is valid only on rank-0; other ranks return an empty dict.
        state_dict = accelerator.get_state_dict(model)

        # Router (gate) weights are base model parameters and are not saved by PEFT's save_pretrained.
        # They would be lost during merge_and_unload, so we extract and save them separately.
        # Extracted from the already-aggregated state_dict for correctness under ZeRO Stage 2/3.
        router_state_dict = {}
        if not freeze_moe_router and state_dict:
            for name, param in state_dict.items():
                if ".gate." in name and "lora_" not in name:
                    router_state_dict[name] = param.cpu()

        if accelerator.is_main_process:
            os.makedirs(hf_dir, exist_ok=True)
            unwrapped_model.save_pretrained(
                hf_dir,
                is_main_process=True,
                save_function=accelerator.save,
                state_dict=state_dict,
                safe_serialization=True,
            )
            tokenizer.save_pretrained(hf_dir)
            logger.info(f"[HF-EVAL] Saved HF format evaluation checkpoint to {hf_dir}")

            if router_state_dict:
                router_path = os.path.join(hf_dir, "router_state_dict.pt")
                torch.save(router_state_dict, router_path)
                logger.info(f"[HF-EVAL] Saved router weights ({len(router_state_dict)} tensors) to {router_path}")
            else:
                with open(os.path.join(hf_dir, "router_frozen.txt"), "w") as f:
                    f.write("router was frozen during training, no router weights to restore.\n")

    def _save_adalora_state(adalora_state, output_dir, accelerator):
        if adalora_state is None or not accelerator.is_main_process: return
        masks, scores = {}, {}
        for lid, layer_info in adalora_state["layers"].items():
            for entry in layer_info["modules"]:
                name, mod = entry["name"], entry["module"]
                masks[name] = mod._adalora_active_mask.detach().cpu()
                scores[name] = mod._adalora_score.detach().cpu()
        state_to_save = {"masks": masks, "scores": scores, "expert_usage": adalora_state.get("expert_usage", {})}
        path = os.path.join(output_dir, "adalora_state.pt")
        torch.save(state_to_save, path)
        logger.info(f"[AdaLoRA] Saved AdaLoRA state to {path}")

    def _restore_adalora_state(adalora_state, ckpt_dir, accelerator):
        if adalora_state is None: return
        path = os.path.join(ckpt_dir, "adalora_state.pt")
        if not os.path.exists(path):
            logger.warning(f"[AdaLoRA] No AdaLoRA state file found at {path}, skipping restore.")
            return
        loaded = torch.load(path, map_location="cpu")  # [2-GPU patch] copied to each GPU below
        name2entry = {entry["name"]: entry for layer_info in adalora_state["layers"].values() for entry in
                      layer_info["modules"]}
        restored_cnt = 0
        for name, mask in loaded.get("masks", {}).items():
            if name in name2entry:
                mod = name2entry[name]["module"]
                mod._adalora_active_mask.data.copy_(mask.to(mod.lora_A["default"].weight.device))
                restored_cnt += 1
        for name, score in loaded.get("scores", {}).items():
            if name in name2entry:
                mod = name2entry[name]["module"]
                mod._adalora_score.data.copy_(score.to(mod.lora_A["default"].weight.device))
        adalora_state["expert_usage"] = loaded.get("expert_usage", {})
        logger.info(f"[AdaLoRA] Restored AdaLoRA state for {restored_cnt} modules from {path}")

    def _adalora_update_expert_usage(adalora_state, usage_incr_dict, decay: float = 0.9):
        if adalora_state is None: return
        usage = adalora_state.setdefault("expert_usage", {})

        if 0.0 < decay < 1.0:
            for k in list(usage.keys()):
                usage[k] *= decay

        for (layer_id, expert_id), cnt in usage_incr_dict.items():
            key = (layer_id, expert_id)
            usage[key] = usage.get(key, 0.0) + float(cnt)

        if accelerator.is_main_process:
            total_usage = float(sum(usage.values())) if usage else 0.0
            logger.info(f"[DEBUG][expert_usage] sum={total_usage:.4f}, num_entries={len(usage)}")

    # ---------------- [2-GPU patch] keep DR-LoRA the same on every GPU ----------------
    # With 2 GPUs (DDP) each GPU trains on its own half of the batch. DDP averages the
    # LoRA gradients, but DR-LoRA's own state (expert usage, active experts, growth
    # scores, rank masks) is computed on each GPU separately. These helpers combine it,
    # so both GPUs make the same growth decision and the model copies never drift apart.
    # On 1 GPU none of them is called.
    def _ddp_on():
        return (accelerator.num_processes > 1 and torch.distributed.is_available()
                and torch.distributed.is_initialized())

    def _all_expert_modules(adalora_state):
        # same order on every GPU (built from named_modules)
        return [e["module"] for li in adalora_state["layers"].values() for e in li["modules"]]

    def _ddp_sync_usage(active_union, window_usage_incr, n_layers, n_experts):
        """Add up expert token counts from all GPUs. An expert is active if any GPU used it."""
        t = torch.zeros(2, n_layers, n_experts, dtype=torch.float32)
        for (L, e), c in window_usage_incr.items():
            t[0, L, e] += float(c)
        for L, ss in active_union.items():
            for e in ss:
                t[1, L, e] = 1.0
        t = t.to(accelerator.device)
        torch.distributed.all_reduce(t)
        t = t.cpu()
        new_active, new_usage = defaultdict(set), defaultdict(float)
        for L, e in torch.nonzero(t[1] > 0).tolist():
            new_active[L].add(e)
        for L, e in torch.nonzero(t[0] > 0).tolist():
            new_usage[(L, e)] = float(t[0, L, e])
        return new_active, new_usage

    def _ddp_mean_scores(adalora_state):
        """Average the growth scores over GPUs (each GPU only saw its own gradients)."""
        bufs = [m._adalora_score for m in _all_expert_modules(adalora_state)]
        flat = torch.cat([b.reshape(-1) for b in bufs])
        torch.distributed.all_reduce(flat)
        flat /= accelerator.num_processes
        off = 0
        for b in bufs:
            n = b.numel()
            b.copy_(flat[off:off + n].view_as(b))
            off += n

    def _ddp_share_masks(adalora_state, grown, saturated, step):
        """Copy GPU 0's rank masks to every GPU and report if they differed."""
        masks = [m._adalora_active_mask for m in _all_expert_modules(adalora_state)]
        local = torch.cat([m.reshape(-1) for m in masks]).to(torch.uint8)
        flat = local.clone()
        flags = torch.tensor([int(grown), int(saturated)], dtype=torch.long, device=flat.device)
        torch.distributed.broadcast(flat, src=0)
        torch.distributed.broadcast(flags, src=0)
        n_diff = (flat != local).sum().to(torch.long).reshape(1)
        torch.distributed.all_reduce(n_diff)
        off = 0
        for m in masks:
            n = m.numel()
            m.copy_(flat[off:off + n].view_as(m).bool())
            off += n
        logger.info(f"[DDP-sync] step {step}: rank masks "
                    f"{'identical' if int(n_diff) == 0 else 'DIFFERED (' + str(int(n_diff)) + ' dims), fixed'} "
                    f"on {accelerator.num_processes} GPUs, grown={int(flags[0])}")
        return int(flags[0]), bool(flags[1])

    def _ddp_check_weights(model, step):
        """Log whether the LoRA weights are the same on every GPU (they should be)."""
        total = torch.zeros(1, dtype=torch.float64, device=accelerator.device)
        for n, p in model.named_parameters():
            if "lora_" in n:
                total += p.detach().double().sum()
        allv = [torch.zeros_like(total) for _ in range(accelerator.num_processes)]
        torch.distributed.all_gather(allv, total)
        vals = [float(v) for v in allv]
        same = max(vals) == min(vals)
        logger.info(f"[DDP-sync] step {step}: LoRA weights "
                    f"{'identical' if same else 'DIFFER'} on {len(vals)} GPUs (checksums {vals})")

    def _adalora_grow_once_per_layer(adalora_state, usage_alpha: float = 1.0, score_beta: float = 1.0,
                                     rank_gamma: float = 1.2, module_grow_frac: float = 0.10,
                                     debug_top_k: int = 5):
        """
        Determine rank grow decisions at the expert granularity.
        Prints diagnostic info per layer before growing:
          - raw and normalized usage per expert
          - average gradient score per expert
          - final priority and current active rank
        """
        if adalora_state is None: return 0, False
        baseline_rank, max_rank = adalora_state["baseline_rank"], adalora_state["max_rank"]
        expert_usage = adalora_state.get("expert_usage", {})
        layer_grow_quota = adalora_state.get("layer_grow_quota_per_step", {})
        total_grown_global, all_saturated = 0, True

        nonzero_usage = sum(1 for v in expert_usage.values() if v > 1e-6)
        logger.info(
            f"[Diag-Usage] expert_usage global: "
            f"total_entries={len(expert_usage)}, "
            f"nonzero={nonzero_usage}, "
            f"max={max(expert_usage.values(), default=0):.4f}, "
            f"min_nonzero={min((v for v in expert_usage.values() if v > 1e-6), default=0):.6f}"
        )

        eps = 1e-6

        for layer_id, layer_info in adalora_state["layers"].items():
            cur_total = _adalora_current_total_rank_layer(layer_info)
            target_total = layer_info.get("target_total_rank", cur_total)
            if cur_total < target_total: all_saturated = False
            if cur_total >= target_total: continue

            remaining = target_total - cur_total
            per_step_quota = layer_grow_quota.get(layer_id, 0)
            if per_step_quota <= 0 or remaining <= 0: continue

            num_to_grow = min(per_step_quota, remaining)
            modules = layer_info["modules"]

            from collections import defaultdict as _defaultdict
            expert_groups = _defaultdict(list)
            for entry in modules:
                expert_groups[entry["expert_id"]].append(entry)

            # Collect diagnostic info per expert
            diag_rows = []
            for eid, entries in sorted(expert_groups.items()):
                usage_raw = float(expert_usage.get((layer_id, eid), 0.0))
                proj_scores = []
                total_active_r = 0
                for entry in entries:
                    mod = entry["module"]
                    active_mask = mod._adalora_active_mask
                    s = float(mod._adalora_score[active_mask].mean().item()) if active_mask.any() else 0.0
                    proj_scores.append(s)
                    total_active_r += int(active_mask.sum().item())
                avg_score = sum(proj_scores) / len(proj_scores) if proj_scores else 0.0
                avg_active_r = total_active_r / len(entries) if entries else 0
                free_rank = sum(
                    int((~e["module"]._adalora_active_mask).sum().item()) for e in entries
                )
                diag_rows.append({
                    "eid": eid,
                    "usage_raw": usage_raw,
                    "avg_score": avg_score,
                    "avg_active_r": avg_active_r,
                    "free_rank": free_rank,
                    "proj_scores": proj_scores,
                })

            layer_usage_sum = max(sum(r["usage_raw"] for r in diag_rows), eps)
            layer_score_sum = max(sum(r["avg_score"] for r in diag_rows), eps)

            for r in diag_rows:
                u_norm = r["usage_raw"] / layer_usage_sum
                s_norm = r["avg_score"] / layer_score_sum
                priority = ((u_norm + eps) ** usage_alpha * (s_norm + eps) ** score_beta) / (
                        (r["avg_active_r"] + 1) ** rank_gamma)
                r["u_norm"] = u_norm
                r["s_norm"] = s_norm
                r["priority"] = priority

            sorted_rows = sorted(diag_rows, key=lambda x: -x["priority"])

            usage_all_zero = all(r["usage_raw"] < eps for r in diag_rows)
            score_all_zero = all(r["avg_score"] < eps for r in diag_rows)

            header = (
                f"\n{'=' * 80}\n"
                f"[AdaLoRA-Diag] Layer {layer_id} | "
                f"cur_total={cur_total}, target={target_total}, quota={num_to_grow}\n"
                f"  usage_all_zero={usage_all_zero}, score_all_zero={score_all_zero}\n"
                f"  layer_usage_sum={layer_usage_sum:.4f}, layer_score_sum={layer_score_sum:.6f}\n"
                f"{'─' * 80}\n"
                f"  {'EID':>4} | {'usage_raw':>10} | {'u_norm':>8} | "
                f"{'avg_score':>12} | {'s_norm':>8} | {'avg_r':>6} | {'free':>5} | {'priority':>12}\n"
                f"{'─' * 80}"
            )
            logger.info(header)

            logger.info(f"  [TOP-{debug_top_k} by priority]")
            for r in sorted_rows[:debug_top_k]:
                logger.info(
                    f"  {r['eid']:>4} | {r['usage_raw']:>10.4f} | {r['u_norm']:>8.4f} | "
                    f"{r['avg_score']:>12.6f} | {r['s_norm']:>8.4f} | "
                    f"{r['avg_active_r']:>6.1f} | {r['free_rank']:>5} | {r['priority']:>12.6e}"
                )

            if len(sorted_rows) > debug_top_k:
                logger.info(f"  [BOTTOM-{debug_top_k} by priority]")
                for r in sorted_rows[-debug_top_k:]:
                    logger.info(
                        f"  {r['eid']:>4} | {r['usage_raw']:>10.4f} | {r['u_norm']:>8.4f} | "
                        f"{r['avg_score']:>12.6f} | {r['s_norm']:>8.4f} | "
                        f"{r['avg_active_r']:>6.1f} | {r['free_rank']:>5} | {r['priority']:>12.6e}"
                    )

            logger.info(f"  [Expert ID 0~15, raw order]")
            for r in diag_rows[:16]:
                logger.info(
                    f"  {r['eid']:>4} | {r['usage_raw']:>10.4f} | {r['u_norm']:>8.4f} | "
                    f"{r['avg_score']:>12.6f} | {r['s_norm']:>8.4f} | "
                    f"{r['avg_active_r']:>6.1f} | {r['free_rank']:>5} | {r['priority']:>12.6e}"
                )
            logger.info(f"{'=' * 80}")

            # Grow logic
            expert_priorities = [(r["priority"], r["eid"]) for r in diag_rows]
            expert_priorities.sort(key=lambda x: -x[0])

            grown = 0
            for _, eid in expert_priorities:
                if grown >= num_to_grow: break
                entries = expert_groups[eid]
                has_free = any(
                    torch.nonzero(~e["module"]._adalora_active_mask, as_tuple=False).numel() > 0
                    for e in entries
                )
                if not has_free: continue

                per_module_cap = max(1, int(math.ceil(
                    entries[0].get("initial_free", max_rank - baseline_rank) * module_grow_frac)))

                grew_this_expert = 0
                for entry in entries:
                    mod = entry["module"]
                    mask = mod._adalora_active_mask
                    free_idx = torch.nonzero(~mask, as_tuple=False).view(-1)
                    if free_idx.numel() == 0: continue

                    can_grow_here = min(per_module_cap, free_idx.numel(), num_to_grow - grown)
                    if can_grow_here <= 0: continue

                    chosen = free_idx[:can_grow_here].tolist()
                    with torch.no_grad():
                        for r_idx in chosen:
                            mask[r_idx] = True
                            mod._adalora_score[r_idx] = 0.0
                    grew_this_expert = max(grew_this_expert, can_grow_here)

                grown += grew_this_expert

            if grown > 0:
                total_grown_global += grown
                new_total = _adalora_current_total_rank_layer(layer_info)
                logger.info(
                    f"[AdaLoRA-Grow] layer {layer_id}: grew {grown} ranks "
                    f"(quota={num_to_grow}). total_rank: {cur_total} -> {new_total}, "
                    f"target_total={target_total}."
                )

        return total_grown_global, all_saturated

    # ============================ Main Logic ============================
    accelerator = Accelerator(
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),
        gradient_accumulation_plugin=GradientAccumulationPlugin(num_steps=args.gradient_accumulation_steps,
                                                                sync_each_batch=args.sync_each_batch),
        # [2-GPU patch] each batch uses only some experts, so some LoRA weights get no
        # gradient on a GPU; DDP must be told. Longer timeout for the model download.
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True),
                         InitProcessGroupKwargs(timeout=timedelta(hours=2))],
        **({"log_with": args.report_to, "project_dir": args.output_dir} if args.with_tracking else {})
    )

    logger_utils.setup_logger()
    # [4-bit patch] some platforms (e.g. Kaggle) set up logging first at WARNING,
    # which hides every INFO line (loss, rank growth, finish). Show them.
    logging.getLogger().setLevel(logging.INFO)
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # --- 1. Load Model, Tokenizer ---
    tc.tokenizer_name_or_path = args.model_name_or_path if tc.tokenizer_name_or_path is None else tc.tokenizer_name_or_path
    tokenizer = tc.tokenizer
    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True,
                                        **args.additional_model_arguments)
    # [4-bit patch] bf16 needs Ampere or newer; T4 gets fp16, CPU fp32.
    if torch.cuda.is_available():
        model_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        model_dtype = torch.float32
    quant_kwargs = {}
    if args.use_qlora:
        # Keep every router (mlp.gate) and lm_head in full precision.
        keep = ["lm_head"] + [f"model.layers.{i}.mlp.gate" for i in range(config.num_hidden_layers)]
        quant_kwargs = dict(
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=model_dtype, llm_int8_skip_modules=keep),
            # accelerate needs an integer GPU index here (a bare "cuda" device crashes it)
            device_map={"": torch.cuda.current_device() if torch.cuda.is_available() else "cpu"},
        )
    logger.info(f"[4-bit patch] dtype={model_dtype}, use_qlora={args.use_qlora}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, config=config, trust_remote_code=True, low_cpu_mem_usage=args.low_cpu_mem_usage,
        torch_dtype=model_dtype, attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
        **quant_kwargs
    )
    if args.use_qlora:
        # [2-GPU patch] DDP with find_unused_parameters needs non-reentrant checkpointing
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False} if accelerator.num_processes > 1 else None)
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

    # --- 2. Add LoRA ---
    if args.use_lora:
        logger.info("Initializing LoRA model...")

        _found_linear_names = {
            name.split(".")[-1]
            for name, module in model.named_modules()
            if module.__class__.__name__ in ("Linear", "Linear4bit")  # [4-bit patch]
        }
        _EXPERT_MLP_CANDIDATES = {"gate_proj", "up_proj", "down_proj"}
        _ATTN_CANDIDATES       = {"q_proj", "k_proj", "v_proj", "o_proj"}

        _expert_modules = sorted(_EXPERT_MLP_CANDIDATES & _found_linear_names)
        _attn_modules   = sorted(_ATTN_CANDIDATES       & _found_linear_names)

        if not _expert_modules:
            raise ValueError(
                f"[LoRA] Cannot find expert MLP layers. Expected at least one of {_EXPERT_MLP_CANDIDATES}, "
                f"but found Linear layer suffixes: {_found_linear_names}. "
                f"Please verify the model architecture."
            )

        if args.lora_experts_only:
            _valid_mlp_projs = {"gate_proj", "up_proj", "down_proj"}
            _requested = set(args.lora_target_mlp_projs)
            _invalid = _requested - _valid_mlp_projs
            if _invalid:
                raise ValueError(f"[LoRA] lora_target_mlp_projs contains invalid values: {_invalid}. "
                                 f"Only subsets of {_valid_mlp_projs} are allowed.")
            _target_projs = sorted(_requested & _found_linear_names)
            if not _target_projs:
                raise ValueError(f"[LoRA] None of lora_target_mlp_projs={args.lora_target_mlp_projs} "
                                 f"exist in model Linear layers. "
                                 f"Found Linear layer suffixes: {_found_linear_names}")
            _proj_pattern = "|".join(_target_projs)
            target_modules = rf".*\.experts\.\d+\.({_proj_pattern})$"
            logger.info(f"[LoRA] lora_experts_only=True, target projs: {_target_projs}, "
                        f"regex: {target_modules}")
        else:
            target_modules = _expert_modules + _attn_modules

        _model_type = getattr(model.config, "model_type", "unknown")
        logger.info(
            f"[LoRA] model_type={_model_type}, "
            f"lora_experts_only={args.lora_experts_only}, "
            f"target_modules={target_modules}"
        )

        max_rank = args.adalora_max_rank if args.use_adalora else args.lora_rank
        lora_alpha = args.lora_scaling_factor * max_rank
        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=max_rank, lora_alpha=lora_alpha,
                                 lora_dropout=args.lora_dropout, target_modules=target_modules)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if n_trainable == 0:
            raise RuntimeError(
                f"[LoRA] No trainable parameters after get_peft_model!\n"
                f"  target_modules={target_modules} did not match any module.\n"
                f"  Linear layer suffixes in model: {_found_linear_names}"
            )
    elif args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False} if accelerator.num_processes > 1 else None)

    # --- 3. Re-init LoRA weights BEFORE prepare ---
    if args.use_lora and args.use_adalora:
        _reinit_lora_weights(model)

    def _enable_record_routing(m):
        """
        Register forward hooks to record routing decisions without modifying model source code.

        For each MoE layer's mlp, a forward hook captures the top-k expert indices
        and writes them to mlp._last_selected_experts.

        Two model types are supported and auto-detected:

        OLMoE: mlp (OlmoeSparseMoeBlock) returns (final_hidden_states, router_logits).
               router_logits shape: [batch*seq, num_experts].
               The hook recomputes softmax + topk to obtain selected_experts.

        Qwen1.5-MoE: hook is registered on mlp.gate (Qwen2MoeTop2Router).
               The hook reads router logits from the gate output and applies topk.

        For other MoE architectures where mlp.forward returns router_logits as a 2D tensor
        [tokens, num_experts] as its second return value, the OLMoE path is used automatically.
        """
        _routing_hooks = []
        try:
            dec = m.get_decoder() if hasattr(m, "get_decoder") else getattr(m, "model", None)
            if dec is None:
                logger.warning("[record_routing] Cannot find decoder, routing hooks not registered.")
                return _routing_hooks

            for layer in dec.layers:
                mlp = layer.mlp

                if hasattr(mlp, "gate"):
                    # Qwen1.5-MoE path: hook on gate module.
                    # Finds the 2D logit tensor from gate output and applies topk to get expert indices.
                    def _make_gate_hook(target_mlp):
                        def _gate_hook(module, input, output):
                            with torch.no_grad():
                                logits = None
                                if isinstance(output, (tuple, list)):
                                    for o in reversed(output):
                                        if isinstance(o, torch.Tensor) and o.dim() == 2 and o.shape[-1] > 2:
                                            logits = o.float()
                                            break
                                elif isinstance(output, torch.Tensor) and output.dim() == 2:
                                    logits = output.float()

                                if logits is not None:
                                    # [fix] mlp.gate is a plain Linear with no top_k, so the original fell
                                    # back to 2; OLMoE routes top-8 and Qwen1.5-MoE top-4 (mlp.top_k).
                                    top_k = getattr(module, "top_k", None) or getattr(
                                        getattr(module, "config", None), "num_experts_per_tok", None) or getattr(
                                        target_mlp, "top_k", None) or 2
                                    _, selected = torch.topk(logits, top_k, dim=-1)
                                    target_mlp._last_selected_experts = selected.detach()

                                if not hasattr(module, '_diag_printed'):
                                    module._diag_printed = True
                                    print(
                                        f"[GateHook] output types: {[type(o).__name__ + str(getattr(o, 'shape', '')) for o in (output if isinstance(output, (tuple, list)) else [output])]}")
                                    if logits is not None:
                                        print(
                                            f"[GateHook] logits shape={logits.shape}, selected shape={selected.shape}, "
                                            f"min_idx={selected.min().item()}, max_idx={selected.max().item()}, "
                                            f"unique_count={selected.unique().numel()}")

                        return _gate_hook

                    h = mlp.gate.register_forward_hook(_make_gate_hook(mlp))
                    _routing_hooks.append(h)

                else:
                    # OLMoE and generic path: hook on mlp module.
                    # OlmoeSparseMoeBlock returns (final_hidden_states, router_logits),
                    # router_logits shape: [batch*seq, num_experts].
                    # Recomputes softmax + topk to obtain selected expert indices.
                    _top_k = (getattr(mlp, "top_k", None)
                              or getattr(getattr(m, "config", None), "num_experts_per_tok", None)
                              or 2)

                    def _make_mlp_hook(target_mlp, top_k):
                        def _mlp_hook(module, input, output):
                            if not isinstance(output, (tuple, list)) or len(output) < 2:
                                return
                            router_logits = output[1]
                            if not isinstance(router_logits, torch.Tensor) or router_logits.dim() != 2:
                                return
                            with torch.no_grad():
                                routing_weights = torch.softmax(router_logits.float(), dim=-1)
                                _, selected = torch.topk(routing_weights, top_k, dim=-1)
                                target_mlp._last_selected_experts = selected.detach()
                        return _mlp_hook

                    h = mlp.register_forward_hook(_make_mlp_hook(mlp, _top_k))
                    _routing_hooks.append(h)

        except Exception as e:
            logger.warning(f"Failed to enable record_routing: {e}")
        return _routing_hooks

    _routing_hooks = _enable_record_routing(model)

    # --- 4. Setup Dataloader, Optimizer, Scheduler, etc. ---
    with accelerator.main_process_first():
        train_dataset = get_cached_dataset_tulu(
            dataset_mixer_list=args.dataset_mixer_list,
            dataset_mixer_list_splits=args.dataset_mixer_list_splits,
            tc=tc,
            dataset_transform_fn=args.dataset_transform_fn,
            transform_fn_args=[{"max_seq_length": args.max_seq_length}, {}],
            target_columns=args.dataset_target_columns,
            dataset_cache_mode=args.dataset_cache_mode,
            dataset_local_cache_dir=args.dataset_local_cache_dir,
            dataset_skip_cache=args.dataset_skip_cache,
        )
        train_dataset = train_dataset.shuffle(seed=args.seed)
        train_dataset.set_format(type="pt")

    collate_fn = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest")
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=collate_fn,
                                  batch_size=args.per_device_train_batch_size)
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer_grouped_parameters = [g for g in optimizer_grouped_parameters if len(g["params"]) > 0]
    if not optimizer_grouped_parameters:
        raise RuntimeError("optimizer_grouped_parameters is empty; no trainable parameters found. Check LoRA attachment.")
    logger.info(f"[Optimizer] param groups: {[len(g['params']) for g in optimizer_grouped_parameters]} tensors")
    if args.use_8bit_optimizer:  # [4-bit patch]
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(optimizer_grouped_parameters, lr=args.learning_rate)
    else:
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, fused=args.fused_optimizer)

    overrode_max_train_steps = args.max_train_steps is None
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes if not overrode_max_train_steps else args.max_train_steps
    num_warmup_steps = int(num_training_steps_for_scheduler * args.warmup_ratio)
    lr_scheduler = get_scheduler(name=args.lr_scheduler_type, optimizer=optimizer,
                                 num_training_steps=num_training_steps_for_scheduler, num_warmup_steps=num_warmup_steps)

    def _set_moe_router_trainable(m, requires_grad: bool):
        try:
            dec = m.get_decoder() if hasattr(m, "get_decoder") else getattr(m, "model", None)
            if dec:
                for layer in dec.layers:
                    if hasattr(layer.mlp, "gate"):
                        for p in layer.mlp.gate.parameters(): p.requires_grad = requires_grad
        except Exception as e:
            logger.warning(f"Failed to set router trainable={requires_grad}: {e}")

    # Router is always frozen at initialization.
    # freeze_moe_router=True  -> frozen throughout; router_unfreeze_step=-1 never unfreezes
    # freeze_moe_router=False -> unfrozen after warmup by the training loop
    _set_moe_router_trainable(model, False)

    # --- 5. ACCELERATOR.PREPARE ---
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader,
                                                                           lr_scheduler)

    # --- 6. Init AdaLoRA state AFTER prepare ---
    adalora_state = _init_adalora_state(model, args) if args.use_lora and args.use_adalora else None
    # [2-GPU patch] layer and expert counts, for combining expert usage across GPUs
    _cfg = accelerator.unwrap_model(model).config
    moe_dims = (_cfg.num_hidden_layers,
                getattr(_cfg, "num_experts", None) or getattr(_cfg, "num_local_experts", 0))
    if accelerator.num_processes > 1:
        logger.info(f"[DDP-sync] {accelerator.num_processes} GPUs; DR-LoRA state is combined "
                    f"across them (layers, experts = {moe_dims})")

    # --- Training Plan ---
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and str(checkpointing_steps).lower() != "epoch":
        checkpointing_steps = int(checkpointing_steps)

    adalora_grow_start_step, adalora_grow_end_step, adalora_finished_step = -1, -1, None

    # Router unfreeze schedule: independent of AdaLoRA, controlled by freeze_moe_router and warmup.
    # freeze_moe_router=True  -> router frozen throughout (both baseline and ALOE)
    # freeze_moe_router=False -> unfrozen after warmup, trained jointly with LoRA
    total_steps = args.max_train_steps
    warmup_steps_opt = int(total_steps * args.warmup_ratio)
    router_unfreeze_step = warmup_steps_opt if not args.freeze_moe_router else -1

    # AdaLoRA grow schedule: only active when use_adalora=True
    if adalora_state:
        grow_freq = max(1, int(args.adalora_grow_interval))
        grow_gap = max(grow_freq, 1)
        last_grow_allowed = max(warmup_steps_opt, total_steps - grow_gap)
        if last_grow_allowed > warmup_steps_opt:
            adalora_grow_start_step = warmup_steps_opt + args.adalora_grow_offset_after_warmup
            adalora_grow_end_step = last_grow_allowed
        _plan_layerwise_grow(adalora_state, args, adalora_grow_start_step, adalora_grow_end_step)

    logger.info("***** Running training *****")
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    # Router is initially frozen (see _set_moe_router_trainable(model, False) above)
    completed_steps, starting_epoch, router_currently_frozen = 0, 0, True

    # ckpt_dir initialized to None to avoid NameError if no step checkpoint is saved before epoch checkpoint.
    ckpt_dir = None

    last_checkpoint_path = get_last_checkpoint_path(args)
    if last_checkpoint_path:
        accelerator.print(f"Resumed from checkpoint: {last_checkpoint_path}")
        # [4-bit patch] the checkpoint also holds the frozen 4-bit base weights' quantization
        # constants, which PEFT will not load back strictly. Only LoRA, router, optimizer and
        # scheduler change during training, and those are all restored.
        accelerator.load_state(last_checkpoint_path, strict=not args.use_qlora)
        if adalora_state: _restore_adalora_state(adalora_state, last_checkpoint_path, accelerator)
        training_difference = os.path.splitext(os.path.basename(last_checkpoint_path))[0]
        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_batch_idx, completed_steps = 0, starting_epoch * num_update_steps_per_epoch
        else:
            completed_steps = int(training_difference.replace("step_", ""))
            starting_epoch = completed_steps // num_update_steps_per_epoch
            resume_batch_idx = (completed_steps % num_update_steps_per_epoch) * args.gradient_accumulation_steps
        progress_bar.update(completed_steps)
    else:
        resume_batch_idx = 0

    # --- TRAINING LOOP ---
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        active_dataloader = train_dataloader
        if epoch == starting_epoch and resume_batch_idx > 0:
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_batch_idx)

        total_loss, active_union, window_usage_incr = 0, defaultdict(set), defaultdict(float)

        for batch in active_dataloader:
            with accelerator.accumulate(model):
                outputs = model(**batch, use_cache=False)
                loss = outputs.loss
                total_loss += loss.detach().float()
                accelerator.backward(loss)

                if args.use_lora and args.lora_topk_only:
                    step_active, step_usage = _collect_active_experts(model)

                    if completed_steps == 0 and accelerator.is_main_process:
                        raw_model = accelerator.unwrap_model(model)
                        for _ in range(5):
                            if hasattr(raw_model, "get_decoder") or hasattr(raw_model, "layers"):
                                break
                            raw_model = getattr(raw_model, "base_model", raw_model)
                        dec = raw_model.get_decoder() if hasattr(raw_model, "get_decoder") else raw_model
                        if hasattr(dec, "layers"):
                            for i, layer in enumerate(dec.layers):
                                mlp = layer.mlp
                                sel = getattr(mlp, "_last_selected_experts", None)
                                if sel is None:
                                    logger.warning(f"[DiagInit] layer {i}: _last_selected_experts=None")
                                else:
                                    logger.info(f"[DiagInit] layer {i}: selected shape={sel.shape}, "
                                                f"min={sel.min().item()}, max={sel.max().item()}, "
                                                f"unique_count={sel.unique().numel()}")

                    for L, ss in step_active.items(): active_union[L].update(ss)
                    for L, d in step_usage.items():
                        for eid, cnt in d.items(): window_usage_incr[(L, eid)] += cnt

                if accelerator.sync_gradients:
                    if adalora_state:
                        _adalora_mask_pruned_grads(adalora_state)

                    if args.lora_topk_only:
                        if _ddp_on():  # [2-GPU patch] usage and active experts from all GPUs
                            active_union, window_usage_incr = _ddp_sync_usage(
                                active_union, window_usage_incr, *moe_dims)
                        _zero_inactive_expert_lora_grads(model, active_union)
                        if adalora_state:
                            _adalora_update_expert_usage(adalora_state, window_usage_incr,
                                                         decay=args.adalora_usage_decay)
                        active_union, window_usage_incr = defaultdict(set), defaultdict(float)

                    next_step = completed_steps + 1
                    in_grow_window = adalora_grow_start_step > 0 and adalora_grow_start_step <= next_step <= adalora_grow_end_step
                    if adalora_state and in_grow_window and (
                            next_step - adalora_grow_start_step) % args.adalora_grow_interval == 0:
                        if _ddp_on():  # [2-GPU patch] same scores on every GPU
                            _ddp_mean_scores(adalora_state)
                        grown, saturated = _adalora_grow_once_per_layer(adalora_state,
                                                                        module_grow_frac=args.adalora_module_grow_frac)
                        if _ddp_on():  # [2-GPU patch] same masks on every GPU
                            grown, saturated = _ddp_share_masks(adalora_state, grown, saturated, next_step)
                            _ddp_check_weights(model, next_step)
                        if grown > 0:
                            for layer_info in adalora_state["layers"].values():
                                for entry in layer_info["modules"]:
                                    entry["module"]._adalora_score.zero_()
                        if saturated and adalora_finished_step is None:
                            adalora_finished_step = next_step

                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                step_now = completed_steps

                if router_currently_frozen and router_unfreeze_step >= 0 and step_now >= router_unfreeze_step:
                    _set_moe_router_trainable(model, True)
                    router_currently_frozen = False
                    logger.info(f"[RouterSchedule] Unfroze MoE router at step {step_now}.")

                if args.logging_steps and completed_steps % args.logging_steps == 0:
                    avg_loss = accelerator.gather(
                        total_loss).mean().item() / args.gradient_accumulation_steps / args.logging_steps
                    logger.info(f"Step: {completed_steps}, Loss: {avg_loss:.4f}")
                    metrics_to_log = {"train_loss": avg_loss, "learning_rate": lr_scheduler.get_last_lr()[0]}
                    if adalora_state:
                        snapshot = _collect_adalora_rank_snapshot(adalora_state)
                        if snapshot:
                            metrics_to_log["adalora_total_rank"] = snapshot["total_rank"]
                            if accelerator.is_main_process:
                                rank_log_dir = os.path.join(args.output_dir, "adalora_rank_logs")
                                os.makedirs(rank_log_dir, exist_ok=True)
                                with open(os.path.join(rank_log_dir, f"adalora_ranks_step_{completed_steps:07d}.json"),
                                          "w") as f:
                                    json.dump(snapshot, f, indent=2)
                    if args.with_tracking:
                        accelerator.log(metrics_to_log, step=completed_steps)
                    total_loss = 0

                if isinstance(checkpointing_steps, int) and completed_steps % checkpointing_steps == 0:
                    ckpt_dir = os.path.join(args.output_dir, f"step_{completed_steps}")
                    accelerator.save_state(ckpt_dir)
                    if adalora_state:
                        if _ddp_on():  # [2-GPU patch] save the scores of all GPUs
                            _ddp_mean_scores(adalora_state)
                        _save_adalora_state(adalora_state, ckpt_dir, accelerator)
                    _save_hf_eval_checkpoint(accelerator, model, tokenizer, ckpt_dir, args.freeze_moe_router)
                    accelerator.wait_for_everyone()
                    if accelerator.is_local_main_process:
                        with open(os.path.join(ckpt_dir, "COMPLETED"), "w") as f: f.write("COMPLETED")
                        clean_last_n_checkpoints(args.output_dir, args.keep_last_n_checkpoints)

                if completed_steps >= args.max_train_steps:
                    break

        if args.checkpointing_steps == "epoch":
            output_dir_epoch = os.path.join(args.output_dir, f"epoch_{epoch}")
            accelerator.save_state(output_dir_epoch)
            if adalora_state:
                if _ddp_on():  # [2-GPU patch]
                    _ddp_mean_scores(adalora_state)
                _save_adalora_state(adalora_state, output_dir_epoch, accelerator)
            _save_hf_eval_checkpoint(accelerator, model, tokenizer, output_dir_epoch, args.freeze_moe_router)
            accelerator.wait_for_everyone()
            if accelerator.is_local_main_process:
                with open(os.path.join(output_dir_epoch, "COMPLETED"), "w") as f:
                    f.write("COMPLETED")
                clean_last_n_checkpoints(args.output_dir, args.keep_last_n_checkpoints)

    if args.output_dir is not None:
        save_with_accelerate(accelerator, model, tokenizer, args.output_dir, args.use_lora,
                             chat_template_name=tc.chat_template_name)

    # Merge LoRA adapters and save the full model.
    # Correct procedure for DeepSpeed ZeRO Stage 2/3:
    #   Step A (all ranks): get_state_dict(model) triggers cross-rank ZeRO aggregation.
    #   Step B (rank-0 only):
    #       - Load the complete state_dict into unwrapped_model.
    #       - Call merge_and_unload() with correct full weights.
    #       - Save the merged model.
    if args.use_lora and args.use_qlora and not args.freeze_moe_router:
        # [4-bit patch] the adapter saved above has no router weights, but DR-LoRA trains the
        # router after warm-up. Save them next to the adapter for evaluation.
        router_state = {k: v.cpu() for k, v in accelerator.get_state_dict(model).items()
                        if ".gate." in k and "lora_" not in k}
        if accelerator.is_main_process:
            torch.save(router_state, os.path.join(args.output_dir, "router_state_dict.pt"))
            logger.info(f"[4-bit patch] saved {len(router_state)} router tensors to {args.output_dir}")

    if args.use_lora and not args.use_qlora:  # [4-bit patch] adapter is saved above
        logger.info("Merging LoRA adapters and saving the full model...")
        final_model_dir = os.path.join(args.output_dir, "final_merged_model")

        # Step A: all ranks must participate to trigger ZeRO cross-rank parameter aggregation.
        # full_state_dict is complete on rank-0; None or empty on other ranks.
        full_state_dict = accelerator.get_state_dict(model)

        if accelerator.is_main_process:
            os.makedirs(final_model_dir, exist_ok=True)
            unwrapped_model = accelerator.unwrap_model(model)

            # Step B1: load the complete state_dict into unwrapped_model.
            unwrapped_model.load_state_dict(full_state_dict, strict=False)

            # Step B2: merge LoRA. unwrapped_model now holds complete weights; merge is correct.
            # merge_and_unload() returns a plain AutoModelForCausalLM containing:
            #   - base weights + LoRA delta (merged)
            #   - router (gate) weights (included regardless of whether they were trained)
            merged_model = unwrapped_model.merge_and_unload()

            # Step B3: verify router weights exist in the merged model (debug check).
            router_param_count = sum(
                p.numel() for n, p in merged_model.named_parameters() if ".gate." in n
            )
            logger.info(f"[FinalSave] merged_model router param count: {router_param_count:,} "
                        f"(freeze_moe_router={args.freeze_moe_router})")

            # Step B4: save. Call merged_model.state_dict() directly since merged_model
            # is not managed by accelerator and rank-0 already holds complete weights.
            merged_model.save_pretrained(
                final_model_dir,
                safe_serialization=True,
            )
            tokenizer.save_pretrained(final_model_dir)
            logger.info(f"Full merged model saved to {final_model_dir}")

        accelerator.wait_for_everyone()

    if args.clean_checkpoints_at_end and accelerator.is_local_main_process:
        clean_last_n_checkpoints(args.output_dir, keep_last_n_checkpoints=0)
    if args.push_to_hub:
        push_folder_to_hub(accelerator, args.output_dir, args.hf_repo_id, args.hf_repo_revision)
    if args.with_tracking:
        accelerator.end_training()

    logger.info("Training finished.")


if __name__ == "__main__":
    parser = ArgumentParserPlus((FlatArguments, TokenizerConfig))
    args, tc = parser.parse_args_into_dataclasses()
    main(args, tc)
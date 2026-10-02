"""Check a DR-LoRA test run: 4-bit load, routing, growth, loss, memory.

Usage:  !python dr-lora/kaggle/check_run.py
"""

import glob
import json
import os
import re

WORK = os.environ.get("WORK", "/kaggle/working")
OUT = os.environ.get("OUT", f"{WORK}/out_drlora_4bit")
LOG = os.environ.get("LOG", f"{WORK}/train_test.log")
N_LAYERS = int(os.environ.get("N_LAYERS", 16))      # OLMoE-1B-7B
MIN_EVENTS = int(os.environ.get("MIN_EVENTS", 3))   # 20 steps, every 5
TOP_K = int(os.environ.get("TOP_K", 8))             # OLMoE routes top-8
R_INIT = int(os.environ.get("R_INIT", 8))

log = open(LOG).read()
all_ok = True


def check(name, passed, detail=""):
    global all_ok
    all_ok = all_ok and passed
    print(f"[{'PASS' if passed else 'FAIL'}] {name} {detail}")


logs = sorted(glob.glob(f"{OUT}/adalora_rank_logs/*.json"))
adapter = os.path.exists(f"{OUT}/adapter_model.safetensors")

check("loaded in 4-bit", "use_qlora=True" in log,
      "" if "use_qlora=True" in log else "(INFO log lines missing)")
# a 2-part run: part 1 ends at the time limit (SIGTERM traceback), so only
# check the log after the last resume
last_part = log[log.rfind("Resumed from checkpoint"):] if "Resumed from checkpoint" in log else log
check("finished without error",
      "Traceback" not in last_part and ("Training finished" in log or adapter))

gate = re.findall(r"\[GateHook\] logits shape=torch.Size\(\[(\d+), (\d+)\]\)",
                  log)
check("routing recorded", bool(gate),
      f"(router output tokens x experts: {gate[0] if gate else None})")
k = re.findall(r"selected shape=torch.Size\(\[\d+, (\d+)\]\)", log)
check(f"router top-k = {TOP_K}", bool(k) and int(k[0]) == TOP_K,
      f"(recorded top-{k[0] if k else '?'})")

grows = re.findall(r"\[AdaLoRA-Grow\] layer \d+: grew \d+ ranks", log)
events = len(grows) // N_LAYERS
grew_files = False
if logs:
    last = json.load(open(logs[-1]))["per_module_rank"].values()
    grew_files = any(m["active_rank"] > R_INIT for m in last)
# DR-LoRA's code stops growing once a layer reaches its target rank
# total, which in a full run happens after about half the planned events.
final = {}
for lid, after, target in re.findall(
        r"layer (\d+): grew \d+ ranks.*?total_rank: \d+ -> (\d+), "
        r"target_total=(\d+)", log):
    final[lid] = (int(after), int(target))
reached = len(final) == N_LAYERS and all(a >= t for a, t in final.values())
check("ranks grew", events >= MIN_EVENTS or reached
      or (not grows and grew_files),
      f"({events} growth events x {N_LAYERS} layers in log; "
      f"all layers reached target: {reached}; "
      f"rank files show growth: {grew_files})")
check("no NaN growth scores", "layer_score_sum=nan" not in log)

# 2-GPU runs: both GPUs must keep the same rank masks and LoRA weights
ddp = re.findall(r"\[DDP-sync\] step (\d+): (rank masks|LoRA weights) (\S+)", log)
if ddp:
    bad = [f"step {st}: {what} {res}" for st, what, res in ddp if res != "identical"]
    check("2 GPUs in sync", not bad,
          f"({len(ddp) // 2} growth events checked" + (f"; {bad[:3]}" if bad else "") + ")")

losses = [float(x) for x in
          re.findall(r"Step: \d+, Loss: ([0-9.eE+-]+|nan|inf)", log)]
check("loss finite", bool(losses) and all(x == x and x < 1e4 for x in losses),
      f"{losses}")
check("LoRA adapter saved", adapter)

if logs:
    snap = json.load(open(logs[-1]))
    ranks = [m["active_rank"] for m in snap["per_module_rank"].values()]
    print(f"\nrank log {os.path.basename(logs[-1])}: {len(ranks)} modules, "
          f"mean {sum(ranks) / len(ranks):.2f}, min {min(ranks)}, "
          f"max {max(ranks)}, above 8: {sum(r > 8 for r in ranks)}")

mem_file = f"{WORK}/gpu_mem.csv"
if os.path.exists(mem_file):
    mem = [int(x) for x in open(mem_file).read().split() if x.isdigit()]
    if mem:
        check("GPU memory < 15 GB", max(mem) < 15000,
              f"(peak {max(mem) / 1024:.1f} GB)")

speed = re.findall(r"(\d+\.\d+)s/it", log)
if speed:
    sec = float(speed[-1])
    print(f"time per step ~{sec:.1f} s -> 3,750 steps ~{sec * 3750 / 3600:.1f} h "
          f"(batch 4 per GPU)")
print("\nALL CHECKS PASSED" if all_ok
      else "\nSOME CHECKS FAILED: send train_test.log")

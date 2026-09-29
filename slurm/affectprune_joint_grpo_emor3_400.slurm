#!/usr/bin/env bash
#SBATCH --job-name=affectprune_joint_emor3_400
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=logs/affectprune_joint_emor3_400_%j.out
#SBATCH --error=logs/affectprune_joint_emor3_400_%j.err
set -euo pipefail
ROOT=${EMOR3_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export EMOR3_PROJECT_ROOT="$ROOT"
export AFFECT_JOINT_BASE_MODEL="$ROOT/checkpoints/emor3_base100_frozen/global_step_100/actor/huggingface"
export AFFECT_JOINT_SFT_ADAPTER="$ROOT/outputs/affectprune_emor3start_randomdrop_sft_train2000_20260929/adapter_step2400.pt"
export AFFECT_JOINT_OUTPUT="$ROOT/outputs/affectprune_emor3start_joint_sft2400_400step_20260929"
export AFFECT_JOINT_STEPS=400
export AFFECT_JOINT_GROUP_SIZE=2
export AFFECT_JOINT_MAX_TOKENS=384
export AFFECT_JOINT_REFLECT_TOKENS=32
export AFFECT_JOINT_SAVE_EVERY=20
export AFFECT_JOINT_SEED=20260928
export AFFECT_JOINT_ALIGN_WEIGHT=0.15
export AFFECT_JOINT_DROP_WEIGHT=0.05
export AFFECT_JOINT_KL_COEF=0.03
test -s "$AFFECT_JOINT_SFT_ADAPTER"
test -s "$ROOT/outputs/affectprune_emor3start_joint_gate1_20260929/joint_step1.pt"
export AFFECT_JOINT_VALIDATION_ROOT="$ROOT/outputs/affectprune_emor3start_sft_validation_20260929"
${EMOR3_PYTHON:-python} - <<'PY'
import glob
import json
import os
from pathlib import Path

root = Path(os.environ["AFFECT_JOINT_VALIDATION_ROOT"])
paths = sorted(glob.glob(str(root / "*.jsonl")))
if len(paths) != 5:
    raise RuntimeError(f"Require five completed TRAIN-only SFT validation ratios, found {len(paths)}")
expected_ratios = {0.0, 0.1, 0.2, 0.3, 0.4}
seen_ratios = set()
reference_ids = None
for path in paths:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
    ids = {str(row["id"]) for row in rows}
    if len(rows) != 64 or len(ids) != 64 or (reference_ids is not None and ids != reference_ids):
        raise RuntimeError(f"Incomplete or misaligned TRAIN validation: {path}")
    reference_ids = ids
    ratios = {float(row["ratio_target"]) for row in rows}
    if len(ratios) != 1:
        raise RuntimeError(f"Mixed ratios in {path}")
    ratio = ratios.pop()
    seen_ratios.add(ratio)
    strict = sum(bool(row["strict_set"]) for row in rows)
    truncated = sum(bool(row["hit_length_cap"]) for row in rows)
    if strict < 60 or truncated:
        raise RuntimeError(f"SFT cold start did not protect format at ratio {ratio}: strict={strict}/64, truncated={truncated}")
    print(f"EMOR3_SFT_VALIDATION {ratio:.1f} strict={strict}/64 truncated=0 correct={sum(bool(row['correct']) for row in rows)}/64", flush=True)
if seen_ratios != expected_ratios:
    raise RuntimeError(f"Missing or repeated validation ratios: {seen_ratios}")
gate_history = Path(os.environ["EMOR3_PROJECT_ROOT"]) / "outputs/affectprune_emor3start_joint_gate1_20260929/train.jsonl"
gate_rows = [json.loads(line) for line in gate_history.read_text(encoding="utf-8").splitlines() if line]
if (len(gate_rows) != 1 or int(gate_rows[0]["step"]) != 1
        or float(gate_rows[0]["actor_gradient_norm"]) <= 0
        or float(gate_rows[0]["selector_gradient_norm"]) <= 0):
    raise RuntimeError("Joint gate did not verify nonzero actor and selector gradients")
PY
bash "$ROOT/slurm/affectprune_joint_grpo.slurm"

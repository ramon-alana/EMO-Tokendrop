#!/usr/bin/env bash
#SBATCH --job-name=affectprune_joint_free_emor3_400
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=logs/affectprune_joint_free_emor3_400_%j.out
#SBATCH --error=logs/affectprune_joint_free_emor3_400_%j.err
set -euo pipefail
ROOT=${EMOR3_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PY=${EMOR3_PYTHON:-python}
export AFFECT_JOINT_BASE_MODEL="$ROOT/checkpoints/emor3_base100_frozen/global_step_100/actor/huggingface"
export AFFECT_JOINT_SFT_ADAPTER="$ROOT/outputs/affectprune_emor3start_randomdrop_sft_train2000_20260929/adapter_step2400.pt"
export AFFECT_JOINT_DEV_FILE="$ROOT/data/EmoSet2k_free_train_dev200_20260929.jsonl"
export AFFECT_JOINT_OUTPUT="$ROOT/outputs/affectprune_emor3start_free_ratio_stage1_step400_20260929"
export AFFECT_JOINT_SELECTOR_POLICY=free_bernoulli
export AFFECT_JOINT_STEPS=400
export AFFECT_JOINT_GROUP_SIZE=2
export AFFECT_JOINT_MAX_TOKENS=384
export AFFECT_JOINT_REFLECT_TOKENS=32
export AFFECT_JOINT_SAVE_EVERY=20
export AFFECT_JOINT_FULL_SAVE_EVERY=200
export AFFECT_JOINT_SEED=20260928
export AFFECT_JOINT_ALIGN_WEIGHT=0.15
export AFFECT_JOINT_DROP_WEIGHT=0.05
export AFFECT_JOINT_KL_COEF=0.03
test -s "$AFFECT_JOINT_SFT_ADAPTER"
test -s "$ROOT/outputs/affectprune_free_resume_gate_stage2_20260929/resume_step2.pt"
test -s "$AFFECT_JOINT_DEV_FILE"
export AFFECT_FREE_DEV_OUTPUT="$ROOT/outputs/affectprune_free_gate_dev64_20260929"
"$PY" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["AFFECT_FREE_DEV_OUTPUT"])
ids = None
for decode in ("argmax", "sample"):
    path = root / f"learned_{decode}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    current = [str(row["id"]) for row in rows]
    if len(rows) != 64 or len(set(current)) != 64 or (ids is not None and current != ids):
        raise RuntimeError(f"Incomplete or unmatched TRAIN-only development results: {path}")
    ids = current
    if any(row["selector_policy"] != "free_bernoulli" or row["budget_decode"] != decode for row in rows):
        raise RuntimeError("Wrong free-ratio evaluation protocol")
    if any(row["hit_length_cap"] for row in rows):
        raise RuntimeError("TRAIN-only development response was truncated")
    ratios = [float(row["ratio_actual"]) for row in rows]
    print("FREE_TRAIN_DEV", decode,
          "correct", sum(bool(row["correct"]) for row in rows),
          "strict", sum(bool(row["strict_set"]) for row in rows),
          "drop_min", min(ratios), "drop_mean", sum(ratios) / len(ratios),
          "drop_max", max(ratios),
          "unique_token_counts", len({int(row["visual_tokens_after"]) for row in rows}), flush=True)
    if decode == "sample" and not 0.05 <= sum(ratios) / len(ratios) <= 0.85:
        raise RuntimeError("Free-ratio sampled policy collapsed before formal RL")
PY
bash "$ROOT/slurm/affectprune_joint_grpo_free.slurm"

#!/usr/bin/env bash
# Each four array entries evaluate one 200-image shard: full/random/FoE/learned.
#SBATCH --job-name=emor3_free_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/emor3_free_eval_%A_%a.out
#SBATCH --error=logs/emor3_free_eval_%A_%a.err
set -euo pipefail
ROOT=${EMOR3_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PY=${EMOR3_PYTHON:-python}
: "${SLURM_ARRAY_TASK_ID:?Submit with 0-19 for 1000 images or 0-39 for 2000}"
: "${AFFECT_JOINT_EVAL_BASE_MODEL:?Set the same starting checkpoint used in training}"
: "${AFFECT_JOINT_EVAL_CHECKPOINT:?Set one complete actor-selector checkpoint}"
: "${AFFECT_JOINT_EVAL_TEST_FILE:?Set the frozen TEST JSONL}"
: "${AFFECT_JOINT_EVAL_OUTPUT_ROOT:?Use a new result directory}"
: "${AFFECT_JOINT_EVAL_BUDGET_DECODE:?Preselect sample or argmax on TRAIN only}"
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID > 39 )); then exit 2; fi
methods=(full random_matched foe_matched learned)
shard=$((SLURM_ARRAY_TASK_ID / 4))
export AFFECT_JOINT_EVAL_METHOD="${methods[$((SLURM_ARRAY_TASK_ID % 4))]}"
export AFFECT_JOINT_EVAL_START_SAMPLE="$((shard * 200))"
export AFFECT_JOINT_EVAL_MAX_SAMPLES=200
export AFFECT_JOINT_EVAL_MAX_NEW_TOKENS=${AFFECT_JOINT_EVAL_MAX_NEW_TOKENS:-384}
export AFFECT_JOINT_EVAL_PREFILL_STEP1=${AFFECT_JOINT_EVAL_PREFILL_STEP1:-0}
printf -v shard_tag '%02d' "$shard"
export AFFECT_JOINT_EVAL_OUTPUT="$AFFECT_JOINT_EVAL_OUTPUT_ROOT/${AFFECT_JOINT_EVAL_METHOD}_shard$shard_tag.jsonl"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
"$PY" -u "$ROOT/affectprune_joint_eval_free.py"

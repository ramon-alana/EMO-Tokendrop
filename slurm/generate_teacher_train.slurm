#!/usr/bin/env bash
# Regenerate TRAIN-only pseudo-label candidates without a legacy selector.
# This is a clean re-generation route; it may not be bit-identical to the
# historical teacher inference used for the recorded 2026-09-29 checkpoint.
#SBATCH --job-name=emor3_teacher_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-9%2
#SBATCH --time=04:00:00
#SBATCH --output=logs/emor3_teacher_%A_%a.out
#SBATCH --error=logs/emor3_teacher_%A_%a.err
set -euo pipefail
ROOT=${EMOR3_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PY=${EMOR3_PYTHON:-python}
: "${EMOR3_BASE_MODEL:?Set the local EMO-R3 checkpoint directory}"
: "${SLURM_ARRAY_TASK_ID:?Submit as an array with indices 0-9}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export AFFECT_JOINT_EVAL_BASE_MODEL="$EMOR3_BASE_MODEL"
export AFFECT_JOINT_EVAL_BASELINE_ONLY=1
export AFFECT_JOINT_EVAL_TEST_FILE="$ROOT/data/EmoSet2k_full/train.jsonl"
export AFFECT_JOINT_EVAL_METHOD=full
export AFFECT_JOINT_EVAL_BUDGET_DECODE=argmax
export AFFECT_JOINT_EVAL_MAX_NEW_TOKENS=384
export AFFECT_JOINT_EVAL_PREFILL_STEP1=0
export AFFECT_JOINT_EVAL_START_SAMPLE="$((SLURM_ARRAY_TASK_ID * 200))"
export AFFECT_JOINT_EVAL_MAX_SAMPLES=200
printf -v shard '%02d' "$SLURM_ARRAY_TASK_ID"
export AFFECT_JOINT_EVAL_OUTPUT="$ROOT/outputs/affectprune_sft_teacher_train2000/teacher_shard$shard/full.jsonl"
"$PY" -u "$ROOT/affectprune_joint_eval_free.py"

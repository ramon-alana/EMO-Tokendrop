#!/usr/bin/env bash
# Same prompt/parser/384-token limit as the joint evaluator, without TokenDrop.
#SBATCH --job-name=emor3_start_baseline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/emor3_baseline_%A_%a.out
#SBATCH --error=logs/emor3_baseline_%A_%a.err
set -euo pipefail
ROOT=${EMOR3_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PY=${EMOR3_PYTHON:-python}
: "${SLURM_ARRAY_TASK_ID:?Submit as 0-4 for 1000 or 0-9 for 2000}"
: "${EMOR3_BASE_MODEL:?Set the local starting checkpoint}"
: "${AFFECT_JOINT_EVAL_TEST_FILE:?Set the frozen TEST JSONL}"
: "${AFFECT_JOINT_EVAL_OUTPUT_ROOT:?Use a new baseline result directory}"
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID > 9 )); then exit 2; fi
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export AFFECT_JOINT_EVAL_BASE_MODEL="$EMOR3_BASE_MODEL"
export AFFECT_JOINT_EVAL_BASELINE_ONLY=1
export AFFECT_JOINT_EVAL_METHOD=full
export AFFECT_JOINT_EVAL_BUDGET_DECODE=argmax
export AFFECT_JOINT_EVAL_MAX_NEW_TOKENS=384
export AFFECT_JOINT_EVAL_PREFILL_STEP1=0
export AFFECT_JOINT_EVAL_START_SAMPLE="$((SLURM_ARRAY_TASK_ID * 200))"
export AFFECT_JOINT_EVAL_MAX_SAMPLES=200
printf -v shard '%02d' "$SLURM_ARRAY_TASK_ID"
export AFFECT_JOINT_EVAL_OUTPUT="$AFFECT_JOINT_EVAL_OUTPUT_ROOT/full_shard$shard.jsonl"
"$PY" -u "$ROOT/affectprune_joint_eval_free.py"

#!/usr/bin/env bash
#SBATCH --job-name=affectprune_emor3_fourbin_eval1000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/affectprune_emor3_fourbin_eval1000_%A_%a.out
#SBATCH --error=logs/affectprune_emor3_fourbin_eval1000_%A_%a.err
set -euo pipefail
ROOT=${EMOR3_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
: "${SLURM_ARRAY_TASK_ID:?Submit only array indices 0-19 for the first 1000 TEST images}"
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID > 19 )); then
  echo 'Only the first 1000 TEST images are authorized now' >&2
  exit 2
fi
export AFFECT_JOINT_EVAL_BASE_MODEL="$ROOT/checkpoints/emor3_base100_frozen/global_step_100/actor/huggingface"
export AFFECT_JOINT_EVAL_CHECKPOINT="$ROOT/outputs/affectprune_emor3start_joint_sft2400_400step_20260929/joint_step400.pt"
export AFFECT_JOINT_EVAL_TEST_FILE="$ROOT/data/EmoSet2k_full/test.jsonl"
export AFFECT_JOINT_EVAL_OUTPUT_ROOT="$ROOT/outputs/affectprune_emor3_fourbin_step400_first1000_20260929"
export AFFECT_JOINT_EVAL_MAX_SAMPLES=200
export AFFECT_JOINT_EVAL_MAX_NEW_TOKENS=384
export AFFECT_JOINT_EVAL_BUDGET_DECODE=argmax
export AFFECT_JOINT_EVAL_PREFILL_STEP1=0
export AFFECT_JOINT_EVAL_SAVE_SPATIAL=0
bash "$ROOT/slurm/affectprune_joint_eval.slurm"

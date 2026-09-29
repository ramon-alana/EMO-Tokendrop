#!/usr/bin/env bash
#SBATCH --job-name=affectprune_joint_selection_probe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/affectprune_joint_selection_probe_%j.out
#SBATCH --error=logs/affectprune_joint_selection_probe_%j.err
set -euo pipefail
ROOT=${EMOR3_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export AFFECT_JOINT_EVAL_BASE_MODEL="$ROOT/checkpoints/emor3_base100_frozen/global_step_100/actor/huggingface"
export AFFECT_JOINT_EVAL_CHECKPOINT="$ROOT/outputs/affectprune_emor3start_joint_sft2400_400step_20260929/joint_step400.pt"
export AFFECT_JOINT_EVAL_TEST_FILE="$ROOT/data/EmoSet2k_balanced200/test.jsonl"
export AFFECT_JOINT_EVAL_MAX_NEW_TOKENS=384
export AFFECT_JOINT_EVAL_BUDGET_DECODE=argmax
export AFFECT_JOINT_EVAL_PREFILL_STEP1=0
export AFFECT_JOINT_EVAL_START_SAMPLE=0
export AFFECT_JOINT_EVAL_MAX_SAMPLES=200
export AFFECT_JOINT_EVAL_SAVE_SPATIAL=1
OUT="$ROOT/outputs/affectprune_emor3start_selection_probe200_20260929"
test -s "$AFFECT_JOINT_EVAL_CHECKPOINT"
for method in learned learned_inverse; do
    export AFFECT_JOINT_EVAL_METHOD="$method"
    export AFFECT_JOINT_EVAL_OUTPUT="$OUT/${method}_shard00.jsonl"
    bash "$ROOT/slurm/affectprune_joint_eval.slurm"
done

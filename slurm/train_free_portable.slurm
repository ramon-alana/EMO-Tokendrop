#!/usr/bin/env bash
# Configure checkpoint/data paths in the environment before submission.
#SBATCH --job-name=emor3_free_tokendrop
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=logs/emor3_free_train_%j.out
#SBATCH --error=logs/emor3_free_train_%j.err
set -euo pipefail
ROOT=${EMOR3_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PY=${EMOR3_PYTHON:-python}
: "${AFFECT_JOINT_BASE_MODEL:?Set the local EMO-R3 checkpoint}"
: "${AFFECT_JOINT_SFT_ADAPTER:?Set the completed random-drop SFT adapter}"
: "${AFFECT_JOINT_DEV_FILE:?Set the frozen TRAIN-internal development split}"
: "${AFFECT_JOINT_OUTPUT:?Use a new, empty output directory}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export AFFECT_JOINT_SELECTOR_POLICY=free_bernoulli
export AFFECT_JOINT_STEPS=${AFFECT_JOINT_STEPS:-400}
export AFFECT_JOINT_GROUP_SIZE=${AFFECT_JOINT_GROUP_SIZE:-2}
export AFFECT_JOINT_MAX_TOKENS=${AFFECT_JOINT_MAX_TOKENS:-384}
export AFFECT_JOINT_REFLECT_TOKENS=${AFFECT_JOINT_REFLECT_TOKENS:-32}
export AFFECT_JOINT_SAVE_EVERY=${AFFECT_JOINT_SAVE_EVERY:-20}
export AFFECT_JOINT_FULL_SAVE_EVERY=${AFFECT_JOINT_FULL_SAVE_EVERY:-200}
export AFFECT_JOINT_ALIGN_WEIGHT=${AFFECT_JOINT_ALIGN_WEIGHT:-0.15}
export AFFECT_JOINT_DROP_WEIGHT=${AFFECT_JOINT_DROP_WEIGHT:-0.05}
export AFFECT_JOINT_KL_COEF=${AFFECT_JOINT_KL_COEF:-0.03}
export AFFECT_JOINT_SEED=${AFFECT_JOINT_SEED:-20260928}
"$PY" -u "$ROOT/affectprune_joint_grpo_free.py"

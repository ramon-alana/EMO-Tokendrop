#!/usr/bin/env bash
#SBATCH --job-name=affectprune_sft_validate_emor3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=logs/affectprune_sft_validate_emor3_%A_%a.out
#SBATCH --error=logs/affectprune_sft_validate_emor3_%A_%a.err
set -euo pipefail
ROOT=${EMOR3_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export AFFECT_VAL_MODEL_PATH="$ROOT/checkpoints/emor3_base100_frozen/global_step_100/actor/huggingface"
export AFFECT_VAL_SFT_ADAPTER="$ROOT/outputs/affectprune_emor3start_randomdrop_sft_train2000_20260929/adapter_step2400.pt"
export AFFECT_VAL_OUTPUT_ROOT="$ROOT/outputs/affectprune_emor3start_sft_validation_20260929"
export AFFECT_VAL_PREFILL_STEP1=0
export AFFECT_VAL_PER_CLASS=8
export AFFECT_VAL_MAX_NEW_TOKENS=384
test -s "$AFFECT_VAL_SFT_ADAPTER"
bash "$ROOT/slurm/affectprune_sft_validate.slurm"

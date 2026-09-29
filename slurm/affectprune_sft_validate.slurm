#!/usr/bin/env bash
#SBATCH --job-name=affectprune_sft_validate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=logs/affectprune_sft_validate_%A_%a.out
#SBATCH --error=logs/affectprune_sft_validate_%A_%a.err
set -euo pipefail
module purge
module load compilers/gcc/11.3.0 compilers/cuda/12.1 cudnn/8.8.1.3_cuda12.x nccl/2.18.3-1_cuda12.1
ROOT=${EMOR3_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PY=${EMOR3_PYTHON:-python}
export PYTHONPATH="$ROOT:$ROOT/pydeps_train:$ROOT/pydeps_v2:$ROOT/emo-r3"
# Use the active CUDA/PyTorch environment library path.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HOME="$ROOT/hf_cache" TMPDIR="$ROOT/tmp_pip" PYTHONUNBUFFERED=1
export AFFECT_VAL_SFT_ADAPTER="${AFFECT_VAL_SFT_ADAPTER:-$ROOT/outputs/affectprune_randomdrop_sft_train2000_20260928/adapter_step2400.pt}"
export AFFECT_VAL_PREFILL_STEP1="${AFFECT_VAL_PREFILL_STEP1:-0}"
export AFFECT_VAL_PER_CLASS="${AFFECT_VAL_PER_CLASS:-8}"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  ratios=(0 0.1 0.2 0.3 0.4)
  if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID > 4 )); then
    echo 'Validation array index must be 0–4' >&2
    exit 2
  fi
  export AFFECT_VAL_DROP_RATIO="${ratios[$SLURM_ARRAY_TASK_ID]}"
fi
export AFFECT_VAL_DROP_RATIO="${AFFECT_VAL_DROP_RATIO:-0.2}"
if [[ -z "${AFFECT_VAL_OUTPUT:-}" ]]; then
  ratio_tag="${AFFECT_VAL_DROP_RATIO/./p}"
  if [[ -n "${AFFECT_VAL_BASELINE_MODEL:-}" ]]; then
    checkpoint_tag="baseline_$(basename "$AFFECT_VAL_BASELINE_MODEL")"
  else
    checkpoint_tag="step$(basename "$AFFECT_VAL_SFT_ADAPTER" .pt)"
  fi
  export AFFECT_VAL_OUTPUT="${AFFECT_VAL_OUTPUT_ROOT:-$ROOT/outputs/affectprune_sft_validation_20260928}/${checkpoint_tag}_ratio${ratio_tag}_prefix${AFFECT_VAL_PREFILL_STEP1}_n${AFFECT_VAL_PER_CLASS}.jsonl"
fi
test -s "$AFFECT_VAL_SFT_ADAPTER"
"$PY" -u "$ROOT/affectprune_sft_validate.py"

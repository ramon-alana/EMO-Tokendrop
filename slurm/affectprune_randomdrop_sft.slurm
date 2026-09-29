#!/usr/bin/env bash
#SBATCH --job-name=affectprune_randomdrop_sft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=logs/affectprune_randomdrop_sft_%j.out
#SBATCH --error=logs/affectprune_randomdrop_sft_%j.err
set -euo pipefail
module purge
module load compilers/gcc/11.3.0 compilers/cuda/12.1 cudnn/8.8.1.3_cuda12.x nccl/2.18.3-1_cuda12.1
ROOT=${EMOR3_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PY=${EMOR3_PYTHON:-python}
export PYTHONPATH="$ROOT:$ROOT/pydeps_train:$ROOT/pydeps_v2:$ROOT/emo-r3"
# Use the active CUDA/PyTorch environment library path.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HOME="$ROOT/hf_cache" TMPDIR="$ROOT/tmp_pip" PYTHONUNBUFFERED=1
export AFFECT_SFT_BASE_MODEL="${AFFECT_SFT_BASE_MODEL:-$ROOT/models/Qwen2.5-VL-3B-Instruct}"
: "${AFFECT_SFT_STEPS:?Provide SFT optimizer steps}"
: "${AFFECT_SFT_OUTPUT:?Provide a unique SFT output directory}"
: "${AFFECT_SFT_TRACE_GLOB:?Provide train-only strict teacher traces}"
"$PY" -u "$ROOT/affectprune_randomdrop_sft.py"

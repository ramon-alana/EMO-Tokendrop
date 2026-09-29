#!/usr/bin/env bash
#SBATCH --job-name=affectprune_joint_grpo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=logs/affectprune_joint_grpo_%j.out
#SBATCH --error=logs/affectprune_joint_grpo_%j.err
set -euo pipefail
module purge
module load compilers/gcc/11.3.0 compilers/cuda/12.1 cudnn/8.8.1.3_cuda12.x nccl/2.18.3-1_cuda12.1
ROOT=${EMOR3_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PY=${EMOR3_PYTHON:-python}
export PYTHONPATH="$ROOT:$ROOT/pydeps_train:$ROOT/pydeps_v2:$ROOT/emo-r3"
# Use the active CUDA/PyTorch environment library path.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HOME="$ROOT/hf_cache" TMPDIR="$ROOT/tmp_pip" PYTHONUNBUFFERED=1
: "${AFFECT_JOINT_SFT_ADAPTER:?Provide completed SFT adapter checkpoint}"
: "${AFFECT_JOINT_OUTPUT:?Provide a unique joint RL output directory}"
: "${AFFECT_JOINT_STEPS:?Provide joint RL steps}"
test -s "$AFFECT_JOINT_SFT_ADAPTER"
"$PY" -u "$ROOT/affectprune_joint_grpo.py"

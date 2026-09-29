#!/usr/bin/env bash
#SBATCH --job-name=affectprune_joint_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/affectprune_joint_eval_%j.out
#SBATCH --error=logs/affectprune_joint_eval_%j.err

set -euo pipefail
module purge
module load compilers/gcc/11.3.0 compilers/cuda/12.1 cudnn/8.8.1.3_cuda12.x nccl/2.18.3-1_cuda12.1
ROOT=${EMOR3_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PY=${EMOR3_PYTHON:-python}
export PYTHONPATH="$ROOT:$ROOT/pydeps_train:$ROOT/pydeps_v2:$ROOT/emo-r3"
# Use the active CUDA/PyTorch environment library path.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HOME="$ROOT/hf_cache" TMPDIR="$ROOT/tmp_pip" PYTHONUNBUFFERED=1
: "${AFFECT_JOINT_EVAL_CHECKPOINT:?Provide a complete joint actor-selector checkpoint}"
: "${AFFECT_JOINT_EVAL_OUTPUT_ROOT:-}"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID > 39 )); then
    echo 'Array index must be 0–39 (ten 200-image shards × four methods)' >&2
    exit 2
  fi
  : "${AFFECT_JOINT_EVAL_OUTPUT_ROOT:?Provide array result directory}"
  shard=$((SLURM_ARRAY_TASK_ID / 4))
  case_index=$((SLURM_ARRAY_TASK_ID % 4))
  methods=(full random_matched foe_matched learned)
  export AFFECT_JOINT_EVAL_METHOD="${methods[$case_index]}"
  export AFFECT_JOINT_EVAL_START_SAMPLE="$((shard * 200))"
  export AFFECT_JOINT_EVAL_OUTPUT="$AFFECT_JOINT_EVAL_OUTPUT_ROOT/${AFFECT_JOINT_EVAL_METHOD}_shard$(printf '%02d' "$shard").jsonl"
fi
: "${AFFECT_JOINT_EVAL_OUTPUT:?Provide a unique result path}"
test -s "$AFFECT_JOINT_EVAL_CHECKPOINT"
"$PY" -u "$ROOT/affectprune_joint_eval.py"

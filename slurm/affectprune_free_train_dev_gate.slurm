#!/usr/bin/env bash
#SBATCH --job-name=affectprune_free_dev_gate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=logs/affectprune_free_dev_gate_%A_%a.out
#SBATCH --error=logs/affectprune_free_dev_gate_%A_%a.err
set -euo pipefail
module purge
module load compilers/gcc/11.3.0 compilers/cuda/12.1 cudnn/8.8.1.3_cuda12.x nccl/2.18.3-1_cuda12.1
ROOT=${EMOR3_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PY=${EMOR3_PYTHON:-python}
export PYTHONPATH="$ROOT:$ROOT/pydeps_train:$ROOT/pydeps_v2:$ROOT/emo-r3"
# Use the active CUDA/PyTorch environment library path.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HOME="$ROOT/hf_cache" TMPDIR="$ROOT/tmp_pip" PYTHONUNBUFFERED=1
export AFFECT_JOINT_EVAL_BASE_MODEL="$ROOT/checkpoints/emor3_base100_frozen/global_step_100/actor/huggingface"
export AFFECT_JOINT_EVAL_CHECKPOINT="$ROOT/outputs/affectprune_emor3start_free_ratio_gate1_20260929/joint_step1.pt"
export AFFECT_JOINT_EVAL_TEST_FILE="$ROOT/data/EmoSet2k_free_train_dev200_20260929.jsonl"
export AFFECT_JOINT_EVAL_METHOD=learned
export AFFECT_JOINT_EVAL_PREFILL_STEP1=0
export AFFECT_JOINT_EVAL_MAX_NEW_TOKENS=384
export AFFECT_JOINT_EVAL_START_SAMPLE=0
export AFFECT_JOINT_EVAL_MAX_SAMPLES=64
export AFFECT_JOINT_EVAL_SAVE_SPATIAL=0
case "${SLURM_ARRAY_TASK_ID:?}" in
  0) export AFFECT_JOINT_EVAL_BUDGET_DECODE=argmax; suffix=argmax ;;
  1) export AFFECT_JOINT_EVAL_BUDGET_DECODE=sample; suffix=sample ;;
  *) echo 'Only array indices 0 and 1 are valid' >&2; exit 2 ;;
esac
export AFFECT_JOINT_EVAL_OUTPUT="$ROOT/outputs/affectprune_free_gate_dev64_20260929/learned_${suffix}.jsonl"
test -s "$AFFECT_JOINT_EVAL_CHECKPOINT"
test -s "$AFFECT_JOINT_EVAL_TEST_FILE"
"$PY" -u "$ROOT/affectprune_joint_eval_free.py"

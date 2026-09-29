#!/usr/bin/env bash
#SBATCH --job-name=affectprune_emor3_free_eval1000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/affectprune_emor3_free_eval1000_%A_%a.out
#SBATCH --error=logs/affectprune_emor3_free_eval1000_%A_%a.err
set -euo pipefail
ROOT=${EMOR3_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
: "${SLURM_ARRAY_TASK_ID:?Submit only array indices 0-19 for the first 1000 TEST images}"
: "${AFFECT_FREE_EVAL_DECODE:?Choose argmax or sample using TRAIN-only validation before submission}"
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID > 19 )); then
  echo 'Only the first 1000 TEST images are authorized now' >&2
  exit 2
fi
if [[ "$AFFECT_FREE_EVAL_DECODE" != argmax && "$AFFECT_FREE_EVAL_DECODE" != sample ]]; then
  echo 'AFFECT_FREE_EVAL_DECODE must be argmax or sample' >&2
  exit 2
fi
export AFFECT_JOINT_EVAL_BASE_MODEL="$ROOT/checkpoints/emor3_base100_frozen/global_step_100/actor/huggingface"
export AFFECT_JOINT_EVAL_CHECKPOINT="$ROOT/outputs/affectprune_emor3start_free_ratio_stage1_step400_20260929/joint_step400.pt"
export AFFECT_JOINT_EVAL_TEST_FILE="$ROOT/data/EmoSet2k_full/test.jsonl"
export AFFECT_JOINT_EVAL_OUTPUT_ROOT="$ROOT/outputs/affectprune_emor3_free_step400_first1000_${AFFECT_FREE_EVAL_DECODE}_20260929"
export AFFECT_JOINT_EVAL_MAX_SAMPLES=200
export AFFECT_JOINT_EVAL_MAX_NEW_TOKENS=384
export AFFECT_JOINT_EVAL_BUDGET_DECODE="$AFFECT_FREE_EVAL_DECODE"
export AFFECT_JOINT_EVAL_PREFILL_STEP1=0
export AFFECT_JOINT_EVAL_SAVE_SPATIAL=0
test -s "$AFFECT_JOINT_EVAL_CHECKPOINT"
module purge
module load compilers/gcc/11.3.0 compilers/cuda/12.1 cudnn/8.8.1.3_cuda12.x nccl/2.18.3-1_cuda12.1
PY=${EMOR3_PYTHON:-python}
export PYTHONPATH="$ROOT:$ROOT/pydeps_train:$ROOT/pydeps_v2:$ROOT/emo-r3"
# Use the active CUDA/PyTorch environment library path.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HOME="$ROOT/hf_cache" TMPDIR="$ROOT/tmp_pip" PYTHONUNBUFFERED=1
shard=$((SLURM_ARRAY_TASK_ID / 4))
case_index=$((SLURM_ARRAY_TASK_ID % 4))
methods=(full random_matched foe_matched learned)
export AFFECT_JOINT_EVAL_METHOD="${methods[$case_index]}"
export AFFECT_JOINT_EVAL_START_SAMPLE="$((shard * 200))"
export AFFECT_JOINT_EVAL_OUTPUT="$AFFECT_JOINT_EVAL_OUTPUT_ROOT/${AFFECT_JOINT_EVAL_METHOD}_shard$(printf '%02d' "$shard").jsonl"
"$PY" -u "$ROOT/affectprune_joint_eval_free.py"

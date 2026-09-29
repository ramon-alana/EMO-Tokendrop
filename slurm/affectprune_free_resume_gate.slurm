#!/usr/bin/env bash
#SBATCH --job-name=affectprune_free_resume_gate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=logs/affectprune_free_resume_gate_%j.out
#SBATCH --error=logs/affectprune_free_resume_gate_%j.err
set -euo pipefail
ROOT=${EMOR3_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export AFFECT_JOINT_BASE_MODEL="$ROOT/checkpoints/emor3_base100_frozen/global_step_100/actor/huggingface"
export AFFECT_JOINT_SFT_ADAPTER="$ROOT/outputs/affectprune_emor3start_randomdrop_sft_train2000_20260929/adapter_step2400.pt"
export AFFECT_JOINT_DEV_FILE="$ROOT/data/EmoSet2k_free_train_dev200_20260929.jsonl"
export AFFECT_JOINT_SELECTOR_POLICY=free_bernoulli
export AFFECT_JOINT_GROUP_SIZE=2
export AFFECT_JOINT_MAX_TOKENS=384
export AFFECT_JOINT_REFLECT_TOKENS=32
export AFFECT_JOINT_SAVE_EVERY=1
export AFFECT_JOINT_FULL_SAVE_EVERY=1
export AFFECT_JOINT_SEED=20260928
export AFFECT_JOINT_ALIGN_WEIGHT=0.15
export AFFECT_JOINT_DROP_WEIGHT=0.05
export AFFECT_JOINT_KL_COEF=0.03
export AFFECT_JOINT_STEPS=1
export AFFECT_JOINT_OUTPUT="$ROOT/outputs/affectprune_free_resume_gate_stage1_20260929"
unset AFFECT_JOINT_RESUME_CHECKPOINT
bash "$ROOT/slurm/affectprune_joint_grpo_free.slurm"
test -s "$AFFECT_JOINT_OUTPUT/resume_step1.pt"
export AFFECT_JOINT_RESUME_CHECKPOINT="$AFFECT_JOINT_OUTPUT/resume_step1.pt"
export AFFECT_JOINT_STEPS=2
export AFFECT_JOINT_OUTPUT="$ROOT/outputs/affectprune_free_resume_gate_stage2_20260929"
bash "$ROOT/slurm/affectprune_joint_grpo_free.slurm"
test -s "$AFFECT_JOINT_OUTPUT/resume_step2.pt"

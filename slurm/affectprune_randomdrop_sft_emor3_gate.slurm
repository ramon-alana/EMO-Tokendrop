#!/usr/bin/env bash
#SBATCH --job-name=affectprune_randomdrop_sft_emor3_gate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=logs/affectprune_randomdrop_sft_emor3_gate_%j.out
#SBATCH --error=logs/affectprune_randomdrop_sft_emor3_gate_%j.err
set -euo pipefail
ROOT=${EMOR3_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export AFFECT_SFT_BASE_MODEL="$ROOT/checkpoints/emor3_base100_frozen/global_step_100/actor/huggingface"
export AFFECT_SFT_TRACE_GLOB="$ROOT/outputs/affectprune_sft_teacher_train2000/teacher_shard*/full.jsonl"
export AFFECT_SFT_EXPECTED_TRACE_FILES=10
export AFFECT_SFT_OUTPUT="$ROOT/outputs/affectprune_emor3start_randomdrop_sft_gate1_20260929"
export AFFECT_SFT_STEPS=1
export AFFECT_SFT_DROP_RATIOS=0.2
export AFFECT_SFT_SAVE_EVERY=1
test -s "$AFFECT_SFT_BASE_MODEL/model.safetensors.index.json"
bash "$ROOT/slurm/affectprune_randomdrop_sft.slurm"

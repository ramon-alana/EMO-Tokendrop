# EMO-R3 + SEPM-inspired Free-Ratio Visual TokenDrop

Research code for learning **which visual tokens to retain** and **how many
tokens to retain per image** during emotion reasoning. The main method starts
from a *local* EMO-R3-style reward-trained checkpoint, uses random genuine
visual-token dropping for SFT cold start, then updates the actor and an
image-conditioned Bernoulli selector in the same RL iteration. The selector
uses an image-level A/B confidence feature and FoE spatial attention. Hard
token choices are optimized by a score-function estimator, not ordinary
backpropagation through indexing. Both the ViT token stream and the LLM image
context are shortened.

This is an **experiment source package**, not a release of a trained model.
The starting checkpoint in the recorded run had received 100 local
original-reward steps. It is **not** the authors' full published EMO-R3
checkpoint. Do not describe comparisons against it as comparisons against
the full official model. Current free-ratio training and its held-out
evaluation must be checked independently; a nonzero gradient is not a
performance result.

## Contents

| Files | Purpose |
| --- | --- |
| `affectprune_randomdrop_sft.py`, `affectprune_sft_validate.py` | Random-drop cold start and TRAIN-only validation at 0/10/20/30/40% |
| `affectprune_joint_grpo_free.py` | Main same-iteration actor + free-ratio selector RL; checkpoint/resume includes both optimizers and RNG |
| `affectprune_joint_eval_free.py` | Full, per-image-budget-matched random/FoE, learned, inverse, and no-FoE views |
| `affectprune_confidence_selector.py` | SEPM-inspired coarse confidence and per-token Bernoulli selector; also the four-bin exploratory selector |
| `affectprune_dynamic_context.py`, `affectprune_lora.py` | Actual visual/LLM compaction and actor LoRA |
| `analyze_emor3_attention.py` | FoE/reasoning attention computations used by the reward and diagnostics |
| `affectprune_joint_grpo.py`, `affectprune_joint_eval.py` | Earlier fixed-four-bin exploratory comparator, **not** the final free-ratio method |
| `analyze_joint_selection.py` | Same-ID, same-token-budget paired result audit |
| `convert_emoset_full.py`, `prepare_affectprune_free_dev.py`, `affectprune_sft_trace_audit.py` | Data conversion, frozen TRAIN-dev split, teacher-trace checks |
| `slurm/` | Portable entry points plus the recorded 2026-09-29 cluster settings and engineering gates |
| `emo-r3/examples/` | Unmodified upstream reward function and format template, with upstream Apache-2.0 license |

Historical two-stage/frozen-selector experiments, manuscript mockups, local
monitoring credentials, and duplicated exploratory scripts are deliberately
excluded. They are not needed to run the present main method. The four-bin
code is retained because it is a necessary ablation.

## What is deliberately **not** in the ZIP

- Model weights, SFT adapters, actor/selector checkpoints, optimizer states.
- EmoSet images/parquet/JSONL, pseudo-label responses, evaluation outputs,
  logs, Hugging Face caches or Python environment copies.
- Credentials, server login helpers, personal chat documents.
- The complete upstream EMO-R3 trainer. Obtain that separately from the
  [authors' repository](https://github.com/SeerRay-Lab/emo-r3) if you need to
  recreate the *starting* checkpoint from Qwen. The two upstream files
  directly used in this continuation are bundled with attribution.

These exclusions keep the archive GitHub-safe and small. Without a licensed
dataset and a checkpoint, the training commands cannot produce a model.

## Environment and data

The recorded server used Python 3.10, PyTorch 2.5.1+cu121, Transformers
4.52.4, NumPy 2.1.3 and the dependencies in `requirements.txt`. Install a
CUDA-compatible PyTorch build for your own system first. One training job
loads both actor and reference Qwen2.5-VL-3B-class models; plan GPU memory
accordingly. Site-specific Slurm modules and partitions in the recorded
wrappers may need editing on other clusters.

Clone this repository to the target GPU host, create `logs/` and `outputs/`,
and set `EMOR3_PROJECT_ROOT` to its absolute path and `EMOR3_PYTHON` to
the environment's Python. For a clean checkout, the Python sources also
default to their own directory when `EMOR3_PROJECT_ROOT` is unset. The
recorded Slurm files were changed **only for path portability**: model
logic, rewards, sampling, and hyperparameters were not rewritten.

Put licensed EmoSet parquet files in the layout described by
`data/README.md`, then run `python convert_emoset_full.py`. The recorded
study used 2,000 TRAIN and 2,000 TEST images with separate IDs. Never
generate pseudo-labels from TEST or choose a checkpoint/decoding rule using
TEST.

## Reproduction order

1. Supply a local EMO-R3-style checkpoint directory as
   `EMOR3_BASE_MODEL`. Keep its origin and training-step count in the run
   log; the recorded checkpoint was a *local 100-step* original-reward
   checkpoint, not official full weights.
2. Generate **TRAIN-only** teacher responses. The clean route is
   `sbatch slurm/generate_teacher_train.slurm`. It uses the full-token
   baseline evaluator and requires no older learned selector. It is a
   re-generation recipe, not a claim of bit-identical teacher traces to
   the historical run. Audit 2,000 unique TRAIN IDs and retain only
   correct, strict-SET, uncapped responses.
3. Run random-drop SFT with `AFFECT_SFT_BASE_MODEL`,
   `AFFECT_SFT_TRACE_GLOB`, `AFFECT_SFT_OUTPUT`,
   `AFFECT_SFT_STEPS=2400`, and
   `AFFECT_SFT_DROP_RATIOS=0,0.1,0.2,0.3,0.4`:
   `python affectprune_randomdrop_sft.py`. The recorded Slurm
   configuration is `slurm/affectprune_randomdrop_sft_emor3_main.slurm`.
4. Use the SFT validation code on TRAIN-only held-out examples at each
   drop level. Freeze the TRAIN-dev IDs and the checkpoint/decoding rule
   *before* TEST. `prepare_affectprune_free_dev.py` reconstructs the
   recorded 64-ID development split from its recorded validation output;
   for a new dataset, choose and save a new TRAIN-only split in advance.
5. For the main method, set `AFFECT_JOINT_BASE_MODEL`,
   `AFFECT_JOINT_SFT_ADAPTER`, `AFFECT_JOINT_DEV_FILE`, and an
   empty `AFFECT_JOINT_OUTPUT`, then submit
   `slurm/train_free_portable.slurm` (or run
   `python affectprune_joint_grpo_free.py`). The recorded first stage was
   400 cumulative RL steps, group size 2, generation limit 384, attention
   reward 0.15, actual-drop reward 0.05, KL coefficient 0.03. The official
   correctness/reflection/format reward is still called. The recorded
   wrapper is `slurm/affectprune_joint_free_emor3_400.slurm`.
6. To continue past 400 steps, set
   `AFFECT_JOINT_RESUME_CHECKPOINT` to **resume_step400.pt** (not
   `joint_step400.pt`), set `AFFECT_JOINT_STEPS` to the *cumulative*
   target, and use a **new empty** output directory. The portable training
   wrapper also accepts these variables. The resume file contains actor
   and selector optimizer states plus RNG state. Do not call a restart from
   only `joint_step400.pt` a seamless continuation.
7. Evaluate one frozen checkpoint and one preselected decode rule. The
   recorded free-ratio diagnostic rule was seeded Bernoulli
   `AFFECT_JOINT_EVAL_BUDGET_DECODE=sample`, chosen on TRAIN before
   looking at free-ratio TEST; its TRAIN gate was less accurate than
   `argmax`, but it matched the stochastic training action. Submit
   `slurm/eval_free_portable.slurm` as 0–19 for the first 1,000 images
   or 0–39 for 2,000 (in batches if the scheduler limits submissions).
   Every four indices cover the same 200 images for full, matched-random,
   matched-FoE and learned methods. Retain all negative results.
8. Audit the raw files and compare exactly matching IDs, labels, actual
   retained-token counts, strict SET, truncation, and cost including the
   confidence/FoE probe. The high-vs-low score inverse control and a
   no-FoE control can be run with
   `AFFECT_JOINT_EVAL_METHOD=learned_inverse` and
   `learned_no_foe`; these are mechanism diagnostics, not a license to
   choose a TEST-favored checkpoint.

For the local starting-point baseline, use
`slurm/eval_baseline_portable.slurm` on the **same TEST IDs**. This is
full-token evaluation of the local checkpoint, not of the authors' complete
released model.

The four-bin training/evaluation scripts are provided solely as an
exploratory ablation. Its 10/20/30/40% action set is **not** the free-ratio
main method. Fixed 0/10/20/30/40/50% sweeps, when used, are post-training
inference controls and must not be described as a learned adaptive ratio.

## Methodological boundaries

- FoE attention is a selector input *and* part of the attention reward.
  FoE self-agreement alone is not independent proof of emotion-rich token
  discovery. Use per-image matched random/FoE controls, inverse-token
  counterfactuals, spatial masks and (when possible) independent region
  annotations.
- A local 100-step starting checkpoint supports **same-starting-point**
  ablations. It cannot substitute for the paper's fully trained EMO-R3
  weights or establish superiority over the authors' published number.
- The full pipeline can be slower even if the second ViT pass is faster.
  Report probe, ViT, generation and total time separately on the same GPU
  conditions.
- The `emo-r3.py` reward file is the upstream *repository*
  implementation. State its exact formula in a paper rather than assuming
  that a differently typeset paper equation is numerically identical.
- Paths serialized into checkpoints are validated at load/resume time.
  For an exact continuation, retain the same model/SFT paths and dataset
  split, or consciously migrate checkpoint metadata and revalidate it.

## Attribution

See `THIRD_PARTY.md`. No simulated manuscript table values or draft
figures are in this repository.

# AffectDrop

**Joint Token Selection and Reflective Reinforcement Learning for Visual Emotion Classification**

## Abstract

Visual emotion classification depends on which cues a model retains and how it interprets them. Reflective training optimizes generated responses, while attention-guided pruning selects visual inputs through a separate rule. We propose AffectDrop to learn these decisions jointly. Random TokenDrop supervised fine-tuning first adapts the actor to incomplete visual inputs. A token-wise Bernoulli selector and the reflective actor are then optimized from the same sampled mask and response outcomes. The selector combines a confidence-modulated Focus-on-Emotion prior with learned spatial logits, allowing retained positions and count to vary with each image. Shared group-relative rewards couple selection to classification and reflection, and physical gathering shortens the input before the main vision Transformer blocks. Across three training seeds, the complete pipeline achieves 58.45 ± 0.22% accuracy on 2,000 EmoSet images, improving over local EMO-R3 continuation by 8.38 ± 0.55 percentage points while removing 28.97 ± 0.25% of visual groups. Values are means and sample standard deviations. With the final actor and per-image retained count fixed, learned masks outperform random and Focus-on-Emotion masks by 4.35 and 3.75 points in the reference run. Frozen-model evaluation further gives average gains of 6.68 points on Emotion6 and 2.53 points on WebEmo. These complementary comparisons support both the complete pipeline and the learned choice of retained positions.

## Environment

Linux, Python 3.10, and a CUDA GPU supporting BF16. The RL stage loads both the actor and frozen reference on one GPU.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

The reward and SET prompt are from [EMO-R3](https://github.com/SeerRay-Lab/emo-r3), under [Apache-2.0](LICENSE-EMO-R3).

## Training

Run from the repository root. Supply a local Qwen2.5-VL-3B-compatible base checkpoint with its processor, training images, and prepared teacher traces. The reported pipeline starts from the local EMO-R3 100-step checkpoint (`base100`), not the unadapted Qwen checkpoint. Model weights and data are not bundled in this code-only package.

Training JSONL records contain `id` (unique, prefixed `train_`), `images` (one absolute image path), `problem` (prompt containing `<image>`), and `answer` (emotion name). Teacher JSONL records contain the matching `id`, `response`, and Boolean `correct`, `strict_set`, and `hit_length_cap` fields. SFT uses correct, strict, uncapped traces. The frozen development file contains 200 TRAIN records; its first 64 unique IDs are excluded from the 2,000-image RL training pool, leaving 1,936 images.

```bash
# Random TokenDrop SFT: 2,400 steps
python train.py sft \
  --model /path/to/base100 \
  --train /path/to/train.jsonl \
  --traces '/path/to/teacher/*.jsonl' \
  --output outputs/sft_seed18 --seed 18

# Joint actor-selector RL: 1,000 steps
python train.py rl \
  --model /path/to/base100 \
  --train /path/to/train.jsonl \
  --dev-file /path/to/train_dev200.jsonl \
  --sft-adapter outputs/sft_seed18/adapter_step2400.pt \
  --output outputs/rl_seed18 --seed 18
```

Use the same base checkpoint in both stages and a new output directory for each run. Repeat both stages with seeds 24 and 36 and their corresponding output/adapter paths. SFT defaults to rank-16 LoRA, learning rate 1e-4, and cosine decay; RL uses group size 2, actor/selector learning rates 2e-5/3e-4, and 384 response tokens. `python train.py sft --help` and `python train.py rl --help` list the options.

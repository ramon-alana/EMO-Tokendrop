# AffectDrop

**Joint Token Selection and Reflective Reinforcement Learning for Visual Emotion Classification**

## Abstract

Abstract—Multimodal large language models (MLLMs) have
achieved impressive progress in visual understanding, but still
face significant challenges in emotional reasoning, where an
emotional prediction depends on which visual evidence the
model relies on. In this work, we focus on a model that does
both: it reasons about the emotion conveyed by an image and
autonomously selects the visual evidence that supports its own
judgment. These two requirements pull in opposite directions:
redundant visual content competes with key emotional cues,
while removing content may discard the subtle expressions or
context needed to separate semantically similar emotions. To
address these, we propose AffectDrop, which jointly optimizes
visual token selection and reflective emotion reasoning through
shared task feedback. Specifically, we first perform random
TokenDrop supervised fine-tuning, removing visual groups so
that the actor can learn to reason from incomplete images. A
confidence-guided spatial selector then combines image features
with a Focus-on-Emotion prior modulated by coarse emotional
confidence, deciding which groups to retain and how many.
Finally, joint reinforcement learning scores each sampled mask
by the response generated on its retained input and updates both
selection and prediction policies with one shared advantage. Ex-
perimental results demonstrate that AffectDrop attains 58.45%
mean accuracy on the EmoSet subset while removing 28.97% of
visual groups, a 16.75% mean relative gain compared with the
baseline. Without additional fine-tuning, the same checkpoints
transfer to Emotion6 and WebEmo with gains of 13.95% and
5.47%. Ablation studies and a user study show that the cues
retained by the proposed method better support the model’s
explanations and predictions. These results indicate that visual
emotion classification benefits from jointly adapting the emotional
reasoning and reflective visual token selection. Our code is
available at https://github.com/ramon-alana/EMO-Tokendrop.

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
  --model /path/t/base100 \
  --train /path/t/train.jsonl \
  --traces '/path/t/teacher/*.jsonl' \
  --output outputs/sft_seed18 --seed 18

# Joint actor-selector RL: 1,000 steps
python train.py rl \
  --model /path/t/base100 \
  --train /path/t/train.jsonl \
  --dev-file /path/t/train_dev200.jsonl 
  --sft-adapter outputs/sft_seed18/adapter_step2400.pt \
  --output outputs/rl_seed18 --seed 18
```

Use the same base checkpoint in both stages and a new output directory for each run. Repeat both stages with seeds 24 and 36 and their corresponding output/adapter paths. SFT defaults to rank-16 LoRA, learning rate 1e-4, and cosine decay; RL uses group size 2, actor/selector learning rates 2e-5/3e-4, and 384 response tokens. `python train.py sft --help` and `python train.py rl --help` list the options.

"""Random-visual-TokenDrop SFT cold start with strict SET targets.

Targets are restricted to correct, strict, uncapped teacher traces generated
from training images; test images are never used for training. Each optimizer
step samples a fresh random visual mask at 0/10/20/30/40% and removes the
corresponding image tokens from both the ViT and LLM context.
"""

from __future__ import annotations

import glob
import json
import math
import os
import random
from collections import Counter
from pathlib import Path

import torch
from jinja2 import Template
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from tokens import append_response, compact_prompt
from lora import adapter_state, install_lora, load_adapter


ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ["AFFECT_SFT_BASE_MODEL"])
TRACE_GLOB = os.environ["AFFECT_SFT_TRACE_GLOB"]
OUT = Path(os.environ["AFFECT_SFT_OUTPUT"])
STEPS = int(os.environ.get("AFFECT_SFT_STEPS", "1"))
GRAD_ACCUM = int(os.environ.get("AFFECT_SFT_GRAD_ACCUM", "1"))
SEED = int(os.environ.get("AFFECT_SFT_SEED", "20260928"))
LR = float(os.environ.get("AFFECT_SFT_LR", "0.0001"))
RANK = int(os.environ.get("AFFECT_SFT_LORA_RANK", "16"))
SAVE_EVERY = int(os.environ.get("AFFECT_SFT_SAVE_EVERY", "50"))
CLASS_BALANCE_ALPHA = float(os.environ.get("AFFECT_SFT_CLASS_BALANCE_ALPHA", "0"))
SCHEDULE = os.environ.get("AFFECT_SFT_SCHEDULE", "constant")
INIT_ADAPTER = os.environ.get("AFFECT_SFT_INIT_ADAPTER", "")
DROP_RATIOS = tuple(float(v) for v in os.environ.get("AFFECT_SFT_DROP_RATIOS", "0,0.1,0.2,0.3,0.4").split(","))
if (STEPS <= 0 or GRAD_ACCUM <= 0 or SAVE_EVERY <= 0 or not DROP_RATIOS
        or any(not 0 <= x < 1 for x in DROP_RATIOS)
        or not 0 <= CLASS_BALANCE_ALPHA <= 1
        or SCHEDULE not in ("constant", "cosine")):
    raise ValueError("Invalid SFT steps, accumulation or drop ratios")
if OUT.exists() and any(OUT.iterdir()):
    raise FileExistsError(f"Refusing to overwrite existing SFT output: {OUT}")
OUT.mkdir(parents=True, exist_ok=True)
torch.manual_seed(SEED)
random.seed(SEED)

train_rows = [json.loads(line) for line in Path(os.environ["AFFECT_TRAIN_FILE"]).read_text(encoding="utf-8").splitlines() if line]
train_by_id = {str(row["id"]): row for row in train_rows}
if len(train_by_id) != len(train_rows):
    raise ValueError("Duplicate training IDs")
trace_paths = sorted(glob.glob(TRACE_GLOB))
if not trace_paths:
    raise FileNotFoundError(f"No strict training traces: {TRACE_GLOB}")
expected_files = int(os.environ.get("AFFECT_SFT_EXPECTED_TRACE_FILES", "0"))
if expected_files and len(trace_paths) != expected_files:
    raise RuntimeError(f"Expected {expected_files} complete train trace files, found {len(trace_paths)}")
targets = {}
seen = set()
for path in trace_paths:
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        trace = json.loads(line)
        sample_id = str(trace["id"])
        if sample_id in seen:
            raise ValueError(f"Duplicate trace: {sample_id}")
        seen.add(sample_id)
        if sample_id not in train_by_id or not sample_id.startswith("train_"):
            raise ValueError(f"Non-training image in SFT trace: {sample_id}")
        if trace.get("correct") and trace.get("strict_set") and not trace.get("hit_length_cap"):
            targets[sample_id] = trace["response"]
if not targets:
    raise RuntimeError("No correct, strict, uncapped training targets")
if expected_files and len(seen) != len(train_rows):
    raise RuntimeError(f"Expected {len(train_rows)} unique training traces, found {len(seen)}")
if STEPS > 1 and len(targets) < 32:
    raise RuntimeError("Full SFT requires at least 32 filtered training targets")

processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda"
)
trainable, replacements = install_lora(model, rank=RANK, alpha=2.0 * RANK)
if INIT_ADAPTER:
    initial = torch.load(INIT_ADAPTER, map_location="cpu", weights_only=True)
    if (initial["base_model"] != str(MODEL_PATH) or int(initial["rank"]) != RANK
            or float(initial["alpha"]) != 2.0 * RANK):
        raise ValueError("Initial adapter is incompatible with this SFT model")
    load_adapter(model, initial["lora"])
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.train()
optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.0)
format_prompt = Template((ROOT / "format_prompt.jinja").read_text(encoding="utf-8"))
ids = sorted(targets)
class_counts = dict(Counter(str(train_by_id[sample_id]["answer"]) for sample_id in ids))
cycle = []
history = OUT / "train.jsonl"


def refill_cycle():
    result = ids.copy()
    if CLASS_BALANCE_ALPHA:
        by_class = {}
        for sample_id in ids:
            label = str(train_by_id[sample_id]["answer"])
            by_class.setdefault(label, []).append(sample_id)
        largest = max(map(len, by_class.values()))
        for members in by_class.values():
            target = round(len(members) ** (1 - CLASS_BALANCE_ALPHA) * largest ** CLASS_BALANCE_ALPHA)
            result.extend(random.choices(members, k=max(0, target - len(members))))
    random.shuffle(result)
    return result


balanced_cycle_size = len(refill_cycle())
print("RANDOMDROP_SFT_START", json.dumps({
    "base_model": str(MODEL_PATH), "trace_files": len(trace_paths),
    "candidate_traces": len(seen), "filtered_targets": len(targets),
    "train_images": len(train_rows), "steps": STEPS, "accum": GRAD_ACCUM,
    "drop_ratios": DROP_RATIOS, "trainable_parameters": sum(p.numel() for p in trainable),
    "lora_modules": len(replacements), "target_class_counts": class_counts,
    "class_balance_alpha": CLASS_BALANCE_ALPHA, "balanced_cycle_size": balanced_cycle_size,
    "schedule": SCHEDULE, "init_adapter": INIT_ADAPTER or None,
}), flush=True)
for step in range(STEPS):
    if SCHEDULE == "cosine":
        warmup = max(1, round(STEPS * 0.03))
        if step < warmup:
            lr_factor = 0.1 + 0.9 * (step + 1) / warmup
        else:
            progress = (step + 1 - warmup) / max(1, STEPS - warmup)
            lr_factor = 0.1 + 0.45 * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = LR * lr_factor
    optimizer.zero_grad(set_to_none=True)
    measurements = []
    for _ in range(GRAD_ACCUM):
        if not cycle:
            cycle = refill_cycle()
        sample_id = cycle.pop()
        row = train_by_id[sample_id]
        image = Image.open(row["images"][0]).convert("RGB")
        factor = math.sqrt(262144 / (image.width * image.height))
        image = image.resize((int(image.width * factor), int(image.height * factor)))
        prompt = format_prompt.render(content=row["problem"]).strip()
        content = []
        for part_index, part in enumerate(prompt.split("<image>")):
            if part_index:
                content.append({"type": "image", "image": image})
            if part:
                content.append({"type": "text", "text": part})
        rendered = processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[rendered], images=[image], padding=True, return_tensors="pt")
        inputs = {key: value.to(model.device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}
        groups = int((inputs["input_ids"] == model.config.image_token_id).sum())
        drop_target = random.choice(DROP_RATIOS)
        drop_count = min(groups - 1, round(groups * drop_target))
        keep = torch.ones(groups, dtype=torch.bool, device=model.device)
        if drop_count:
            drop_indices = torch.randperm(groups, device=model.device)[:drop_count]
            keep[drop_indices] = False
        with torch.no_grad():
            compact = compact_prompt(model, inputs, keep)
        response = targets[sample_id] + processor.tokenizer.eos_token
        response_ids = processor.tokenizer(response, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
        batch = append_response(model, compact, response_ids)
        output = model(
            inputs_embeds=batch["inputs_embeds"],
            attention_mask=batch["attention_mask"],
            position_ids=batch["position_ids"],
            labels=batch["labels"],
            use_cache=False,
        )
        if not torch.isfinite(output.loss):
            raise FloatingPointError(f"Nonfinite loss at step {step + 1}")
        (output.loss / GRAD_ACCUM).backward()
        measurements.append({
            "id": sample_id,
            "loss": float(output.loss.detach()),
            "drop_ratio": 1.0 - compact["visual_tokens_after"] / compact["visual_tokens_before"],
            "visual_tokens_before": compact["visual_tokens_before"],
            "visual_tokens_after": compact["visual_tokens_after"],
            "llm_prompt_tokens_after": int(compact["input_ids"].shape[1]),
        })
        del output, batch, inputs, compact
    grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
    if not math.isfinite(grad_norm):
        raise FloatingPointError("Nonfinite LoRA gradient")
    optimizer.step()
    item = {"step": step + 1, "gradient_norm": grad_norm, "lr": optimizer.param_groups[0]["lr"],
            "samples": measurements}
    with history.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    print("RANDOMDROP_SFT_STEP", json.dumps({
        "step": step + 1,
        "loss": round(sum(x["loss"] for x in measurements) / len(measurements), 5),
        "drop": [round(x["drop_ratio"], 3) for x in measurements],
        "gradient_norm": round(grad_norm, 5),
    }), flush=True)
    if (step + 1) % SAVE_EVERY == 0 or step + 1 == STEPS:
        torch.save({
            "lora": adapter_state(model), "step": step + 1,
            "base_model": str(MODEL_PATH), "rank": RANK,
            "alpha": 2.0 * RANK, "drop_ratios": DROP_RATIOS,
            "filtered_targets": len(targets), "seed": SEED,
            "init_adapter": INIT_ADAPTER or None,
        }, OUT / f"adapter_step{step + 1}.pt")
print("RANDOMDROP_SFT_COMPLETED", STEPS, flush=True)

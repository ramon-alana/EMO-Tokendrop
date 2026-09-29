"""Train-internal validation of random-drop SFT under real dynamic context.

The validation IDs are TRAIN examples that were never SFT targets. This file
does not read the test split. A common deterministic mask is used at each
ratio across checkpoints, so checkpoint/prefix comparisons are paired.
"""

from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import random
import re
from collections import defaultdict
from pathlib import Path

import torch
from jinja2 import Template
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from affectprune_dynamic_context import compact_prompt, generate_compacted
from affectprune_lora import install_lora, load_adapter


ROOT = Path(os.environ.get("EMOR3_PROJECT_ROOT", Path(__file__).resolve().parent))
MODEL_PATH = Path(os.environ.get("AFFECT_VAL_MODEL_PATH", ROOT / "models/Qwen2.5-VL-3B-Instruct"))
ADAPTER_PATH = Path(os.environ["AFFECT_VAL_SFT_ADAPTER"])
BASELINE_MODEL_PATH = os.environ.get("AFFECT_VAL_BASELINE_MODEL", "")
RESULTS = Path(os.environ["AFFECT_VAL_OUTPUT"])
RATIO = float(os.environ.get("AFFECT_VAL_DROP_RATIO", "0.2"))
PREFILL = os.environ.get("AFFECT_VAL_PREFILL_STEP1", "0") == "1"
PER_CLASS = int(os.environ.get("AFFECT_VAL_PER_CLASS", "8"))
MAX_NEW_TOKENS = int(os.environ.get("AFFECT_VAL_MAX_NEW_TOKENS", "384"))
TRACE_GLOB = os.environ.get(
    "AFFECT_VAL_TRACE_GLOB",
    str(ROOT / "outputs/affectprune_sft_teacher_train2000/teacher_shard*/full.jsonl"),
)
if not 0 <= RATIO < 1 or PER_CLASS < 1 or MAX_NEW_TOKENS < 8:
    raise ValueError("Invalid SFT validation ratio, class count, or answer length")
if not ADAPTER_PATH.is_file():
    raise FileNotFoundError(ADAPTER_PATH)
trace_paths = sorted(glob.glob(TRACE_GLOB))
if len(trace_paths) != 10:
    raise RuntimeError(f"Expected ten TRAIN trace shards, found {len(trace_paths)}")
train_rows = [json.loads(line) for line in (ROOT / "data/EmoSet2k_full/train.jsonl").read_text(encoding="utf-8").splitlines() if line]
train_by_id = {str(row["id"]): row for row in train_rows}
if len(train_rows) != 2000 or len(train_by_id) != len(train_rows):
    raise RuntimeError("Expected 2000 unique TRAIN images")
seen = set()
sft_target_ids = set()
for path in trace_paths:
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        trace = json.loads(line)
        sample_id = str(trace["id"])
        if sample_id in seen or sample_id not in train_by_id:
            raise RuntimeError(f"Bad or repeated TRAIN trace: {sample_id}")
        seen.add(sample_id)
        if trace.get("correct") and trace.get("strict_set") and not trace.get("hit_length_cap"):
            sft_target_ids.add(sample_id)
if len(seen) != 2000:
    raise RuntimeError("TRAIN trace audit did not cover all 2000 images")
by_class = defaultdict(list)
for sample_id, row in train_by_id.items():
    if sample_id not in sft_target_ids:
        by_class[str(row["answer"])].append(sample_id)
chosen_ids = []
for label, ids in sorted(by_class.items()):
    rng = random.Random(20260928 + sum((index + 1) * ord(char) for index, char in enumerate(label)))
    rng.shuffle(ids)
    if len(ids) < PER_CLASS:
        raise RuntimeError(f"Too few untrained validation images in class {label}")
    chosen_ids.extend(ids[:PER_CLASS])
chosen_ids.sort()
if len(chosen_ids) != 8 * PER_CLASS:
    raise RuntimeError("Expected eight emotion classes in TRAIN holdout")
chosen_hash = hashlib.sha256("\n".join(chosen_ids).encode()).hexdigest()[:16]

saved = None
if not BASELINE_MODEL_PATH:
    saved = torch.load(ADAPTER_PATH, map_location="cpu", weights_only=True)
    if saved["base_model"] != str(MODEL_PATH) or int(saved["filtered_targets"]) != len(sft_target_ids):
        raise RuntimeError("SFT checkpoint model or target pool differs from validation")
model_path = Path(BASELINE_MODEL_PATH) if BASELINE_MODEL_PATH else MODEL_PATH
checkpoint_name = f"baseline:{model_path.name}" if BASELINE_MODEL_PATH else f"sft:{saved['step']}"
processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)
model = AutoModelForVision2Seq.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda"
)
if saved is not None:
    install_lora(model, rank=int(saved["rank"]), alpha=float(saved["alpha"]), dropout=0.0)
    load_adapter(model, saved["lora"])
model.eval()
template = Template((ROOT / "emo-r3/examples/format_prompt/emor3.jinja").read_text(encoding="utf-8"))
strict_pattern = re.compile(r"<step1>.*</step1>.*<step2>.*</step2>.*<step3>.*</step3>.*\\boxed\{.*\}.*", re.DOTALL)
RESULTS.parent.mkdir(parents=True, exist_ok=True)
done = {}
if RESULTS.exists():
    for line in RESULTS.read_text(encoding="utf-8").splitlines():
        if line:
            item = json.loads(line)
            if item["id"] in done or item["id"] not in chosen_ids:
                raise RuntimeError("Unexpected or repeated validation output")
            recorded_checkpoint = item.get("checkpoint_name", f"sft:{item.get('adapter_step')}")
            if item["ratio_target"] != RATIO or item["prefill_step1"] != PREFILL or recorded_checkpoint != checkpoint_name:
                raise RuntimeError("Cannot resume output from a different validation condition")
            done[item["id"]] = item
print("SFT_VAL_START", json.dumps({
    "checkpoint_name": checkpoint_name, "ratio": RATIO, "prefill": PREFILL,
    "validation_images": len(chosen_ids), "validation_hash": chosen_hash,
    "sft_target_images": len(sft_target_ids), "completed_rows": len(done),
}), flush=True)
for index, sample_id in enumerate(chosen_ids):
    if sample_id in done:
        continue
    row = train_by_id[sample_id]
    with Image.open(row["images"][0]) as source:
        image = source.convert("RGB")
    factor = math.sqrt(262144 / (image.width * image.height))
    image = image.resize((int(image.width * factor), int(image.height * factor)))
    prompt = template.render(content=row["problem"]).strip()
    content = []
    for part_index, part in enumerate(prompt.split("<image>")):
        if part_index:
            content.append({"type": "image", "image": image})
        if part:
            content.append({"type": "text", "text": part})
    rendered = processor.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
    )
    if PREFILL:
        rendered += "<step1>"
    inputs = processor(text=[rendered], images=[image], padding=True, return_tensors="pt")
    inputs = {key: value.to(model.device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}
    groups = int((inputs["input_ids"] == model.config.image_token_id).sum())
    drop_count = min(groups - 1, round(groups * RATIO))
    keep = torch.ones(groups, dtype=torch.bool, device=model.device)
    if drop_count:
        seed = int(hashlib.sha256(f"{sample_id}:{RATIO}".encode()).hexdigest()[:16], 16)
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(groups, generator=generator)[:drop_count].to(model.device)
        keep[indices] = False
    with torch.inference_mode():
        compact = compact_prompt(model, inputs, keep)
        generated = generate_compacted(model, compact, MAX_NEW_TOKENS, do_sample=False)
    response = ("<step1>" if PREFILL else "") + processor.batch_decode(generated, skip_special_tokens=True)[0]
    normalized = re.sub(r"\s*(<|>|/)\s*", r"\1", response)
    labels = re.findall(r"\\boxed\{([^{}]+)\}", response)
    prediction = labels[-1].strip().casefold() if labels else None
    item = {
        "id": sample_id, "answer": row["answer"], "prediction": prediction,
        "correct": prediction == str(row["answer"]).casefold(),
        "strict_set": bool(strict_pattern.fullmatch(normalized)),
        "hit_length_cap": int(generated.shape[1]) >= MAX_NEW_TOKENS,
        "ratio_target": RATIO, "ratio_actual": 1 - compact["visual_tokens_after"] / compact["visual_tokens_before"],
        "visual_tokens_before": compact["visual_tokens_before"],
        "visual_tokens_after": compact["visual_tokens_after"],
        "llm_prompt_tokens_after": int(compact["input_ids"].shape[1]),
        "prefill_step1": PREFILL, "checkpoint_name": checkpoint_name,
        "validation_hash": chosen_hash, "response": response,
    }
    with RESULTS.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    done[sample_id] = item
    print("SFT_VAL_IMAGE", json.dumps({
        "index": index + 1, "total": len(chosen_ids), "id": sample_id,
        "correct": item["correct"], "strict_set": item["strict_set"],
        "actual_drop": round(item["ratio_actual"], 4),
    }), flush=True)
rows = [done[sample_id] for sample_id in chosen_ids]
print("SFT_VAL_COMPLETED", json.dumps({
    "count": len(rows), "correct": sum(x["correct"] for x in rows),
    "strict_set": sum(x["strict_set"] for x in rows),
    "hit_length_cap": sum(x["hit_length_cap"] for x in rows),
    "actual_drop_mean": sum(x["ratio_actual"] for x in rows) / len(rows),
    "validation_hash": chosen_hash,
}), flush=True)

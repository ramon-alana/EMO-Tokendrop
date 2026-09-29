"""Paired evaluation for same-step EMO-R3 actor/selector RL checkpoints.

No test image is used for checkpoint selection or training. All TokenDrop
methods use the same learned per-image budget; only retained positions differ.
Probe, visual-encoding, generation, and whole-pipeline timings are separate.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from jinja2 import Template
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from affectprune_confidence_selector import (
    ConfidenceSelector,
    FreeRatioSelector,
    image_features,
    sepm_coarse_confidence,
)
from affectprune_dynamic_context import compact_prompt, generate_compacted
from affectprune_lora import install_lora, load_adapter
from analyze_emor3_attention import FOE_TEXT, attention_view


ROOT = Path(os.environ.get("EMOR3_PROJECT_ROOT", Path(__file__).resolve().parent))
MODEL_PATH = Path(os.environ.get("AFFECT_JOINT_EVAL_BASE_MODEL", ROOT / "models/Qwen2.5-VL-3B-Instruct"))
DATA_PATH = Path(os.environ.get(
    "AFFECT_JOINT_EVAL_TEST_FILE", str(ROOT / "data/EmoSet2k_full/test.jsonl")
))
BASELINE_ONLY = os.environ.get("AFFECT_JOINT_EVAL_BASELINE_ONLY", "0") == "1"
CHECKPOINT = (
    MODEL_PATH if BASELINE_ONLY else Path(os.environ["AFFECT_JOINT_EVAL_CHECKPOINT"])
)
RESULTS = Path(os.environ["AFFECT_JOINT_EVAL_OUTPUT"])
METHOD = os.environ.get("AFFECT_JOINT_EVAL_METHOD", "learned")
DECODE = os.environ.get("AFFECT_JOINT_EVAL_BUDGET_DECODE", "argmax")
START = int(os.environ.get("AFFECT_JOINT_EVAL_START_SAMPLE", "0"))
COUNT = int(os.environ.get("AFFECT_JOINT_EVAL_MAX_SAMPLES", "200"))
MAX_TOKENS = int(os.environ.get("AFFECT_JOINT_EVAL_MAX_NEW_TOKENS", "384"))
PREFILL = os.environ.get("AFFECT_JOINT_EVAL_PREFILL_STEP1", "0") == "1"
SAVE_SPATIAL = os.environ.get("AFFECT_JOINT_EVAL_SAVE_SPATIAL", "0") == "1"
if METHOD not in {"full", "random_matched", "foe_matched", "learned", "learned_inverse", "learned_no_foe"}:
    raise ValueError("Unknown evaluation method")
if DECODE not in {"argmax", "sample"} or START < 0 or COUNT < 1 or MAX_TOKENS < 8:
    raise ValueError("Invalid evaluation protocol")
if BASELINE_ONLY and METHOD != "full":
    raise ValueError("Unadapted EMO-R3 baseline is only defined for the full-token view")
if not CHECKPOINT.is_file():
    if not (BASELINE_ONLY and CHECKPOINT.is_dir()):
        raise FileNotFoundError(CHECKPOINT)

if BASELINE_ONLY:
    state = {"step": 0, "selector_policy": "none"}
else:
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    state.setdefault("selector_policy", "four_bins")
    if state["selector_policy"] not in {"four_bins", "free_bernoulli"}:
        raise ValueError("Unknown selector policy in checkpoint")
    sft_path = Path(state["sft_adapter"])
    sft = torch.load(sft_path, map_location="cpu", weights_only=True)
    if sft["base_model"] != str(MODEL_PATH):
        raise ValueError("Joint checkpoint uses a different base model")
    if state["step"] < 1:
        raise ValueError("Incomplete joint checkpoint")
processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation="eager", device_map="cuda"
)
if not BASELINE_ONLY:
    install_lora(model, rank=int(sft["rank"]), alpha=float(sft["alpha"]), dropout=0.0)
    load_adapter(model, state["actor_lora"])
model.eval()
if not BASELINE_ONLY:
    selector = (
        FreeRatioSelector() if state["selector_policy"] == "free_bernoulli"
        else ConfidenceSelector()
    ).to(model.device)
    selector.load_state_dict(state["selector"], strict=True)
    selector.eval()
    if state["selector_policy"] == "four_bins" and tuple(selector.budgets) != tuple(state["budgets"]):
        raise ValueError("Selector budget schema differs from checkpoint")
template = Template((ROOT / "emo-r3/examples/format_prompt/emor3.jinja").read_text(encoding="utf-8"))
strict_pattern = re.compile(r"<step1>.*</step1>.*<step2>.*</step2>.*<step3>.*</step3>.*\\boxed\{.*\}.*", re.DOTALL)
all_rows = [json.loads(line) for line in DATA_PATH.read_text(encoding="utf-8").splitlines() if line]
if len(all_rows) not in {200, 2000} or len({str(row["id"]) for row in all_rows}) != len(all_rows):
    raise ValueError("Expected 200 or 2000 distinct TEST image IDs")
rows = all_rows[START:START + COUNT]
if len(rows) != COUNT:
    raise ValueError("Requested test shard is incomplete")
allowed_ids = {str(row["id"]) for row in rows}
RESULTS.parent.mkdir(parents=True, exist_ok=True)
done = {}
if RESULTS.is_file():
    for line in RESULTS.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        item = json.loads(line)
        if item["id"] in done or item["id"] not in allowed_ids:
            raise ValueError("Duplicate or out-of-shard evaluation ID")
        if (item["method"] != METHOD or item["budget_decode"] != DECODE
                or item.get("test_file", str(ROOT / "data/EmoSet2k_full/test.jsonl")) != str(DATA_PATH)
                or item["joint_step"] != state["step"] or item["prefill_step1"] != PREFILL
                or item.get("selector_policy", "four_bins") != state["selector_policy"]
                or bool(item.get("save_spatial_trace", False)) != SAVE_SPATIAL):
            raise ValueError("Cannot resume a different evaluation protocol")
        done[item["id"]] = item


def sync():
    torch.cuda.synchronize(model.device)


def model_inputs(image, prompt):
    content = []
    for index, part in enumerate(prompt.split("<image>")):
        if index:
            content.append({"type": "image", "image": image})
        if part:
            content.append({"type": "text", "text": part})
    rendered = processor.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
    )
    if PREFILL:
        rendered += "<step1>"
    result = processor(text=[rendered], images=[image], padding=True, return_tensors="pt")
    return {key: value.to(model.device) if isinstance(value, torch.Tensor) else value for key, value in result.items()}


print("JOINT_EVAL_START", json.dumps({
    "checkpoint": str(CHECKPOINT), "step": state["step"], "method": METHOD,
    "selector_policy": state["selector_policy"],
    "test_file": str(DATA_PATH),
    "budget_decode": DECODE, "start": START, "count": COUNT,
    "prefill_step1": PREFILL, "baseline_only": BASELINE_ONLY, "already_done": len(done),
}), flush=True)
for index, row in enumerate(rows):
    sample_id = str(row["id"])
    if sample_id in done:
        continue
    sync()
    whole_start = time.perf_counter()
    with Image.open(row["images"][0]) as source:
        image = source.convert("RGB")
    factor = math.sqrt(262144 / (image.width * image.height))
    image = image.resize((int(image.width * factor), int(image.height * factor)))
    prompt = template.render(content=row["problem"]).strip()
    inputs = model_inputs(image, prompt)
    total_groups = int((inputs["input_ids"] == model.config.image_token_id).sum())
    keep = torch.ones(total_groups, dtype=torch.bool, device=model.device)
    budget_probs = None
    entropy = None
    confidence = None
    spatial_trace = None
    chosen_budget = 0.0
    probe_seconds = 0.0
    if METHOD != "full":
        sync()
        probe_start = time.perf_counter()
        with torch.inference_mode():
            foe, _, _ = attention_view(
                model, processor, image,
                f"{FOE_TEXT}\n{row['problem'].replace('<image>', '').strip()}",
                [FOE_TEXT], last_layers=1,
            )
            confidence = sepm_coarse_confidence(model, processor, image)
            visual = model.visual
            features = image_features(
                inputs["pixel_values"], inputs["image_grid_thw"],
                visual.spatial_merge_unit, visual.spatial_merge_size,
            )
            if len(foe) != total_groups:
                raise ValueError("FoE map and visual groups differ")
            spatial_logits, budget_logits = selector(features, foe, confidence["confidence"])
            if METHOD == "learned_no_foe":
                spatial_logits, _ = selector(
                    features, np.full(total_groups, 1.0 / total_groups), confidence["confidence"]
                )
            seed = int(hashlib.sha256(f"joint:{sample_id}:{state['seed']}".encode()).hexdigest()[:16], 16)
            generator = torch.Generator(device=model.device).manual_seed(seed)
            if state["selector_policy"] == "free_bernoulli":
                keep_probs = torch.sigmoid(spatial_logits.float())
                entropy = float(torch.distributions.Bernoulli(probs=keep_probs).entropy().mean())
                if DECODE == "argmax":
                    learned_keep = keep_probs >= 0.5
                else:
                    learned_keep = torch.rand(keep_probs.shape, generator=generator, device=model.device) < keep_probs
                if not learned_keep.any():
                    learned_keep[torch.argmax(keep_probs)] = True
                keep_count = int(learned_keep.sum())
                chosen_budget = 1.0 - keep_count / total_groups
            else:
                probabilities = torch.softmax(budget_logits.float(), dim=0)
                budget_probs = probabilities.detach().cpu().tolist()
                entropy = float(-(probabilities * probabilities.clamp_min(1e-12).log()).sum())
                if DECODE == "argmax":
                    budget_index = int(torch.argmax(budget_logits))
                else:
                    budget_index = int(torch.multinomial(probabilities, 1, generator=generator))
                chosen_budget = selector.budgets[budget_index]
                keep_count = max(1, min(total_groups - 1, round(total_groups * (1 - chosen_budget))))
            if METHOD in {"learned", "learned_inverse", "learned_no_foe"}:
                if state["selector_policy"] == "free_bernoulli" and METHOD == "learned":
                    keep = learned_keep.clone()
                elif DECODE == "argmax" or state["selector_policy"] == "free_bernoulli":
                    order = torch.argsort(spatial_logits, descending=(METHOD != "learned_inverse"))
                    keep.zero_()
                    keep[order[:keep_count]] = True
                else:
                    uniform = torch.rand(spatial_logits.shape, generator=generator, device=model.device).clamp(1e-6, 1 - 1e-6)
                    gumbel = -torch.log(-torch.log(uniform))
                    signed_logits = spatial_logits if METHOD != "learned_inverse" else -spatial_logits
                    order = torch.argsort(signed_logits + gumbel, descending=True)
                    keep.zero_()
                    keep[order[:keep_count]] = True
            elif METHOD == "foe_matched":
                keep.zero_()
                kept = np.argsort(-np.asarray(foe), kind="stable")[:keep_count].copy()
                keep[torch.as_tensor(kept, device=model.device)] = True
            else:
                rng = np.random.default_rng(seed + 1)
                kept = rng.choice(total_groups, size=keep_count, replace=False)
                keep.zero_()
                keep[torch.as_tensor(kept, device=model.device)] = True
            if SAVE_SPATIAL and METHOD in {"learned", "learned_inverse", "learned_no_foe"}:
                spatial_trace = {
                    "grid_height": int(features.shape[-2]),
                    "grid_width": int(features.shape[-1]),
                    "spatial_logits": [round(float(x), 6) for x in spatial_logits.detach().cpu().tolist()],
                    "foe_attention": [round(float(x), 8) for x in foe],
                    "kept_indices": torch.nonzero(keep).flatten().cpu().tolist(),
                }
        sync()
        probe_seconds = time.perf_counter() - probe_start
    sync()
    vision_start = time.perf_counter()
    with torch.inference_mode():
        compact = compact_prompt(model, inputs, keep)
    sync()
    vision_seconds = time.perf_counter() - vision_start
    sync()
    generation_start = time.perf_counter()
    with torch.inference_mode():
        generated = generate_compacted(model, compact, MAX_TOKENS, do_sample=False)
    sync()
    generation_seconds = time.perf_counter() - generation_start
    response = ("<step1>" if PREFILL else "") + processor.batch_decode(generated, skip_special_tokens=True)[0]
    normalized = re.sub(r"\s*(<|>|/)\s*", r"\1", response)
    labels = re.findall(r"\\boxed\{([^{}]+)\}", response)
    prediction = labels[-1].strip().casefold() if labels else None
    sync()
    item = {
        "id": sample_id, "answer": row["answer"], "prediction": prediction,
        "correct": prediction == str(row["answer"]).casefold(),
        "strict_set": bool(strict_pattern.fullmatch(normalized)),
        "hit_length_cap": int(generated.shape[1]) >= MAX_TOKENS,
        "method": METHOD, "budget_decode": DECODE,
        "budget_target": chosen_budget if state["selector_policy"] == "four_bins" else None,
        "policy_realized_drop_ratio": chosen_budget if state["selector_policy"] == "free_bernoulli" else None,
        "selector_policy": state["selector_policy"],
        "budget_probabilities": budget_probs, "budget_entropy": entropy,
        "sepm_confidence": confidence, "visual_tokens_before": compact["visual_tokens_before"],
        "visual_tokens_after": compact["visual_tokens_after"],
        "ratio_actual": 1 - compact["visual_tokens_after"] / compact["visual_tokens_before"],
        "llm_prompt_tokens_after": int(compact["input_ids"].shape[1]),
        "probe_seconds": probe_seconds, "vision_seconds": vision_seconds,
        "generation_seconds": generation_seconds, "pipeline_seconds": time.perf_counter() - whole_start,
        "checkpoint": str(CHECKPOINT), "joint_step": state["step"], "prefill_step1": PREFILL,
        "save_spatial_trace": SAVE_SPATIAL,
        "test_file": str(DATA_PATH),
        "response": response,
    }
    if spatial_trace is not None:
        item["spatial_trace"] = spatial_trace
    with RESULTS.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    done[sample_id] = item
    print("JOINT_EVAL_IMAGE", json.dumps({
        "index": index + 1, "id": sample_id, "method": METHOD,
        "correct": item["correct"], "strict_set": item["strict_set"],
        "actual_drop": round(item["ratio_actual"], 4),
        "pipeline_seconds": round(item["pipeline_seconds"], 4),
    }), flush=True)
print("JOINT_EVAL_COMPLETED", json.dumps({
    "count": len(done), "correct": sum(item["correct"] for item in done.values()),
    "strict_set": sum(item["strict_set"] for item in done.values()),
    "truncated": sum(item["hit_length_cap"] for item in done.values()),
}), flush=True)

"""Joint actor/selector group-relative RL after random-TokenDrop SFT."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
from jinja2 import Template
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from selector import (
    FreeRatioSelector, image_features, sepm_coarse_confidence,
)
from tokens import append_response, compact_prompt, generate_compacted
from lora import adapter_state, install_lora, load_adapter
from attention import FOE_TEXT, attention_match_score, attention_view, retained_attention_mass


ROOT = Path(__file__).resolve().parent
SFT_ADAPTER = Path(os.environ["AFFECT_JOINT_SFT_ADAPTER"])
OUT = Path(os.environ["AFFECT_JOINT_OUTPUT"])
STEPS = int(os.environ.get("AFFECT_JOINT_STEPS", "1"))
GROUP_SIZE = int(os.environ.get("AFFECT_JOINT_GROUP_SIZE", "2"))
MAX_TOKENS = int(os.environ.get("AFFECT_JOINT_MAX_TOKENS", "256"))
REFLECT_TOKENS = int(os.environ.get("AFFECT_JOINT_REFLECT_TOKENS", "32"))
SEED = int(os.environ.get("AFFECT_JOINT_SEED", "20260928"))
SAVE_EVERY = int(os.environ.get("AFFECT_JOINT_SAVE_EVERY", "20"))
FULL_SAVE_EVERY = int(os.environ.get("AFFECT_JOINT_FULL_SAVE_EVERY", "200"))
RESUME = Path(os.environ["AFFECT_JOINT_RESUME_CHECKPOINT"]) if os.environ.get("AFFECT_JOINT_RESUME_CHECKPOINT") else None
ALIGN_WEIGHT = float(os.environ.get("AFFECT_JOINT_ALIGN_WEIGHT", "0.15"))
DROP_WEIGHT = float(os.environ.get("AFFECT_JOINT_DROP_WEIGHT", "0.05"))
KL_COEF = float(os.environ.get("AFFECT_JOINT_KL_COEF", "0.03"))
SELECTOR_POLICY = "free_bernoulli"
if STEPS < 1 or GROUP_SIZE < 2 or MAX_TOKENS < 8:
    raise ValueError("Invalid joint training dimensions")
if SAVE_EVERY < 1 or FULL_SAVE_EVERY < 1 or (RESUME is not None and not RESUME.is_file()):
    raise ValueError("Invalid checkpoint schedule or missing resumable checkpoint")
if any(not math.isfinite(x) or x < 0 for x in (ALIGN_WEIGHT, DROP_WEIGHT, KL_COEF)):
    raise ValueError("Invalid reward or KL coefficient")
if OUT.exists() and any(OUT.iterdir()):
    raise FileExistsError(f"Refusing to overwrite joint RL output: {OUT}")
OUT.mkdir(parents=True, exist_ok=True)
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

import reward as original_reward
model_path = Path(os.environ["AFFECT_JOINT_BASE_MODEL"])
processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
adapter = torch.load(SFT_ADAPTER, map_location="cpu", weights_only=True)
if adapter["base_model"] != str(model_path):
    raise ValueError("SFT adapter is for a different base model")
model = AutoModelForVision2Seq.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, attn_implementation="eager", device_map="cuda"
)
trainable, layers = install_lora(
    model, rank=int(adapter["rank"]), alpha=float(adapter["alpha"]), dropout=0.0
)
load_adapter(model, adapter["lora"])
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
reference = AutoModelForVision2Seq.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda"
).eval()
install_lora(reference, rank=int(adapter["rank"]), alpha=float(adapter["alpha"]), dropout=0.0)
load_adapter(reference, adapter["lora"])
for parameter in reference.parameters():
    parameter.requires_grad_(False)
selector = FreeRatioSelector().to(model.device)
actor_optimizer = torch.optim.AdamW(trainable, lr=float(os.environ.get("AFFECT_JOINT_ACTOR_LR", "2e-5")))
selector_optimizer = torch.optim.AdamW(selector.parameters(), lr=float(os.environ.get("AFFECT_JOINT_SELECTOR_LR", "3e-4")))
format_prompt = Template((ROOT / "format_prompt.jinja").read_text(encoding="utf-8"))
rows = [json.loads(line) for line in Path(os.environ["AFFECT_TRAIN_FILE"]).read_text(encoding="utf-8").splitlines() if line]
if not rows or len({str(x["id"]) for x in rows}) != len(rows):
    raise ValueError("Missing or duplicated train rows")
dev_file = os.environ.get("AFFECT_JOINT_DEV_FILE")
dev_hash = None
if dev_file:
    dev_rows = [json.loads(line) for line in Path(dev_file).read_text(encoding="utf-8").splitlines() if line]
    dev_ids = [str(item["id"]) for item in dev_rows[:64]]
    if len(dev_rows) != 200 or len(dev_ids) != 64 or len(set(dev_ids)) != 64:
        raise ValueError("Require the frozen 200-row TRAIN-internal development file")
    if not set(dev_ids) <= {str(item["id"]) for item in rows}:
        raise ValueError("Development image absent from TRAIN")
    rows = [item for item in rows if str(item["id"]) not in set(dev_ids)]
    if len(rows) != 1936:
        raise ValueError("TRAIN-development split was not disjoint")
    dev_hash = hashlib.sha256("\n".join(dev_ids).encode()).hexdigest()[:16]
order = np.random.default_rng(SEED).permutation(len(rows))
start_step = 0
if RESUME is not None:
    prior = torch.load(RESUME, map_location="cpu", weights_only=True)
    start_step = int(prior["step"])
    if (prior.get("selector_policy") != SELECTOR_POLICY
            or prior.get("sft_adapter") != str(SFT_ADAPTER)
            or prior.get("base_model") != str(model_path)
            or prior.get("seed") != SEED
            or prior.get("train_dev_hash") != dev_hash
            or not 0 < start_step < STEPS):
        raise ValueError("Resumed training state or development split differs")
    load_adapter(model, prior["actor_lora"])
    selector.load_state_dict(prior["selector"], strict=True)
    actor_optimizer.load_state_dict(prior["actor_optimizer"])
    selector_optimizer.load_state_dict(prior["selector_optimizer"])
    torch.set_rng_state(prior["torch_rng_state"])
    torch.cuda.set_rng_state_all(prior["cuda_rng_state_all"])


def model_inputs(image, text):
    content = []
    for part_index, part in enumerate(text.split("<image>")):
        if part_index:
            content.append({"type": "image", "image": image})
        if part:
            content.append({"type": "text", "text": part})
    rendered = processor.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
    )
    result = processor(
        text=[rendered], images=[image] if "<image>" in text else None,
        padding=True, return_tensors="pt"
    )
    return {key: value.to(model.device) if isinstance(value, torch.Tensor) else value for key, value in result.items()}


@torch.inference_mode()
def ordinary_generate(image, text, tokens):
    inputs = model_inputs(image, text)
    answer = model.generate(**inputs, do_sample=False, max_new_tokens=tokens)
    generated = answer[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def reflection(response, image):
    first_two = re.search(r"(<step1>.*?</step2>)", response, re.DOTALL)
    reasoning = first_two.group(1).strip() if first_two else ""
    text_query = (
        reasoning + "\nWhich emotion best describes the text above?\n"
        "Answer with one category only: amusement, anger, awe, contentment, "
        "disgust, excitement, fear, sadness."
    ) if reasoning else "Answer 'None'"
    first_step = re.search(r"(<step1>.*?)(?=<step2>)", response, re.DOTALL)
    visual_reason = first_step.group(1).strip() if first_step else ""
    image_query = (
        "<image>\nCan the following text describe the image? "
        "Answer yes or no. " + visual_reason
    ) if visual_reason else "<image>\nAnswer 'None'"
    return ordinary_generate(None, text_query, REFLECT_TOKENS), ordinary_generate(image, image_query, REFLECT_TOKENS)


def sampled_token_log_probs(network, batch):
    output = network(
        inputs_embeds=batch["inputs_embeds"],
        attention_mask=batch["attention_mask"],
        position_ids=batch["position_ids"],
        use_cache=False,
    )
    shifted = batch["labels"][:, 1:]
    valid = shifted != -100
    selected = torch.log_softmax(output.logits[:, :-1, :].float(), dim=-1)
    gathered = selected.gather(-1, shifted.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return gathered[valid]


@torch.inference_mode()
def matching_score(compact, response_ids, foe, keep):
    batch = append_response(model, compact, response_ids)
    output = model(
        inputs_embeds=batch["inputs_embeds"],
        attention_mask=batch["attention_mask"],
        position_ids=batch["position_ids"],
        output_attentions=True,
        use_cache=False,
    )
    if not output.attentions or output.attentions[-1] is None:
        raise RuntimeError("Joint attention reward requires eager attention")
    image_positions = torch.where(compact["input_ids"][0] == model.config.image_token_id)[0]
    query_start = compact["input_ids"].shape[1]
    last = output.attentions[-1][0].float().mean(dim=0)
    reasoning = last[query_start:, image_positions].mean(dim=0).detach().cpu().numpy()
    kept_foe = np.asarray(foe)[keep.detach().cpu().numpy()]
    distribution_match = attention_match_score(kept_foe, reasoning)["overall"]
    mass = retained_attention_mass(foe, torch.where(keep)[0].tolist())
    return 0.7 * distribution_match + 0.3 * mass, mass


history = OUT / "train.jsonl"
print("AFFECTPRUNE_JOINT_START", json.dumps({
    "start_step": start_step, "target_step": STEPS, "group_size": GROUP_SIZE, "train_images": len(rows),
    "train_dev_hash": dev_hash,
    "resume_checkpoint": str(RESUME) if RESUME is not None else None,
    "sft_adapter": str(SFT_ADAPTER), "actor_lora_modules": len(layers),
    "reward": {"original_emor3": 1.0, "attention": ALIGN_WEIGHT, "actual_drop": DROP_WEIGHT},
    "kl_coef": KL_COEF, "selector_policy": SELECTOR_POLICY,
    "mode": "same-step joint actor-selector policy gradient",
}), flush=True)
for step in range(start_step, STEPS):
    row = rows[int(order[step % len(order)])]
    image = Image.open(row["images"][0]).convert("RGB")
    factor = math.sqrt(262144 / (image.width * image.height))
    image = image.resize((int(image.width * factor), int(image.height * factor)))
    prompt = format_prompt.render(content=row["problem"]).strip()
    inputs = model_inputs(image, prompt)
    model.eval()
    foe, foe_meta, _ = attention_view(
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
    if len(foe) != features.shape[-2] * features.shape[-1]:
        raise ValueError("FoE map and current visual groups differ")
    spatial_logits, budget_logits = selector(features, foe, confidence["confidence"])
    keep_probabilities = torch.sigmoid(spatial_logits.detach().float())
    selector_stats = {
        "keep_probability_mean": float(keep_probabilities.mean()),
        "keep_probability_min": float(keep_probabilities.min()),
        "keep_probability_max": float(keep_probabilities.max()),
        "keep_probability_std": float(keep_probabilities.std(unbiased=False)),
        "mask_entropy_mean": float(torch.distributions.Bernoulli(probs=keep_probabilities).entropy().mean()),
    }
    # Retain the score-function signal for a zero-gradient diagnostic only.
    # These read-only diagnostics do not consume RNG or change the loss.
    spatial_logits.retain_grad()
    keep_masks = []
    group = []
    for _ in range(GROUP_SIZE):
        keep, selector_logprob, target_drop = selector.sample_action(spatial_logits, budget_logits)
        keep_masks.append(keep.detach().clone())
        with torch.no_grad():
            compact = compact_prompt(model, inputs, keep)
            response_ids = generate_compacted(model, compact, MAX_TOKENS, do_sample=True)
        # Generate uses inference_mode; make ordinary index tensors for the
        # subsequent differentiable teacher-forced actor log-probability pass.
        response_ids = response_ids.clone()
        response = processor.batch_decode(response_ids, skip_special_tokens=True)[0]
        alignment, retained_mass = matching_score(compact, response_ids, foe, keep)
        rethink_text, rethink_image = reflection(response, image)
        official = original_reward.compute_score([{
            "response": response, "ground_truth": row["answer"],
            "rethink_result": rethink_text, "rethink_result_image": rethink_image,
        }])[0]
        drop_ratio = 1 - compact["visual_tokens_after"] / compact["visual_tokens_before"]
        reward = float(official["overall"] + ALIGN_WEIGHT * alignment + DROP_WEIGHT * drop_ratio)
        group.append({
            "compact": compact, "response_ids": response_ids, "selector_logprob": selector_logprob,
            "reward": reward, "original_reward": float(official["overall"]),
            "official_format": float(official["format"]), "accuracy": float(official["accuracy"]),
            "attention_match": alignment, "foe_retained_mass": retained_mass,
            "drop_ratio": drop_ratio, "target_drop": target_drop,
            "response": response, "visual_tokens_before": compact["visual_tokens_before"],
            "visual_tokens_after": compact["visual_tokens_after"],
        })
    rewards = torch.tensor([x["reward"] for x in group], device=model.device)
    advantages = (rewards - rewards.mean()) / rewards.std(unbiased=False).clamp_min(1e-5)
    actor_optimizer.zero_grad(set_to_none=True)
    selector_optimizer.zero_grad(set_to_none=True)
    model.train()
    losses = []
    for item, advantage in zip(group, advantages):
        batch = append_response(model, item["compact"], item["response_ids"])
        actor_tokens = sampled_token_log_probs(model, batch)
        actor_logprob = actor_tokens.sum() / actor_tokens.numel() ** 0.5
        with torch.inference_mode():
            reference_tokens = sampled_token_log_probs(reference, batch)
        # Apply the nonnegative k3 estimator token-wise. Using the aggregate
        # normalized sequence log-probability as a density ratio would make
        # this term length-dependent and would not approximate token KL.
        log_ref_over_actor = (reference_tokens - actor_tokens).clamp(-20, 20)
        sample_kl = (torch.expm1(log_ref_over_actor) - log_ref_over_actor).mean()
        item["sample_kl"] = float(sample_kl.detach())
        losses.append(-advantage.detach() * (actor_logprob + item["selector_logprob"]) + KL_COEF * sample_kl)
    loss = torch.stack(losses).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("Joint loss is nonfinite")
    loss.backward()
    actor_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
    selector_norm = float(torch.nn.utils.clip_grad_norm_(selector.parameters(), 5.0))
    selector_parameters = list(selector.parameters())
    selector_missing = sum(parameter.grad is None for parameter in selector_parameters)
    actor_present = sum(parameter.grad is not None for parameter in trainable)
    selector_finite = all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
                          for parameter in selector_parameters)
    actor_finite = all(bool(torch.isfinite(parameter.grad).all())
                       for parameter in trainable if parameter.grad is not None)
    masks_identical = all(torch.equal(keep_masks[0], mask) for mask in keep_masks[1:])
    zero_advantages = bool((advantages == 0).all())
    advantage_sum = float(advantages.sum())
    logits_gradient = spatial_logits.grad
    logits_gradient_norm = (float(torch.linalg.vector_norm(logits_gradient))
                            if logits_gradient is not None else None)
    # For identical actions, each Bernoulli score is identical. With centered
    # advantages summing to zero, the selector score-function terms cancel
    # exactly even when the two answer rewards differ. This is not a missing
    # gradient and both AdamW steps must still run (momentum/decay included).
    expected_zero_signal = zero_advantages or (masks_identical and abs(advantage_sum) <= 1e-6)
    allowed_zero_selector = (
        SELECTOR_POLICY == "free_bernoulli" and selector_norm == 0
        and selector_missing == 0 and selector_finite and expected_zero_signal
        and logits_gradient is not None and bool(torch.isfinite(logits_gradient).all())
        and logits_gradient_norm == 0
    )
    gradient_diagnostics = {
        "selector_missing_parameters": selector_missing,
        "actor_present_parameters": actor_present,
        "selector_finite": selector_finite, "actor_finite": actor_finite,
        "masks_identical": masks_identical,
        "keep_counts": [int(mask.sum()) for mask in keep_masks],
        "mask_sha256": [hashlib.sha256(mask.cpu().numpy().tobytes()).hexdigest() for mask in keep_masks],
        "advantages": advantages.detach().cpu().tolist(),
        "advantage_sum": advantage_sum,
        "selector_logits_gradient_norm": logits_gradient_norm,
        "selector_logit_min": float(spatial_logits.detach().min()),
        "selector_logit_max": float(spatial_logits.detach().max()),
        "zero_selector_reason": (
            "all_group_advantages_zero" if allowed_zero_selector and zero_advantages
            else "identical_actions_centered_advantage_cancellation" if allowed_zero_selector
            else None
        ),
    }
    if (not math.isfinite(actor_norm) or not math.isfinite(selector_norm)
            or actor_norm <= 0 or actor_present == 0 or not actor_finite
            or selector_missing != 0 or not selector_finite
            or (selector_norm == 0 and not allowed_zero_selector)):
        raise FloatingPointError(
            "Joint gradient failed validation: "
            f"step={step + 1}, id={row['id']}, actor_norm={actor_norm!r}, "
            f"selector_norm={selector_norm!r}, rewards={[x['reward'] for x in group]!r}, "
            f"loss={float(loss.detach())!r}, diagnostics={gradient_diagnostics!r}"
        )
    if allowed_zero_selector:
        print("AFFECTPRUNE_LEGAL_ZERO_SELECTOR", json.dumps({
            "step": step + 1, "id": row["id"],
            "rewards": [item["reward"] for item in group],
            **gradient_diagnostics,
        }), flush=True)
    actor_optimizer.step()
    selector_optimizer.step()
    record = {
        "step": step + 1, "id": row["id"], "loss": float(loss.detach()),
        "actor_gradient_norm": actor_norm, "selector_gradient_norm": selector_norm,
        "gradient_diagnostics": gradient_diagnostics,
        "sepm_confidence": confidence, "selector_policy": SELECTOR_POLICY,
        "selector_stats": selector_stats, "group": [
            {key: value for key, value in item.items() if key not in ("compact", "response_ids", "selector_logprob")}
            for item in group
        ],
    }
    with history.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print("AFFECTPRUNE_JOINT_STEP", json.dumps({
        "step": step + 1, "rewards": [round(x["reward"], 4) for x in group],
        "drop": [round(x["drop_ratio"], 3) for x in group],
        "format": [x["official_format"] for x in group],
        "actor_grad": round(actor_norm, 4), "selector_grad": round(selector_norm, 4),
    }), flush=True)
    if (step + 1) % SAVE_EVERY == 0 or step + 1 == STEPS:
        checkpoint = {
            "actor_lora": adapter_state(model), "selector": selector.state_dict(),
            "sft_adapter": str(SFT_ADAPTER), "step": step + 1,
            "seed": SEED, "selector_policy": SELECTOR_POLICY,
            "base_model": str(model_path),
            "train_dev_hash": dev_hash,
            "budgets": None,
        }
        torch.save(checkpoint, OUT / f"joint_step{step + 1}.pt")
        if (step + 1) % FULL_SAVE_EVERY == 0 or step + 1 == STEPS:
            torch.save({
                **checkpoint,
                "actor_optimizer": actor_optimizer.state_dict(),
                "selector_optimizer": selector_optimizer.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
            }, OUT / f"resume_step{step + 1}.pt")
print("AFFECTPRUNE_JOINT_COMPLETED", STEPS, flush=True)

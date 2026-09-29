#!/usr/bin/env python3
"""Diagnose whether EMO-R3 reasoning attends to emotion-relevant visual tokens.

The script forms two independent attention views for each image:

1. FoE attention: visual attention queried by "Please focus on emotion".
2. Reasoning attention: visual attention queried by the generated SET steps.

Their agreement is a non-tautological candidate reward for later GRPO training.
The script intentionally performs diagnosis only; it does not claim runtime
speedups before actual token compaction is enabled in the rollout engine.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

FOE_TEXT = "Please focus on emotion."
NEUTRAL_TEXT = "Please describe the image."
SET_SUFFIX = r"""

You First think the emotional meaning step by step, and then provide the final answer.

The reasoning process MUST follow the structure below:
<step1>Identify what in the scene could trigger emotion (action, face, environment).</step1>
<step2>Describe how a human would feel about it.</step2>
<step3>Conclude if the emotion is positive or negative, and if it's high or low arousal.</step3>

The final answer MUST BE put in \boxed{}.
""".strip()

_STEP_PATTERN = re.compile(r"<step[123]>(.*?)</step[123]>", re.DOTALL | re.IGNORECASE)
_OPEN_STEP_PATTERN = re.compile(
    r"<step([123])>(.*?)(?=</step\1>|<step[123]>|\\boxed\{|$)",
    re.DOTALL | re.IGNORECASE,
)
_BOX_PATTERN = re.compile(r"\\boxed\{([^{}]+)\}")
_EPS = 1e-12


# Keep the offline diagnostic self-contained. Importing ``verl`` here would
# execute the training package's ``__init__`` and require unrelated runtime
# dependencies (for example codetiming, ray, and the rollout stack).
def _as_attention_vector(values: Iterable[float]) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.size == 0:
        raise ValueError("Attention vectors must not be empty.")
    if not np.all(np.isfinite(vector)):
        raise ValueError("Attention vectors must contain only finite values.")
    return vector


def normalize_attention(values: Iterable[float]) -> np.ndarray:
    vector = _as_attention_vector(values)
    if np.any(vector < 0):
        raise ValueError("Attention scores must be non-negative.")
    total = float(vector.sum())
    if total <= _EPS:
        return np.full_like(vector, 1.0 / vector.size)
    return vector / total


def _jensen_shannon_similarity(first: Iterable[float], second: Iterable[float]) -> float:
    p = normalize_attention(first)
    q = normalize_attention(second)
    if p.shape != q.shape:
        raise ValueError(f"Attention shapes must match, got {p.shape} and {q.shape}.")
    midpoint = 0.5 * (p + q)
    kl_pm = np.sum(np.where(p > 0, p * np.log((p + _EPS) / (midpoint + _EPS)), 0.0))
    kl_qm = np.sum(np.where(q > 0, q * np.log((q + _EPS) / (midpoint + _EPS)), 0.0))
    divergence = 0.5 * float(kl_pm + kl_qm)
    return float(np.clip(1.0 - divergence / math.log(2.0), 0.0, 1.0))


def _cosine_attention_similarity(first: Iterable[float], second: Iterable[float]) -> float:
    p = normalize_attention(first)
    q = normalize_attention(second)
    if p.shape != q.shape:
        raise ValueError(f"Attention shapes must match, got {p.shape} and {q.shape}.")
    denominator = float(np.linalg.norm(p) * np.linalg.norm(q))
    if denominator <= _EPS:
        return 0.0
    return float(np.clip(np.dot(p, q) / denominator, 0.0, 1.0))


def _topk_overlap_metrics(
    reference: Iterable[float], candidate: Iterable[float], top_fraction: float = 0.2
) -> Dict[str, float]:
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("top_fraction must be in (0, 1].")
    ref = _as_attention_vector(reference)
    cand = _as_attention_vector(candidate)
    if ref.shape != cand.shape:
        raise ValueError(f"Attention shapes must match, got {ref.shape} and {cand.shape}.")
    k = max(1, int(math.ceil(ref.size * top_fraction)))
    ref_top = set(np.argpartition(ref, -k)[-k:].tolist())
    cand_top = set(np.argpartition(cand, -k)[-k:].tolist())
    intersection = len(ref_top & cand_top)
    union = len(ref_top | cand_top)
    precision = intersection / len(cand_top)
    recall = intersection / len(ref_top)
    f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(intersection / union),
    }


def attention_match_score(
    emotion_attention: Iterable[float],
    reasoning_attention: Iterable[float],
    top_fraction: float = 0.2,
    distribution_weight: float = 0.7,
) -> Dict[str, float]:
    if not 0.0 <= distribution_weight <= 1.0:
        raise ValueError("distribution_weight must be in [0, 1].")
    js_similarity = _jensen_shannon_similarity(emotion_attention, reasoning_attention)
    cosine_similarity = _cosine_attention_similarity(emotion_attention, reasoning_attention)
    overlap = _topk_overlap_metrics(emotion_attention, reasoning_attention, top_fraction)
    overall = distribution_weight * js_similarity + (1.0 - distribution_weight) * overlap["f1"]
    return {
        "overall": float(overall),
        "js_similarity": js_similarity,
        "cosine_similarity": cosine_similarity,
        "topk_precision": overlap["precision"],
        "topk_recall": overlap["recall"],
        "topk_f1": overlap["f1"],
        "topk_iou": overlap["iou"],
    }


def retained_attention_mass(attention: Iterable[float], keep_indices: Iterable[int]) -> float:
    distribution = normalize_attention(attention)
    indices = np.asarray(list(keep_indices), dtype=np.int64)
    if indices.size == 0:
        return 0.0
    if np.any(indices < 0) or np.any(indices >= distribution.size):
        raise IndexError("keep_indices contains an out-of-range visual token index.")
    return float(distribution[indices].sum())


def contrastive_emotion_attention(
    focus_attention: Iterable[float], neutral_attention: Iterable[float]
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Remove query-independent spatial bias from Focus-on-Emotion attention."""

    focus = normalize_attention(focus_attention)
    neutral = normalize_attention(neutral_attention)
    if focus.shape != neutral.shape:
        raise ValueError(f"Attention shapes must match, got {focus.shape} and {neutral.shape}.")
    positive = np.clip(focus - neutral, 0.0, None)
    positive_mass = float(positive.sum())
    fallback = positive_mass <= _EPS
    contrastive = focus if fallback else normalize_attention(positive)
    return contrastive, {
        "positive_difference_mass": positive_mass,
        "fallback_to_raw_foe": fallback,
    }


def attention_statistics(attention: Iterable[float]) -> Dict[str, float]:
    distribution = normalize_attention(attention)
    entropy = -float(np.sum(np.where(distribution > 0, distribution * np.log(distribution + _EPS), 0.0)))
    normalized_entropy = entropy / math.log(distribution.size) if distribution.size > 1 else 0.0
    sorted_attention = np.sort(distribution)[::-1]
    statistics = {
        "entropy": entropy,
        "normalized_entropy": float(normalized_entropy),
        "maximum": float(sorted_attention[0]),
    }
    for fraction in (0.1, 0.2, 0.4):
        count = max(1, int(math.ceil(distribution.size * fraction)))
        statistics[f"top_{int(fraction * 100)}pct_mass"] = float(sorted_attention[:count].sum())
    return statistics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-jsonl", type=Path, help="JSONL with image(s), problem/prompt, and optional response.")
    source.add_argument("--dataset", help="Hugging Face dataset name, e.g. fuyyy74/EmoSet2k.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--model-path", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--last-layers", type=int, default=1)
    parser.add_argument("--top-fraction", type=float, default=0.2)
    parser.add_argument(
        "--require-set",
        action="store_true",
        help="Fail when a response lacks parseable <step1>-<step3> blocks instead of using the final response.",
    )
    parser.add_argument("--drop-ratios", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4])
    parser.add_argument(
        "--run-counterfactual",
        action="store_true",
        help="Also mask low- vs high-attention image patches and rerun reasoning to test causal faithfulness.",
    )
    parser.add_argument("--counterfactual-ratio", type=float, default=0.2)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-area", type=int, default=None, help="Resize input to this pixel area before analysis.")
    parser.add_argument("--use-fast", action="store_true", help="Use the fast image processor used by training.")
    return parser.parse_args()


def load_records(args: argparse.Namespace) -> Iterable[Mapping[str, Any]]:
    if args.input_jsonl is not None:
        with args.input_jsonl.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)
        return

    from datasets import load_dataset

    yield from load_dataset(args.dataset, split=args.split)


def load_image(value: Any, image_root: Path | None) -> Image.Image:
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("The record contains an empty image list.")
        value = value[0]
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, bytes):
        return Image.open(io.BytesIO(value)).convert("RGB")
    if isinstance(value, Mapping):
        if value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        value = value.get("path")
    if value is None:
        raise ValueError("No image value was found in the record.")

    path = Path(value)
    if image_root is not None and not path.is_absolute():
        path = image_root / path
    with Image.open(path) as image:
        return image.convert("RGB")


def get_record_image(record: Mapping[str, Any], image_root: Path | None) -> Image.Image:
    for key in ("images", "image"):
        if key in record:
            return load_image(record[key], image_root)
    raise KeyError("Each record must contain an 'images' or 'image' field.")


def get_prompt(record: Mapping[str, Any]) -> str:
    for key in ("problem", "prompt", "question", "text"):
        if record.get(key):
            return str(record[key]).replace("<image>", "").strip()
    raise KeyError("Each record must contain problem, prompt, question, or text.")


def get_ground_truth(record: Mapping[str, Any]) -> str | None:
    for key in ("answer", "ground_truth", "label"):
        if record.get(key) is not None:
            return str(record[key]).strip()
    return None


def structured_prompt(prompt: str) -> str:
    return f"{prompt.strip()}\n\n{SET_SUFFIX}"


def user_messages(image: Image.Image, text: str) -> List[Dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text},
            ],
        }
    ]


def prepare_inputs(processor: Any, image: Image.Image, text: str, response: str | None = None) -> Dict[str, Any]:
    messages = user_messages(image, text)
    if response is not None:
        messages.append({"role": "assistant", "content": response})
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=response is None)
    return dict(processor(text=[rendered], images=[image], padding=True, return_tensors="pt"))


def move_to_model_device(inputs: Mapping[str, Any], model: Any) -> Dict[str, Any]:
    device = next(model.parameters()).device
    moved: Dict[str, Any] = {}
    for key, value in inputs.items():
        moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved


def find_subsequence(sequence: Sequence[int], target: Sequence[int], prefer_last: bool = True) -> List[int]:
    if not target or len(target) > len(sequence):
        return []
    matches = [
        start
        for start in range(len(sequence) - len(target) + 1)
        if list(sequence[start : start + len(target)]) == list(target)
    ]
    if not matches:
        return []
    start = matches[-1] if prefer_last else matches[0]
    return list(range(start, start + len(target)))


def token_indices_for_text(tokenizer: Any, input_ids: Sequence[int], text: str) -> List[int]:
    normalized = text.strip()
    candidates = [normalized, " " + normalized, "\n" + normalized, "\n\n" + normalized]
    tokenized_candidates: List[List[int]] = []
    for candidate in candidates:
        token_ids = tokenizer.encode(candidate, add_special_tokens=False)
        if token_ids not in tokenized_candidates:
            tokenized_candidates.append(token_ids)
        indices = find_subsequence(input_ids, token_ids)
        if indices:
            return indices

    # Byte-level BPE tokenizers can merge the first/last word with whitespace
    # inserted by the chat template. Preserve the semantic core by trimming at
    # most two boundary tokens and retrying, preferring the longest match.
    fragments: List[List[int]] = []
    for token_ids in tokenized_candidates:
        for total_trim in (1, 2):
            for left_trim in range(total_trim + 1):
                right_trim = total_trim - left_trim
                stop = len(token_ids) - right_trim if right_trim else len(token_ids)
                fragment = token_ids[left_trim:stop]
                if len(fragment) >= 2 and fragment not in fragments:
                    fragments.append(fragment)
    fragments.sort(key=len, reverse=True)
    for fragment in fragments:
        indices = find_subsequence(input_ids, fragment)
        if indices:
            return indices
    return []


def aggregate_visual_attention(
    attentions: Sequence[torch.Tensor], query_indices: Sequence[int], visual_indices: Sequence[int], last_layers: int
) -> np.ndarray:
    if not query_indices:
        raise ValueError("No query-token span was found for attention aggregation.")
    if not visual_indices:
        raise ValueError("No visual tokens were found in the processed input.")
    if last_layers <= 0:
        raise ValueError("last_layers must be positive.")

    selected_layers = attentions[-min(last_layers, len(attentions)) :]
    accumulated = None
    for layer_attention in selected_layers:
        # [batch, heads, query, key] -> [query, key]
        mean_heads = layer_attention[0].float().mean(dim=0)
        visual_scores = mean_heads[query_indices][:, visual_indices].mean(dim=0).detach().cpu()
        accumulated = visual_scores if accumulated is None else accumulated + visual_scores
    return normalize_attention((accumulated / len(selected_layers)).numpy())


@torch.inference_mode()
def attention_view(
    model: Any,
    processor: Any,
    image: Image.Image,
    prompt: str,
    query_texts: Sequence[str],
    last_layers: int,
    response: str | None = None,
) -> Tuple[np.ndarray, Dict[str, Any], List[Tuple[str, np.ndarray]]]:
    inputs = prepare_inputs(processor, image, prompt, response=response)
    model_inputs = move_to_model_device(inputs, model)
    outputs = model(**model_inputs, output_attentions=True, use_cache=False, return_dict=True)
    if outputs.attentions is None:
        raise RuntimeError("The model returned no attentions. Load it with attn_implementation='eager'.")

    ids = inputs["input_ids"][0].tolist()
    image_token_id = getattr(model.config, "image_token_id", None)
    if image_token_id is None:
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    visual_indices = [index for index, token_id in enumerate(ids) if token_id == image_token_id]

    query_indices: List[int] = []
    query_spans: List[Tuple[str, List[int]]] = []
    for query_text in query_texts:
        span = token_indices_for_text(processor.tokenizer, ids, query_text)
        if span:
            query_indices.extend(span)
            query_spans.append((query_text, span))
    query_indices = sorted(set(query_indices))
    attention = aggregate_visual_attention(outputs.attentions, query_indices, visual_indices, last_layers)
    per_query_attentions = [
        (query_text, aggregate_visual_attention(outputs.attentions, span, visual_indices, last_layers))
        for query_text, span in query_spans
    ]

    grid_thw = inputs.get("image_grid_thw")
    metadata = {
        "visual_token_count": len(visual_indices),
        "query_token_count": len(query_indices),
        "matched_query_count": len(query_spans),
        "requested_query_count": len(query_texts),
        "image_grid_thw": grid_thw[0].tolist() if grid_thw is not None else None,
    }
    del outputs, model_inputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return attention, metadata, per_query_attentions


@torch.inference_mode()
def generate_response(
    model: Any,
    processor: Any,
    image: Image.Image,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    inputs = prepare_inputs(processor, image, prompt)
    model_inputs = move_to_model_device(inputs, model)
    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
        "do_sample": temperature > 0.0,
    }
    if temperature > 0.0:
        generate_kwargs["temperature"] = temperature
    generated = model.generate(**model_inputs, **generate_kwargs)
    prompt_length = inputs["input_ids"].shape[1]
    return processor.batch_decode(generated[:, prompt_length:], skip_special_tokens=True)[0].strip()


def infer_grid_shape(grid_thw: Sequence[int] | None, token_count: int, merge_size: int) -> Tuple[int, int]:
    if grid_thw is not None:
        temporal, height, width = [int(value) for value in grid_thw]
        merged_height = height // merge_size
        merged_width = width // merge_size
        if temporal * merged_height * merged_width == token_count:
            if temporal != 1:
                raise ValueError("The current diagnostic supports images, not multi-frame video attention.")
            return merged_height, merged_width

    height = int(math.sqrt(token_count))
    while height > 1 and token_count % height != 0:
        height -= 1
    return height, token_count // height


def _display_normalize(attention: np.ndarray, grid_shape: Tuple[int, int]) -> np.ndarray:
    heatmap = np.asarray(attention, dtype=np.float64).reshape(grid_shape)
    low = float(np.percentile(heatmap, 2.0))
    high = float(np.percentile(heatmap, 99.0))
    if high <= low + _EPS:
        low = float(heatmap.min())
        high = float(heatmap.max())
    if high <= low + _EPS:
        return np.zeros_like(heatmap, dtype=np.float32)
    return np.asarray(np.clip((heatmap - low) / (high - low), 0.0, 1.0), dtype=np.float32)


def _emotion_colormap(values: np.ndarray) -> np.ndarray:
    """Dark-purple to red/yellow colormap implemented without matplotlib."""

    stops = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    colors = np.asarray(
        [[4, 3, 18], [62, 18, 110], [169, 42, 92], [240, 105, 32], [252, 245, 155]],
        dtype=np.float32,
    )
    flat = np.clip(values, 0.0, 1.0).reshape(-1)
    colored = np.stack([np.interp(flat, stops, colors[:, channel]) for channel in range(3)], axis=-1)
    return np.uint8(np.clip(colored.reshape(*values.shape, 3), 0.0, 255.0))


def save_attention_visuals(
    image: Image.Image,
    attention: np.ndarray,
    grid_shape: Tuple[int, int],
    output_stem: Path,
    top_fraction: float,
) -> None:
    """Save raw attention, a pure heatmap, an overlay, and a top-k mask."""

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_stem.with_name(output_stem.name + "_attention.npy"), np.asarray(attention, dtype=np.float32))

    normalized = _display_normalize(attention, grid_shape)
    colored = Image.fromarray(_emotion_colormap(normalized)).resize(image.size, resample=Image.Resampling.BILINEAR)
    colored.save(output_stem.with_name(output_stem.name + "_heatmap.png"))

    alpha_grid = Image.fromarray(np.uint8(205 * np.power(normalized, 0.75)))
    alpha = alpha_grid.resize(image.size, resample=Image.Resampling.BILINEAR)
    overlay = Image.composite(colored, image.convert("RGB"), alpha)
    overlay.save(output_stem.with_name(output_stem.name + "_overlay.png"))

    count = max(1, int(math.ceil(attention.size * top_fraction)))
    selected = np.argpartition(attention, -count)[-count:]
    mask_grid = np.zeros(attention.size, dtype=np.uint8)
    mask_grid[selected] = 180
    mask = Image.fromarray(mask_grid.reshape(grid_shape)).resize(image.size, resample=Image.Resampling.NEAREST)
    highlight = Image.new("RGB", image.size, color=(255, 45, 35))
    topk_overlay = Image.composite(highlight, image.convert("RGB"), mask)
    topk_overlay.save(output_stem.with_name(output_stem.name + f"_top{int(top_fraction * 100)}.png"))


def save_signed_attention_difference(
    emotion_attention: np.ndarray,
    reasoning_attention: np.ndarray,
    grid_shape: Tuple[int, int],
    output_path: Path,
) -> None:
    """Save a signed map: blue is emotion-only and red is reasoning-only."""

    difference = np.asarray(reasoning_attention - emotion_attention, dtype=np.float64).reshape(grid_shape)
    scale = float(np.percentile(np.abs(difference), 99.0))
    if scale <= _EPS:
        signed = np.zeros_like(difference, dtype=np.float32)
    else:
        signed = np.asarray(np.clip(difference / scale, -1.0, 1.0), dtype=np.float32)
    magnitude = np.abs(signed)[..., None]
    neutral = np.full((*signed.shape, 3), 238.0, dtype=np.float32)
    blue = np.asarray([35.0, 105.0, 220.0], dtype=np.float32)
    red = np.asarray([225.0, 55.0, 45.0], dtype=np.float32)
    endpoints = np.where((signed >= 0)[..., None], red, blue)
    colors = neutral * (1.0 - magnitude) + endpoints * magnitude
    rendered = Image.fromarray(np.uint8(np.clip(colors, 0.0, 255.0))).resize(
        (grid_shape[1] * 24, grid_shape[0] * 24), resample=Image.Resampling.NEAREST
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered.save(output_path)


def mask_attention_patches(
    image: Image.Image,
    attention: np.ndarray,
    grid_shape: Tuple[int, int],
    ratio: float,
    remove_high_attention: bool,
) -> Image.Image:
    """Create a pixel-space intervention for a causal diagnostic.

    This masks image patches rather than compacting model tokens, so it tests
    evidence faithfulness but must not be reported as an acceleration result.
    """

    if not 0.0 < ratio < 1.0:
        raise ValueError("counterfactual ratio must be in (0, 1).")
    count = max(1, int(math.floor(attention.size * ratio)))
    selected = (
        np.argpartition(attention, -count)[-count:]
        if remove_high_attention
        else np.argpartition(attention, count - 1)[:count]
    )
    grid_mask = np.zeros(attention.size, dtype=np.uint8)
    grid_mask[selected] = 255
    grid_mask = grid_mask.reshape(grid_shape)
    mask = Image.fromarray(grid_mask, mode="L").resize(image.size, resample=Image.Resampling.NEAREST)
    neutral = Image.new("RGB", image.size, color=(127, 127, 127))
    return Image.composite(neutral, image.convert("RGB"), mask)


def pruning_diagnostics(
    emotion_attention: np.ndarray, reasoning_attention: np.ndarray, drop_ratios: Sequence[float]
) -> List[Dict[str, float]]:
    results = []
    token_count = emotion_attention.size
    for ratio in drop_ratios:
        if not 0.0 <= ratio < 1.0:
            raise ValueError(f"drop ratio must be in [0, 1), got {ratio}.")
        keep_count = max(1, token_count - int(math.floor(token_count * ratio)))
        keep_indices = np.argpartition(emotion_attention, -keep_count)[-keep_count:]
        results.append(
            {
                "drop_ratio": float(ratio),
                "kept_tokens": float(keep_count),
                "emotion_mass_retained": retained_attention_mass(emotion_attention, keep_indices),
                "reasoning_mass_retained": retained_attention_mass(reasoning_attention, keep_indices),
            }
        )
    return results


def answer_from_response(response: str) -> str | None:
    matches = _BOX_PATTERN.findall(response)
    return matches[-1].strip() if matches else None


def summarize(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"samples": 0}
    matches = np.asarray([row["attention"]["overall"] for row in rows], dtype=np.float64)
    raw_matches = np.asarray([row["attention_raw_foe"]["overall"] for row in rows], dtype=np.float64)
    summary: Dict[str, Any] = {
        "samples": len(rows),
        "mean_attention_match": float(matches.mean()),
        "std_attention_match": float(matches.std()),
        "mean_raw_foe_match": float(raw_matches.mean()),
        "mean_debias_match_change": float((matches - raw_matches).mean()),
        "set_format_rate": float(np.mean([bool(row.get("set_format_valid")) for row in rows])),
        "three_step_tag_rate": float(np.mean([row.get("step_tag_count", 0) == 3 for row in rows])),
        "open_tag_only_rate": float(np.mean([
            row.get("step_tag_count", 0) == 3 and not row.get("set_format_valid") for row in rows
        ])),
        "full_response_fallback_rate": float(np.mean([
            row.get("attention_query_source") == "final_response_fallback" for row in rows
        ])),
    }
    correctness = [row.get("correct") for row in rows]
    valid = [index for index, value in enumerate(correctness) if value is not None]
    if valid:
        labels = np.asarray([float(correctness[index]) for index in valid])
        valid_matches = matches[valid]
        summary["accuracy"] = float(labels.mean())
        if labels.size > 1 and labels.std() > 0 and valid_matches.std() > 0:
            summary["attention_correctness_pearson"] = float(np.corrcoef(valid_matches, labels)[0, 1])
        summary["mean_match_correct"] = float(valid_matches[labels == 1].mean()) if np.any(labels == 1) else None
        summary["mean_match_incorrect"] = float(valid_matches[labels == 0].mean()) if np.any(labels == 0) else None
    return summary


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dtype = getattr(torch, args.dtype)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=False, use_fast=args.use_fast)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        attn_implementation="eager",
        device_map="auto",
        trust_remote_code=False,
    )
    model.eval()

    output_rows: List[Dict[str, Any]] = []
    records_path = args.output_dir / "attention_metrics.jsonl"
    with records_path.open("w", encoding="utf-8") as output_stream:
        for index, record in enumerate(load_records(args)):
            if index >= args.max_samples:
                break
            image = get_record_image(record, args.image_root)
            if args.image_area is not None:
                scale = math.sqrt(args.image_area / (image.width * image.height))
                image = image.resize((int(image.width * scale), int(image.height * scale)))
            base_prompt = get_prompt(record)
            # For saved rollouts, attend to the response under the exact user
            # prompt that produced it. A changed prompt would invalidate the
            # reasoning-to-image attention comparison.
            prompt = str(record.get("generation_prompt") or structured_prompt(base_prompt))
            response = str(record["response"]) if record.get("response") else generate_response(
                model, processor, image, prompt, args.max_new_tokens, args.temperature
            )
            tagged_steps = [
                (int(match.group(1)), match.group(2).strip())
                for match in _OPEN_STEP_PATTERN.finditer(response)
                if match.group(2).strip()
            ]
            step_tag_count = len({number for number, _ in tagged_steps})
            reasoning_steps = [body for _, body in tagged_steps]
            # Keep the previous closed-tag metric distinct from the actual
            # step text used to query visual attention. Some trained models
            # emit all three <stepN> openings but omit their closing tags.
            set_format_valid = bool(_STEP_PATTERN.findall(response))
            attention_query_source = "set_steps_closed" if set_format_valid else "set_steps_open"
            if not reasoning_steps:
                if args.require_set:
                    raise ValueError(f"Sample {index} has no parseable SET reasoning steps: {response[:200]!r}")
                reasoning_steps = [response.strip()]
                attention_query_source = "final_response_fallback"
                print(
                    f"Warning: sample {index} has no parseable SET steps; "
                    "using final-response attention for pipeline validation.",
                    flush=True,
                )

            raw_foe_attention, emotion_meta, _ = attention_view(
                model,
                processor,
                image,
                f"{FOE_TEXT}\n{base_prompt}",
                [FOE_TEXT],
                args.last_layers,
            )
            neutral_attention, neutral_meta, _ = attention_view(
                model,
                processor,
                image,
                f"{NEUTRAL_TEXT}\n{base_prompt}",
                [NEUTRAL_TEXT],
                args.last_layers,
            )
            reasoning_attention, reasoning_meta, per_step_attentions = attention_view(
                model,
                processor,
                image,
                prompt,
                reasoning_steps,
                args.last_layers,
                response=response,
            )
            if not (
                raw_foe_attention.shape == neutral_attention.shape == reasoning_attention.shape
            ):
                raise RuntimeError(
                    "FoE, neutral, and reasoning visual-token counts differ: "
                    f"{raw_foe_attention.shape}, {neutral_attention.shape}, {reasoning_attention.shape}."
                )

            emotion_attention, contrastive_meta = contrastive_emotion_attention(
                raw_foe_attention, neutral_attention
            )
            raw_foe_match = attention_match_score(
                raw_foe_attention, reasoning_attention, top_fraction=args.top_fraction
            )
            match = attention_match_score(emotion_attention, reasoning_attention, top_fraction=args.top_fraction)
            per_step_metrics = [
                {
                    "step": step_index,
                    "query_text": query_text,
                    "attention": attention_match_score(
                        emotion_attention, step_attention, top_fraction=args.top_fraction
                    ),
                    "statistics": attention_statistics(step_attention),
                }
                for step_index, (query_text, step_attention) in enumerate(per_step_attentions, start=1)
            ] if step_tag_count else []
            grid_shape = infer_grid_shape(
                emotion_meta["image_grid_thw"],
                emotion_attention.size,
                int(getattr(processor.image_processor, "merge_size", 2)),
            )
            sample_id = str(record.get("id", record.get("question_id", index)))
            safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id)
            visual_dir = args.output_dir / "attention_visuals"
            visual_dir.mkdir(parents=True, exist_ok=True)
            image.save(visual_dir / f"{safe_id}_original.png")
            save_attention_visuals(
                image, raw_foe_attention, grid_shape, visual_dir / f"{safe_id}_raw_foe", args.top_fraction
            )
            save_attention_visuals(
                image, neutral_attention, grid_shape, visual_dir / f"{safe_id}_neutral", args.top_fraction
            )
            save_attention_visuals(
                image, emotion_attention, grid_shape, visual_dir / f"{safe_id}_contrastive_foe", args.top_fraction
            )
            save_attention_visuals(
                image, reasoning_attention, grid_shape, visual_dir / f"{safe_id}_reasoning", args.top_fraction
            )
            save_signed_attention_difference(
                emotion_attention,
                reasoning_attention,
                grid_shape,
                visual_dir / f"{safe_id}_contrastive_vs_reasoning_difference.png",
            )
            if step_tag_count:
                for step_index, (_, step_attention) in enumerate(per_step_attentions, start=1):
                    save_attention_visuals(
                        image,
                        step_attention,
                        grid_shape,
                        visual_dir / f"{safe_id}_step{step_index}",
                        args.top_fraction,
                    )

            prediction = answer_from_response(response)
            ground_truth = get_ground_truth(record)
            correct = None
            if prediction is not None and ground_truth is not None:
                correct = prediction.casefold() == ground_truth.casefold()
            counterfactual = None
            if args.run_counterfactual:
                # Generate an unmasked control with this same HF model. Saved
                # responses may come from vLLM, so comparing masked HF outputs
                # directly with them would confound masking and engine changes.
                reference_response = generate_response(
                    model, processor, image, prompt, args.max_new_tokens, args.temperature
                )
                reference_prediction = answer_from_response(reference_response)
                low_masked = mask_attention_patches(
                    image,
                    emotion_attention,
                    grid_shape,
                    args.counterfactual_ratio,
                    remove_high_attention=False,
                )
                high_masked = mask_attention_patches(
                    image,
                    emotion_attention,
                    grid_shape,
                    args.counterfactual_ratio,
                    remove_high_attention=True,
                )
                counterfactual_dir = args.output_dir / "counterfactuals"
                counterfactual_dir.mkdir(parents=True, exist_ok=True)
                low_masked.save(counterfactual_dir / f"{safe_id}_remove_low.png")
                high_masked.save(counterfactual_dir / f"{safe_id}_remove_high.png")
                low_response = generate_response(
                    model, processor, low_masked, prompt, args.max_new_tokens, args.temperature
                )
                high_response = generate_response(
                    model, processor, high_masked, prompt, args.max_new_tokens, args.temperature
                )
                low_prediction = answer_from_response(low_response)
                high_prediction = answer_from_response(high_response)
                counterfactual = {
                    "ratio": args.counterfactual_ratio,
                    "reference_prediction": reference_prediction,
                    "reference_response": reference_response,
                    "reference_matches_saved_prediction": reference_prediction is not None
                    and prediction is not None
                    and reference_prediction.casefold() == prediction.casefold(),
                    "remove_low_prediction": low_prediction,
                    "remove_high_prediction": high_prediction,
                    "remove_low_preserves_reference": low_prediction is not None
                    and reference_prediction is not None
                    and low_prediction.casefold() == reference_prediction.casefold(),
                    "remove_high_changes_reference": high_prediction is not None
                    and reference_prediction is not None
                    and high_prediction.casefold() != reference_prediction.casefold(),
                    "remove_low_preserves_answer": low_prediction is not None
                    and prediction is not None
                    and low_prediction.casefold() == prediction.casefold(),
                    "remove_high_changes_answer": high_prediction is not None
                    and prediction is not None
                    and high_prediction.casefold() != prediction.casefold(),
                    "remove_low_response": low_response,
                    "remove_high_response": high_response,
                }
            row = {
                "id": sample_id,
                "prediction": prediction,
                "ground_truth": ground_truth,
                "correct": correct,
                "response": response,
                "set_format_valid": set_format_valid,
                "step_tag_count": step_tag_count,
                "attention_query_source": attention_query_source,
                "generation_prompt_source": "record" if record.get("generation_prompt") else "default",
                "attention_query_texts": reasoning_steps,
                "attention": match,
                "attention_raw_foe": raw_foe_match,
                "per_step_attention": per_step_metrics,
                "attention_statistics": {
                    "raw_foe": attention_statistics(raw_foe_attention),
                    "neutral": attention_statistics(neutral_attention),
                    "contrastive_foe": attention_statistics(emotion_attention),
                    "reasoning": attention_statistics(reasoning_attention),
                },
                "foe_metadata": emotion_meta,
                "neutral_metadata": neutral_meta,
                "contrastive_metadata": contrastive_meta,
                "reasoning_metadata": reasoning_meta,
                "pruning": pruning_diagnostics(emotion_attention, reasoning_attention, args.drop_ratios),
                "pruning_raw_foe": pruning_diagnostics(raw_foe_attention, reasoning_attention, args.drop_ratios),
                "counterfactual": counterfactual,
            }
            output_rows.append(row)
            output_stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            output_stream.flush()

    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summarize(output_rows), stream, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

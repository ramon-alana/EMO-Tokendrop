"""Focus-on-Emotion attention and alignment metrics."""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

FOE_TEXT = "Please focus on emotion."
_EPS = 1e-12

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

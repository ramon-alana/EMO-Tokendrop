"""Confidence-conditioned Bernoulli visual-token selection."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


COARSE_QUERY = (
    "Which description best represents the image?\n"
    "A. Positive emotions: amusement, awe, contentment, excitement.\n"
    "B. Negative emotions: anger, disgust, fear, sadness.\n"
    "Answer directly with A or B."
)


def image_features(pixel_values, grid_thw, spatial_merge_unit, spatial_merge_size):
    groups = int(grid_thw.prod().item()) // spatial_merge_unit
    height = int(grid_thw[0, 1]) // spatial_merge_size
    width = int(grid_thw[0, 2]) // spatial_merge_size
    if groups != height * width or pixel_values.shape[0] != groups * spatial_merge_unit:
        raise ValueError("Visual group geometry mismatch")
    patches = pixel_values.float().reshape(groups, spatial_merge_unit, -1)
    if patches.shape[-1] % 6 == 0:
        channels = patches.reshape(groups, spatial_merge_unit, 2, 3, -1)
        means = channels.mean(dim=(1, 2, 4))
        stds = channels.std(dim=(1, 2, 4), unbiased=False)
    else:
        flat = patches.flatten(1)
        means = flat.mean(dim=-1, keepdim=True).expand(-1, 3)
        stds = flat.std(dim=-1, keepdim=True, unbiased=False).expand(-1, 3)
    ys, xs = torch.meshgrid(
        torch.linspace(-1, 1, height, device=pixel_values.device),
        torch.linspace(-1, 1, width, device=pixel_values.device), indexing="ij"
    )
    coords = torch.stack((ys.flatten(), xs.flatten()), dim=-1)
    return torch.cat((means, stds, coords), dim=-1).reshape(height, width, 8).permute(2, 0, 1).unsqueeze(0)


@torch.inference_mode()
def sepm_coarse_confidence(model, processor, image):
    """Return SEPM's A/B variance and a [0,1] confidence for conditioning."""
    message = [{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": COARSE_QUERY}
    ]}]
    rendered = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[rendered], images=[image], padding=True, return_tensors="pt")
    inputs = {key: value.to(model.device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}
    a_ids = processor.tokenizer.encode("A", add_special_tokens=False)
    b_ids = processor.tokenizer.encode("B", add_special_tokens=False)
    if len(a_ids) != 1 or len(b_ids) != 1:
        raise ValueError("A and B must each have one token for SEPM-style confidence")
    output = model(**inputs, use_cache=False)
    probabilities = torch.softmax(output.logits[0, -1].float(), dim=-1)
    p_a = float(probabilities[a_ids[0]])
    p_b = float(probabilities[b_ids[0]])
    mean = 0.5 * (p_a + p_b)
    variance = 0.5 * ((p_a - mean) ** 2 + (p_b - mean) ** 2)
    normalized = abs(p_a - p_b) / max(p_a + p_b, 1e-12)
    return {"p_a": p_a, "p_b": p_b, "sepm_variance": variance, "confidence": normalized}




class FreeRatioSelector(nn.Module):
    """Image-conditioned independent keep actions; no prescribed drop bins.

    A Bernoulli action is sampled for each visual group. The realized ratio is
    an outcome of the policy, not a training target or a fixed top-k budget.
    Hard indexing is optimized with the score-function estimator.
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(10, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.GELU(),
        )
        self.spatial_head = nn.Conv2d(32, 1, 1)
        self.image_keep_head = nn.Linear(33, 1)
        self.foe_prior_scale = nn.Parameter(torch.tensor(0.0))
        # Initial tendency only; this bias is trainable and sets no keep count.
        nn.init.zeros_(self.image_keep_head.weight)
        nn.init.constant_(self.image_keep_head.bias, 0.8)

    def forward(self, image_features, foe_attention, confidence):
        groups = image_features.shape[-2] * image_features.shape[-1]
        foe = torch.as_tensor(foe_attention, device=image_features.device, dtype=torch.float32).flatten()
        if foe.numel() != groups or not torch.isfinite(foe).all() or torch.any(foe < 0):
            raise ValueError("Invalid FoE spatial saliency")
        foe = foe / foe.sum().clamp_min(1e-12)
        standardized = torch.log(foe.clamp_min(1e-12))
        standardized = (standardized - standardized.mean()) / standardized.std(unbiased=False).clamp_min(1e-6)
        conf = torch.as_tensor(confidence, device=image_features.device, dtype=torch.float32).clamp(0, 1)
        extra = torch.stack((standardized, conf.expand(groups)), dim=0)
        extra = extra.reshape(1, 2, image_features.shape[-2], image_features.shape[-1])
        hidden = self.features(torch.cat((image_features, extra), dim=1))
        pooled = hidden.mean(dim=(2, 3)).squeeze(0)
        image_bias = self.image_keep_head(torch.cat((pooled, conf.view(1)), dim=0)).squeeze()
        logits = self.spatial_head(hidden).flatten() + image_bias
        logits = logits + F.softplus(self.foe_prior_scale) * conf * standardized
        return logits, None

    def sample_action(self, keep_logits, _unused=None):
        distribution = torch.distributions.Bernoulli(logits=keep_logits)
        keep = distribution.sample().bool()
        if not keep.any():
            # The visual encoder cannot process an empty image. Condition the
            # policy on a nonempty sample; in practice this is exceptionally
            # rare while keep probabilities remain nondegenerate.
            for _ in range(16):
                keep = distribution.sample().bool()
                if keep.any():
                    break
            else:
                raise RuntimeError("Free-ratio selector sampled 16 empty masks")
        log_p_empty = F.logsigmoid(-keep_logits).sum()
        log_nonempty = torch.log((-torch.expm1(log_p_empty)).clamp_min(1e-12))
        log_prob = (distribution.log_prob(keep.float()).sum() - log_nonempty) / math.sqrt(keep.numel())
        return keep, log_prob, None

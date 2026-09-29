"""Minimal trainable LoRA layers for Qwen2.5-VL actor cold start.

This local implementation avoids changing the shared training environment.
Only decoder linear layers are adapted; pretrained weights stay frozen until
an explicit merge for the subsequent EMO-R3 RL stage.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


TARGETS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        if rank <= 0 or alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        self.base = base
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def forward(self, x):
        base_result = self.base(x)
        low_rank = F.linear(F.linear(self.dropout(x).float(), self.lora_A), self.lora_B)
        return base_result + (low_rank * self.scaling).to(base_result.dtype)

    @torch.no_grad()
    def merge(self):
        update = (self.lora_B @ self.lora_A) * self.scaling
        self.base.weight.add_(update.to(self.base.weight.dtype))
        return self.base


def install_lora(model, rank: int = 16, alpha: float = 32.0, dropout: float = 0.05):
    """Install LoRA in decoder attention/MLP linears; return trainable params."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    replacements = []
    for name, module in model.named_modules():
        if not name.startswith("model.language_model.layers."):
            continue
        if name.rsplit(".", 1)[-1] in TARGETS and isinstance(module, nn.Linear):
            replacements.append(name)
    if not replacements:
        raise RuntimeError("No compatible Qwen2.5-VL decoder linears found")
    for name in replacements:
        parent_name, attr = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        setattr(parent, attr, LoRALinear(getattr(parent, attr), rank, alpha, dropout))
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("LoRA installation left no trainable parameters")
    return trainable, replacements


def adapter_state(model):
    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if name.endswith(".lora_A") or name.endswith(".lora_B")
    }


def load_adapter(model, state):
    result = model.load_state_dict(state, strict=False)
    unexpected = result.unexpected_keys
    if unexpected:
        raise ValueError(f"Unexpected LoRA keys: {unexpected[:5]}")
    expected_lora = {name for name, _ in model.named_parameters() if name.endswith((".lora_A", ".lora_B"))}
    if expected_lora.intersection(result.missing_keys):
        raise ValueError("LoRA checkpoint misses trainable parameters")


def merge_lora(model):
    count = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            parent_name, attr = name.rsplit(".", 1)
            setattr(model.get_submodule(parent_name), attr, module.merge())
            count += 1
    if not count:
        raise RuntimeError("No LoRA layers to merge")
    return count

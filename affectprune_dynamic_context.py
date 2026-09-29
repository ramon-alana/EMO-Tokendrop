"""Single-image Qwen2.5-VL pre-ViT pruning with genuinely shorter LLM context.

Unlike the earlier zero-restoration probe, dropped image placeholders are
removed from the language sequence. Original multimodal RoPE coordinates are
retained for kept patches; the helper returns inputs_embeds and position_ids
for a direct Qwen2.5-VL forward. It supports one still image per call.
"""

from __future__ import annotations

from types import MethodType

import torch


def install_compacted_visual_forward(model):
    visual = model.visual
    if hasattr(visual, "_dynamic_original_forward"):
        return
    visual._dynamic_original_forward = visual.forward

    def compacted_forward(self, hidden_states, grid_thw):
        keep = getattr(self, "_dynamic_keep_mask", None)
        if keep is None:
            return self._dynamic_original_forward(hidden_states, grid_thw)
        if len(grid_thw) != 1 or int(grid_thw[0, 0]) != 1:
            raise ValueError("Only one still image is supported")
        hidden_states = self.patch_embed(hidden_states)
        rotary = self.rot_pos_emb(grid_thw)
        window_index, original_cu = self.get_window_index(grid_thw)
        unit = self.spatial_merge_unit
        groups = hidden_states.shape[0] // unit
        if hidden_states.shape[0] % unit or keep.numel() != groups:
            raise ValueError("Visual mask and patch grid differ")
        keep_window = keep.to(dtype=torch.bool, device=window_index.device)[window_index]
        if not bool(keep_window.any()):
            raise ValueError("At least one visual token must be kept")
        hidden_states = hidden_states.reshape(groups, unit, -1)[window_index][keep_window].reshape(
            -1, self.config.hidden_size
        )
        rotary = rotary.reshape(groups, unit, -1)[window_index][keep_window].reshape(-1, rotary.shape[-1])
        emb = torch.cat((rotary, rotary), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        new_cu = [0]
        for first, last in zip(original_cu[:-1], original_cu[1:]):
            count = int(keep_window[first // unit:last // unit].sum()) * unit
            if count:
                new_cu.append(new_cu[-1] + count)
        cu_window = torch.tensor(new_cu, device=hidden_states.device, dtype=torch.int32)
        cu_full = torch.tensor([0, hidden_states.shape[0]], device=hidden_states.device, dtype=torch.int32)
        for layer_num, block in enumerate(self.blocks):
            cu = cu_full if layer_num in self.fullatt_block_indexes else cu_window
            hidden_states = block(hidden_states, cu_seqlens=cu, position_embeddings=position_embeddings)
        kept = self.merger(hidden_states)
        original_indices = window_index[keep_window]
        return kept[torch.argsort(original_indices)].contiguous()

    visual.forward = MethodType(compacted_forward, visual)


def compact_prompt(model, inputs, keep_mask):
    """Return compact input IDs, embeddings, RoPE positions, and measurements."""
    install_compacted_visual_forward(model)
    input_ids = inputs["input_ids"]
    if input_ids.shape[0] != 1:
        raise ValueError("Batch size must be one")
    pixel_values = inputs["pixel_values"]
    grid_thw = inputs["image_grid_thw"]
    image_positions = torch.where(input_ids[0] == model.config.image_token_id)[0]
    keep_mask = keep_mask.to(device=input_ids.device, dtype=torch.bool).flatten()
    if image_positions.numel() != keep_mask.numel():
        raise ValueError("Image placeholder count must equal mask size")
    if not bool(keep_mask.any()):
        raise ValueError("Cannot drop every image token")
    full_positions, _ = model.model.get_rope_index(
        input_ids, grid_thw, None, None, inputs.get("attention_mask")
    )
    model.visual._dynamic_keep_mask = keep_mask
    try:
        image_embeds = model.visual(pixel_values, grid_thw)
    finally:
        model.visual._dynamic_keep_mask = None
    if image_embeds.shape[0] != int(keep_mask.sum()):
        raise ValueError("Compacted ViT output size does not match kept tokens")
    seq_keep = torch.ones(input_ids.shape[1], dtype=torch.bool, device=input_ids.device)
    seq_keep[image_positions[~keep_mask]] = False
    compact_ids = input_ids[:, seq_keep]
    compact_positions = full_positions[:, :, seq_keep]
    compact_embeds = model.get_input_embeddings()(compact_ids)
    image_slots = compact_ids == model.config.image_token_id
    compact_embeds = compact_embeds.masked_scatter(
        image_slots.unsqueeze(-1).expand_as(compact_embeds),
        image_embeds.to(dtype=compact_embeds.dtype, device=compact_embeds.device),
    )
    if compact_ids.shape[1] != input_ids.shape[1] - int((~keep_mask).sum()):
        raise AssertionError("LLM context was not shortened by the drop count")
    return {
        "input_ids": compact_ids,
        "inputs_embeds": compact_embeds,
        "position_ids": compact_positions,
        "attention_mask": torch.ones_like(compact_ids),
        "visual_tokens_before": int(keep_mask.numel()),
        "visual_tokens_after": int(keep_mask.sum()),
    }


def append_response(model, compact, response_ids):
    """Attach supervised response tokens while masking prompt loss."""
    if response_ids.ndim != 2 or response_ids.shape[0] != 1:
        raise ValueError("Response must have shape [1, T]")
    response_ids = response_ids.to(compact["input_ids"].device)
    response_embeds = model.get_input_embeddings()(response_ids)
    prefix_max = compact["position_ids"].amax(dim=-1, keepdim=True)
    offsets = torch.arange(1, response_ids.shape[1] + 1, device=response_ids.device).view(1, 1, -1)
    response_positions = prefix_max + offsets
    return {
        "inputs_embeds": torch.cat((compact["inputs_embeds"], response_embeds), dim=1),
        "position_ids": torch.cat((compact["position_ids"], response_positions), dim=-1),
        "attention_mask": torch.ones(
            (1, compact["input_ids"].shape[1] + response_ids.shape[1]),
            dtype=torch.long,
            device=response_ids.device,
        ),
        "labels": torch.cat(
            (torch.full_like(compact["input_ids"], -100), response_ids), dim=1
        ),
    }


@torch.inference_mode()
def generate_compacted(model, compact, max_new_tokens, *, temperature=1.0, do_sample=False):
    """Decode one rollout from the shortened multimodal context with KV cache."""
    if max_new_tokens < 1 or temperature <= 0:
        raise ValueError("Invalid generation settings")
    prompt_length = compact["input_ids"].shape[1]
    output = model(
        inputs_embeds=compact["inputs_embeds"],
        attention_mask=compact["attention_mask"],
        position_ids=compact["position_ids"],
        use_cache=True,
    )
    past = output.past_key_values
    logits = output.logits[:, -1, :]
    next_position = int(compact["position_ids"].amax()) + 1
    tokens = []
    eos = model.config.text_config.eos_token_id
    for step in range(max_new_tokens):
        if do_sample:
            probs = torch.softmax(logits.float() / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = logits.argmax(dim=-1, keepdim=True)
        tokens.append(next_token)
        if int(next_token.item()) == eos or step + 1 == max_new_tokens:
            break
        positions = torch.full((3, 1, 1), next_position, dtype=torch.long, device=next_token.device)
        attention_mask = torch.ones(
            (1, prompt_length + step + 1), dtype=torch.long, device=next_token.device
        )
        output = model(
            input_ids=next_token,
            position_ids=positions,
            attention_mask=attention_mask,
            past_key_values=past,
            use_cache=True,
        )
        past = output.past_key_values
        logits = output.logits[:, -1, :]
        next_position += 1
    return torch.cat(tokens, dim=1)

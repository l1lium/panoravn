from __future__ import annotations

import logging
import math
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be a positive integer, got {rank!r}")
        if base.weight.dim() != 2:
            raise ValueError("LoRALinear only supports 2-D weight tensors (nn.Linear)")
        for p in base.parameters():
            p.requires_grad_(False)
        self.base = base
        d_out, d_in = base.weight.shape
        self.rank = rank
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale

    def extra_repr(self) -> str:
        d_out, d_in = self.base.weight.shape
        return f"in_features={d_in}, out_features={d_out}, rank={self.rank}, scale={self.scale:.4f}"

def inject_lora(model: nn.Module, target_names: Iterable[str], rank: int, alpha: float) -> int:
    target_set = set(target_names)
    replaced = 0
    for parent_name, parent_module in list(model.named_modules()):
        for attr_name, child in list(parent_module.named_children()):
            if attr_name not in target_set:
                continue
            if not isinstance(child, nn.Linear):
                continue
            lora_layer = LoRALinear(child, rank, alpha)
            setattr(parent_module, attr_name, lora_layer)
            replaced += 1
            logger.debug(
                "LoRA injected: %s.%s  (%d→%d, rank=%d)",
                parent_name, attr_name, child.in_features, child.out_features, rank,
            )
    return replaced

def freeze_base_parameters(model: nn.Module) -> int:
    frozen = 0
    for name, param in model.named_parameters():
        if "lora_A" not in name and "lora_B" not in name:
            param.requires_grad_(False)
            frozen += 1
    return frozen

def lora_named_parameters(model: nn.Module):
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            yield name, param

def lora_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for _, p in lora_named_parameters(model))

def lora_payload_bytes(model: nn.Module) -> int:
    return lora_parameter_count(model) * 4

def get_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: param.detach().cpu() for name, param in lora_named_parameters(model)}

def set_lora_state_dict(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    model_params = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state_dict.items():
            if name not in model_params:
                logger.warning("set_lora_state_dict: key %r not in model; skipping.", name)
                continue
            model_params[name].copy_(value)

def build_key_order(model: nn.Module) -> list[str]:
    return sorted(name for name, _ in lora_named_parameters(model))

def get_lora_parameters(model: nn.Module, key_order: list[str]) -> list[np.ndarray]:
    sd = get_lora_state_dict(model)
    missing = set(key_order) - sd.keys()
    if missing:
        raise RuntimeError(f"LoRA keys missing from model: {missing}")
    return [sd[k].numpy() for k in key_order]

def set_lora_parameters(model: nn.Module, params: list[np.ndarray], key_order: list[str]) -> None:
    if len(params) != len(key_order):
        raise ValueError(f"Parameter count mismatch: expected {len(key_order)}, got {len(params)}")
    sd = {k: torch.from_numpy(arr.copy()) for k, arr in zip(key_order, params)}
    set_lora_state_dict(model, sd)

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

import numpy as np
import torch

logger = logging.getLogger(__name__)

def save_lora_checkpoint(
    matcher,
    adapter_arrays: List[np.ndarray],
    round_id: int,
    metrics: Dict[str, Any],
    output_dir: str,
) -> str:
    if not getattr(matcher, "_lora_enabled", False):
        raise RuntimeError("save_lora_checkpoint called but LoRA is not enabled on the matcher.")

    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, f"lora_round_{round_id:03d}.pth")

    matcher.set_lora_parameters(adapter_arrays)

    from models.lora import get_lora_state_dict
    lora_sd = get_lora_state_dict(matcher.model)

    torch.save({
        "round_id": round_id,
        "lora_state_dict": lora_sd,
        "key_order": matcher._lora_key_order,
        "metrics": metrics,
    }, checkpoint_path)

    payload_kb = sum(a.nbytes for a in adapter_arrays) / 1024
    logger.info("Saved LoRA checkpoint: %s  (round=%d, payload=%.1f KB)", checkpoint_path, round_id, payload_kb)
    return checkpoint_path

def append_round_metrics(metrics: Dict[str, Any], output_path: str) -> None:
    if os.path.exists(output_path):
        try:
            with open(output_path, "r") as f:
                all_metrics = json.load(f)
        except (json.JSONDecodeError, IOError):
            all_metrics = []
    else:
        all_metrics = []
    all_metrics.append(metrics)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

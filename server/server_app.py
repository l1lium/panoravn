from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Optional

import flwr as fl
import numpy as np
from flwr.server import ServerApp
from flwr.server.serverapp_components import ServerAppComponents

from models.model_factory import create_matcher
from server.model_registry import append_round_metrics, save_lora_checkpoint
from server.strategies import LoRAFedAvg

logger = logging.getLogger(__name__)

def _init_model(config: Dict[str, Any]):
    model_config = config.get("model", {})
    model = create_matcher(model_config)
    lora_cfg = model_config.get("lora", {})
    use_lora = lora_cfg.get("enabled", False)
    if use_lora:
        model.enable_lora(
            rank=int(lora_cfg.get("rank", 4)),
            alpha=float(lora_cfg.get("alpha", 8.0)),
            target_modules=lora_cfg.get("target_modules", None),
        )
    return model, use_lora

def _build_strategy(
    fl_config: Dict[str, Any],
    global_model,
    use_lora: bool,
    output_dir: str,
    num_rounds: int,
    fit_metrics_aggregation_fn: Optional[Callable] = None,
) -> LoRAFedAvg:
    training_cfg = fl_config.get("training", {
        "local_epochs": 1,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "batch_size": 4,
    })
    strategy_cfg = fl_config.get("strategy", {})
    metrics_path = os.path.join(output_dir, "federated_metrics.json")

    def _checkpoint_fn(server_round: int, adapter_arrays: List[np.ndarray]) -> None:
        if use_lora:
            save_lora_checkpoint(global_model, adapter_arrays, server_round, {}, output_dir)
        append_round_metrics({"round": server_round}, metrics_path)
        if server_round == num_rounds:
            npz_path = os.path.join(output_dir, "final_lora_weights.npz")
            np.savez(npz_path, **{f"w{i:04d}": p for i, p in enumerate(adapter_arrays)})
            logger.info("Exported final LoRA weights → %s", npz_path)

    extra = {"fit_metrics_aggregation_fn": fit_metrics_aggregation_fn} if fit_metrics_aggregation_fn else {}
    return LoRAFedAvg(
        checkpoint_fn=_checkpoint_fn,
        fraction_fit=strategy_cfg.get("fraction_fit", 1.0),
        fraction_evaluate=strategy_cfg.get("fraction_evaluate", 1.0),
        min_fit_clients=strategy_cfg.get("min_fit_clients", 2),
        min_evaluate_clients=strategy_cfg.get("min_evaluate_clients", 2),
        min_available_clients=strategy_cfg.get("min_available_clients", 2),
        on_fit_config_fn=lambda r: dict(**training_cfg, round=r),
        on_evaluate_config_fn=lambda r: {"server_round": r},
        **extra,
    )

def build_server_app(
    config: Dict[str, Any],
    *,
    fit_metrics_aggregation_fn: Optional[Callable] = None,
) -> ServerApp:
    num_rounds = config.get("num_rounds", 10)
    output_dir = config.get("output_dir", "./federated_output")
    os.makedirs(output_dir, exist_ok=True)

    global_model, use_lora = _init_model(config)
    strategy = _build_strategy(
        config, global_model, use_lora, output_dir, num_rounds, fit_metrics_aggregation_fn
    )

    def server_fn(context):
        return ServerAppComponents(
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
        )

    return ServerApp(server_fn=server_fn)

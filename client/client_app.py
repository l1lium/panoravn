from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from flwr.client import NumPyClient

from client.local_training import fit_local_model
from client.privacy import validate_client_payload, validate_model_parameters
from models.model_factory import create_matcher

logger = logging.getLogger(__name__)

class PanoramaMatcherClient(NumPyClient):
    def __init__(self, client_id: str, config: Dict[str, Any]):
        self.client_id = client_id
        self.config = config

        model_config = config.get("model", {})
        self.model = create_matcher(model_config)

        lora_cfg = model_config.get("lora", {})
        if lora_cfg.get("enabled", False):
            self.model.enable_lora(
                rank=int(lora_cfg.get("rank", 4)),
                alpha=float(lora_cfg.get("alpha", 8.0)),
                target_modules=lora_cfg.get("target_modules", None),
            )
            logger.info(
                "Client %s: LoRA enabled (rank=%d, alpha=%.1f).",
                client_id, lora_cfg.get("rank", 4), lora_cfg.get("alpha", 8.0),
            )

        self.train_loader = self._load_train_data()
        self.val_loader = self._load_val_data()

        logger.info("Client %s initialised with %s.", client_id, type(self.model).__name__)

    def _load_train_data(self):
        data_config = self.config.get("data", {})
        data_dir = data_config.get("data_dir", "./data")
        csv_file = data_config.get("train_csv", "train_pairs.csv")

        try:
            from client.dataset import load_pairs_from_csv, create_federated_dataloader
            data_pairs = load_pairs_from_csv(data_dir, csv_file)
            batch_size = self.config.get("training", {}).get("batch_size", 4)
            dataloader = create_federated_dataloader(data_pairs, batch_size=batch_size)
            logger.info(f"Client {self.client_id} loaded {len(data_pairs)} training pairs")
            return dataloader
        except (FileNotFoundError, KeyError) as e:
            logger.warning(f"Client {self.client_id} could not load training data: {e}")
            return None

    def _load_val_data(self):
        data_config = self.config.get("data", {})
        data_dir = data_config.get("data_dir", "./data")
        csv_file = data_config.get("val_csv", "val_pairs.csv")

        try:
            from client.dataset import load_pairs_from_csv, create_federated_dataloader
            data_pairs = load_pairs_from_csv(data_dir, csv_file)
            batch_size = self.config.get("training", {}).get("batch_size", 4)
            dataloader = create_federated_dataloader(data_pairs, batch_size=batch_size, shuffle=False)
            logger.info(f"Client {self.client_id} loaded {len(data_pairs)} validation pairs")
            return dataloader
        except (FileNotFoundError, KeyError) as e:
            logger.warning(f"Client {self.client_id} could not load validation data: {e}")
            return None

    def get_parameters(self, config: Dict[str, Any] = None) -> List[np.ndarray]:
        if getattr(self.model, "_lora_enabled", False) is True:
            parameters = self.model.get_lora_parameters()
            logger.info(
                "Client %s: sending %d LoRA adapter arrays (%.1f KB).",
                self.client_id,
                len(parameters),
                sum(a.nbytes for a in parameters) / 1024,
            )
        else:
            parameters = [
                p.detach().cpu().numpy()
                for p in self.model.model.parameters()
            ]
            logger.info(
                "Client %s: sending %d full-model arrays.",
                self.client_id, len(parameters),
            )

        validate_model_parameters(parameters)
        return parameters

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        validate_model_parameters(parameters)

        if getattr(self.model, "_lora_enabled", False) is True:
            self.model.set_lora_parameters(parameters)
            logger.info(
                "Client %s: loaded %d LoRA adapter arrays from server.",
                self.client_id, len(parameters),
            )
        else:
            param_iter = iter(parameters)
            for param in self.model.model.parameters():
                param.copy_(torch.from_numpy(next(param_iter)).to(param.device))
            logger.info(
                "Client %s: loaded full-model parameters from server.",
                self.client_id,
            )

        logger.info(f"Client {self.client_id} received and set {len(parameters)} parameter arrays")

    def fit(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        validate_client_payload(parameters, "fit_parameters")
        validate_client_payload(config, "fit_config")

        self.set_parameters(parameters)

        if self.train_loader is not None and self.val_loader is not None:
            logger.info(f"Client {self.client_id} starting local training")

            trained_matcher, metrics = fit_local_model(
                self.model,
                self.train_loader,
                self.val_loader,
                config
            )

            self.model = trained_matcher
            updated_params = self.get_parameters()
            num_samples = len(self.train_loader.dataset)
            return updated_params, num_samples, metrics
        else:
            logger.warning(f"Client {self.client_id} has no local data, returning unchanged parameters")
            return parameters, 0, {"loss": 0.0, "no_data": True}

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[float, int, Dict[str, Any]]:
        validate_client_payload(parameters, "evaluate_parameters")
        validate_client_payload(config, "evaluate_config")

        self.set_parameters(parameters)

        if self.val_loader is not None:
            from client.local_training import evaluate_local_model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            metrics = evaluate_local_model(self.model, self.val_loader, device)
            num_samples = len(self.val_loader.dataset)
            loss = metrics.get("loss", 0.0)

            logger.info(f"Client {self.client_id} evaluation: loss={loss:.4f}")
            return loss, num_samples, metrics
        else:
            logger.warning(f"Client {self.client_id} has no validation data")
            return 0.0, 0, {"no_data": True}

def create_client(client_id: str, config: Dict[str, Any]) -> PanoramaMatcherClient:
    return PanoramaMatcherClient(client_id, config)

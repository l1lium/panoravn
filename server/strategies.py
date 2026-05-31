from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from flwr.common import FitRes, Parameters, Scalar, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

logger = logging.getLogger(__name__)

class LoRAFedAvg(FedAvg):
    def __init__(
        self,
        checkpoint_fn: Callable[[int, List[np.ndarray]], None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._checkpoint_fn = checkpoint_fn

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        aggregated_params, metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_params is not None:
            adapter_arrays = parameters_to_ndarrays(aggregated_params)
            total_kb = sum(a.nbytes for a in adapter_arrays) / 1024
            logger.info(
                "Round %d: aggregated %d adapter arrays (%.1f KB) from %d clients.",
                server_round, len(adapter_arrays), total_kb, len(results),
            )
            if self._checkpoint_fn is not None:
                try:
                    self._checkpoint_fn(server_round, adapter_arrays)
                except Exception as exc:
                    logger.error("Checkpoint save failed at round %d: %s", server_round, exc)

        return aggregated_params, metrics

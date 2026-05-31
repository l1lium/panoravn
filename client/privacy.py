from typing import Any
import numpy as np

try:
    from flwr.common.record.configrecord import ConfigRecord as _FlwrConfigRecord
except ImportError:
    _FlwrConfigRecord = None

DISALLOWED_TYPES = (str, bytes, bytearray)

def validate_client_payload(payload: Any, context: str = "payload") -> None:
    if isinstance(payload, DISALLOWED_TYPES):
        raise ValueError(
            f"Privacy violation in {context}: disallowed type {type(payload).__name__}"
        )

    if isinstance(payload, dict):
        for key, value in payload.items():
            validate_client_payload(value, f"{context}.{key}")
    elif isinstance(payload, (list, tuple, set)):
        for i, item in enumerate(payload):
            validate_client_payload(item, f"{context}[{i}]")
    elif isinstance(payload, (int, float, bool, type(None))):
        return
    else:
        if isinstance(payload, np.ndarray):
            return
        if _FlwrConfigRecord is not None and isinstance(payload, _FlwrConfigRecord):
            return
        raise ValueError(
            f"Privacy violation in {context}: unsupported payload type {type(payload)}"
        )

def validate_model_parameters(parameters: Any) -> None:
    if not isinstance(parameters, list):
        raise ValueError("Model parameters must be a list of numpy arrays")

    for i, param in enumerate(parameters):
        if not isinstance(param, np.ndarray):
            raise ValueError(f"Parameter {i} must be a numpy array, got {type(param)}")
        if param.dtype.kind not in 'fiu':
            raise ValueError(f"Parameter {i} has unsupported dtype {param.dtype}")

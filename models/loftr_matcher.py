from __future__ import annotations

import logging

import numpy as np
import torch

from models.base_matcher import MatcherInterface

logger = logging.getLogger(__name__)

class LoFTRMatcher(MatcherInterface):
    def __init__(self, config: dict):
        self.config = config
        self.device = self._get_device()
        self.confidence_threshold = float(config.get("confidence_threshold", 0.1))
        self._conf_matrix: torch.Tensor | None = None
        self._expec_f:     torch.Tensor | None = None
        self.model = self._load_model(config.get("checkpoint_path"))
        self._lora_enabled: bool = False
        self._lora_key_order: list[str] = []

    def _get_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _load_model(self, checkpoint_path=None):
        try:
            from kornia.feature import LoFTR
        except ImportError:
            raise ImportError("LoFTR requires kornia. Install with: pip install kornia")

        if checkpoint_path:
            model = LoFTR(None)
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint)
        else:
            model = LoFTR("outdoor")

        model = model.to(self.device)
        model.eval()

        def _hook_coarse(module, inputs, output):
            if len(inputs) > 2 and isinstance(inputs[2], dict):
                cm = inputs[2].get("conf_matrix")
                if cm is not None:
                    self._conf_matrix = cm

        def _hook_fine(module, inputs, output):
            if len(inputs) > 2 and isinstance(inputs[2], dict):
                ef = inputs[2].get("expec_f")
                if ef is not None and ef.numel() > 0:
                    self._expec_f = ef

        model.coarse_matching.register_forward_hook(_hook_coarse)
        model.fine_matching.register_forward_hook(_hook_fine)

        return model

    def match(self, image_a: np.ndarray, image_b: np.ndarray):
        if image_a is None or image_b is None:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), []

        try:
            tensor_a = self._image_to_tensor(image_a)
            tensor_b = self._image_to_tensor(image_b)
            with torch.no_grad():
                correspondences = self.model({"image0": tensor_a, "image1": tensor_b})
            return self._extract_matches(correspondences)
        except RuntimeError as e:
            raise RuntimeError(f"LoFTR matching failed: {e}")

    def _image_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        if image.ndim == 3 and image.shape[2] == 1:
            image = image[:, :, 0]
        tensor = torch.from_numpy(image).float() / 255.0
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.ndim == 3:
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        else:
            raise ValueError(f"Invalid image shape: {image.shape}")
        return tensor.to(self.device)

    def _extract_matches(self, correspondences: dict):
        if "keypoints0" not in correspondences or "keypoints1" not in correspondences:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), []

        kpts_a = correspondences["keypoints0"].cpu().numpy()
        kpts_b = correspondences["keypoints1"].cpu().numpy()
        conf = correspondences.get("confidence", np.ones(len(kpts_a)))

        if isinstance(conf, torch.Tensor):
            conf = conf.cpu().numpy()

        if len(kpts_a) == 0 or len(kpts_b) == 0:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), []

        mask = conf >= self.confidence_threshold
        points_a = kpts_a[mask].astype(np.float32)
        points_b = kpts_b[mask].astype(np.float32)
        confidences = conf[mask]

        matches = [{"confidence": float(c), "distance": 1.0 - float(c)} for c in confidences]
        return points_a, points_b, matches

    def enable_lora(self, rank: int = 4, alpha: float = 8.0, target_modules: list[str] | None = None) -> None:
        if self._lora_enabled:
            logger.warning("enable_lora called a second time — skipped.")
            return

        if target_modules is None:
            target_modules = ["q_proj", "v_proj"]

        from models.lora import build_key_order, freeze_base_parameters, inject_lora, lora_payload_bytes

        n_frozen = freeze_base_parameters(self.model)
        n_injected = inject_lora(self.model, target_modules, rank, alpha)
        if n_injected == 0:
            raise RuntimeError(
                f"LoRA injection found no target layers for {target_modules!r}. "
                "Verify that target_modules matches the attribute names inside "
                "kornia's LoFTREncoderLayer (e.g. 'q_proj', 'v_proj')."
            )

        self._lora_key_order = build_key_order(self.model)
        self._lora_enabled = True

        payload_kb = lora_payload_bytes(self.model) / 1024
        logger.info(
            "LoRA enabled on LoFTR: rank=%d  alpha=%.1f  "
            "layers_injected=%d  payload=%.1f KB  base_params_frozen=%d",
            rank, alpha, n_injected, payload_kb, n_frozen,
        )

    def get_lora_parameters(self) -> list[np.ndarray]:
        if not self._lora_enabled:
            raise RuntimeError("get_lora_parameters() called but LoRA is not enabled.")
        from models.lora import get_lora_parameters
        return get_lora_parameters(self.model, self._lora_key_order)

    def set_lora_parameters(self, params: list[np.ndarray]) -> None:
        if not self._lora_enabled:
            raise RuntimeError("set_lora_parameters() called but LoRA is not enabled.")
        from models.lora import set_lora_parameters
        set_lora_parameters(self.model, params, self._lora_key_order)

    def compute_loss(self, outputs: dict, batch: dict) -> torch.Tensor:
        loss_parts = []

        cm = self._conf_matrix
        self._conf_matrix = None
        if cm is not None:
            peak_row = cm.max(dim=2)[0]
            peak_col = cm.max(dim=1)[0]
            loss_parts.append(-(peak_row.mean() + peak_col.mean()) * 0.5)

        ef = self._expec_f
        self._expec_f = None
        if ef is not None and ef.numel() > 0:
            loss_parts.append(ef[:, 2].mean())

        if loss_parts:
            return sum(loss_parts) / len(loss_parts)

        logger.warning("No differentiable signal from LoFTR forward; using scalar fallback.")
        return torch.tensor(0.1, dtype=torch.float32, device=self.device, requires_grad=True)

    def train(self):
        self.model.train()
        self.model.coarse_matching.eval()
        self.model.fine_matching.eval()

    def eval(self):
        self.model.eval()

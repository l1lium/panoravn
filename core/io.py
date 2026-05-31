import os
from datetime import datetime
from typing import Any

import cv2
import yaml

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def create_run_dir(base_output_dir: str, experiment_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_output_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def save_config(config: dict, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)

def save_image(path: str, image: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if image.dtype != "uint8":
        image = image.astype("uint8")
    cv2.imwrite(path, image)

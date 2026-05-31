import cv2
import numpy as np

def load_image(path: str):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Unable to load image: {path}")
    return image

def _resize_image(image: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(1.0, max_width / w if w else 1.0, max_height / h if h else 1.0)
    if scale == 1.0:
        return image
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

def preprocess_image(image: np.ndarray, config: dict) -> np.ndarray:
    image = _resize_image(image, config["image"]["max_width"], config["image"]["max_height"])
    if config["image"].get("to_gray", False):
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def preprocess_pair(image_a: np.ndarray, image_b: np.ndarray, config: dict):
    return preprocess_image(image_a, config), preprocess_image(image_b, config)

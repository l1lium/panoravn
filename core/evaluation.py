import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2

def compute_ssim(image_a: np.ndarray, image_b: np.ndarray) -> float:
    if image_a.ndim == 3 and image_a.shape[2] == 3:
        image_a_gray = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
        image_b_gray = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)
    else:
        image_a_gray = image_a
        image_b_gray = image_b
    return float(ssim(image_a_gray, image_b_gray, data_range=255))

def compute_psnr(image_a: np.ndarray, image_b: np.ndarray) -> float:
    return float(cv2.PSNR(image_a, image_b))

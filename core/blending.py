import cv2
import numpy as np

def alpha_blend(image_a: np.ndarray, image_b: np.ndarray, mask_a: np.ndarray = None, mask_b: np.ndarray = None, alpha: float = 0.5) -> np.ndarray:
    if mask_a is None:
        mask_a = np.any(image_a != 0, axis=2).astype(np.uint8)
    if mask_b is None:
        mask_b = np.any(image_b != 0, axis=2).astype(np.uint8)

    alpha = float(alpha)
    pano = image_a.copy()
    overlap = (mask_a > 0) & (mask_b > 0)
    only_b = (mask_b > 0) & ~overlap

    if np.any(overlap):
        pano[overlap] = cv2.addWeighted(image_a[overlap], alpha, image_b[overlap], 1.0 - alpha, 0)
    pano[only_b] = image_b[only_b]
    return pano

def multiband_blend(image_a: np.ndarray, image_b: np.ndarray, mask_a: np.ndarray = None, mask_b: np.ndarray = None, levels: int = 4) -> np.ndarray:
    if mask_a is None:
        mask_a = np.any(image_a != 0, axis=2).astype(np.uint8)
    if mask_b is None:
        mask_b = np.any(image_b != 0, axis=2).astype(np.uint8)

    mask = np.zeros_like(mask_a, dtype=np.float32)
    overlap = (mask_a > 0) & (mask_b > 0)
    mask[overlap] = 0.5
    mask[(mask_a > 0) & ~overlap] = 1.0
    mask[(mask_b > 0) & ~overlap] = 0.0
    mask = cv2.merge([mask, mask, mask])

    gp_a = [image_a.astype(np.float32)]
    gp_b = [image_b.astype(np.float32)]
    gp_mask = [mask.astype(np.float32)]
    for _ in range(levels):
        gp_a.append(cv2.pyrDown(gp_a[-1]))
        gp_b.append(cv2.pyrDown(gp_b[-1]))
        gp_mask.append(cv2.pyrDown(gp_mask[-1]))

    lp_a = [gp_a[-1]]
    lp_b = [gp_b[-1]]
    for i in range(levels, 0, -1):
        size_a = (gp_a[i - 1].shape[1], gp_a[i - 1].shape[0])
        la = gp_a[i - 1] - cv2.pyrUp(gp_a[i], dstsize=size_a)
        lb = gp_b[i - 1] - cv2.pyrUp(gp_b[i], dstsize=size_a)
        lp_a.append(la)
        lp_b.append(lb)

    blended = lp_a[0] * gp_mask[0] + lp_b[0] * (1 - gp_mask[0])
    for i in range(1, len(lp_a)):
        blended = cv2.pyrUp(blended, dstsize=(lp_a[i].shape[1], lp_a[i].shape[0]))
        blended = blended + lp_a[i] * gp_mask[i] + lp_b[i] * (1 - gp_mask[i])

    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return blended

def blend_panorama(warped_images, config: dict):
    if len(warped_images) != 2:
        raise ValueError("blend_panorama currently supports two warped image inputs")

    image_a, image_b = warped_images
    mask_a = np.any(image_a != 0, axis=2).astype(np.uint8)
    mask_b = np.any(image_b != 0, axis=2).astype(np.uint8)
    method = config.get("blending", {}).get("method", "alpha")
    alpha = float(config.get("blending", {}).get("alpha", 0.5))

    if method == "alpha":
        return alpha_blend(image_a, image_b, mask_a, mask_b, alpha)
    if method == "multiband":
        return multiband_blend(image_a, image_b, mask_a, mask_b)
    return alpha_blend(image_a, image_b, mask_a, mask_b, alpha)

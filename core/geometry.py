from typing import Optional, Tuple

import cv2
import numpy as np

def estimate_homography(points_a: np.ndarray, points_b: np.ndarray, ransac_threshold: float = 5.0) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if points_a.shape[0] < 4 or points_b.shape[0] < 4:
        return None, None

    homography, mask = cv2.findHomography(points_a, points_b, cv2.RANSAC, ransac_threshold)
    if homography is None:
        return None, None
    return homography, mask

def validate_homography(
    homography: np.ndarray,
    inlier_mask: np.ndarray,
    min_inliers: int,
    min_inlier_ratio: float,
    max_condition_number: float = 20.0,
) -> bool:
    if homography is None or inlier_mask is None:
        return False

    inliers = int(np.sum(inlier_mask))
    total = int(len(inlier_mask))
    if inliers < min_inliers:
        return False
    if total > 0 and (inliers / total) < min_inlier_ratio:
        return False
    if not np.isfinite(homography).all():
        return False

    h22 = homography[2, 2]
    if abs(h22) < 1e-10:
        return False
    H = homography / h22

    if np.linalg.det(H) <= 0:
        return False

    sv = np.linalg.svd(H[:2, :2], compute_uv=False)
    if sv.min() < 1e-10 or sv.max() / sv.min() > max_condition_number:
        return False

    return True

def compute_reprojection_error(points_a: np.ndarray, points_b: np.ndarray, homography: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    if homography is None or points_a.size == 0 or points_b.size == 0:
        return float("inf")
    projected = cv2.perspectiveTransform(points_a.reshape(-1, 1, 2), homography).reshape(-1, 2)
    error = np.linalg.norm(projected - points_b, axis=1)
    if mask is not None:
        mask = mask.reshape(-1).astype(bool)
        if mask.sum() == 0:
            return float("inf")
        error = error[mask]
    return float(np.mean(error))

_MAX_CANVAS_PIXELS = 10_000_000

def warp_images(image_a: np.ndarray, image_b: np.ndarray, homography: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h_a, w_a = image_a.shape[:2]
    h_b, w_b = image_b.shape[:2]

    corners_b = np.array([[0, 0], [w_b, 0], [w_b, h_b], [0, h_b]], dtype=float).reshape(-1, 1, 2)
    warped_corners_b = cv2.perspectiveTransform(corners_b, homography)

    corners_a = np.array([[0, 0], [w_a, 0], [w_a, h_a], [0, h_a]], dtype=float).reshape(-1, 1, 2)
    all_corners = np.concatenate([warped_corners_b, corners_a], axis=0)

    [xmin, ymin] = np.floor(all_corners.min(axis=0).ravel()).astype(int)
    [xmax, ymax] = np.ceil(all_corners.max(axis=0).ravel()).astype(int)

    canvas_w, canvas_h = xmax - xmin, ymax - ymin
    if canvas_w * canvas_h > _MAX_CANVAS_PIXELS:
        raise ValueError(
            f"warp_images: canvas {canvas_w}×{canvas_h} = {canvas_w * canvas_h // 1_000_000} Mpx "
            f"exceeds {_MAX_CANVAS_PIXELS // 1_000_000} Mpx limit — homography likely degenerate."
        )

    translation = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], dtype=float)
    output_shape = (canvas_w, canvas_h)

    warped_a = cv2.warpPerspective(image_a, translation, output_shape)
    warped_b = cv2.warpPerspective(image_b, translation.dot(homography), output_shape)

    mask_a = np.zeros((output_shape[1], output_shape[0]), dtype=np.uint8)
    cv2.warpPerspective(np.ones((h_a, w_a), dtype=np.uint8) * 255, translation, output_shape, dst=mask_a)
    mask_b = np.zeros((output_shape[1], output_shape[0]), dtype=np.uint8)
    cv2.warpPerspective(np.ones((h_b, w_b), dtype=np.uint8) * 255, translation.dot(homography), output_shape, dst=mask_b)

    return warped_a, warped_b, mask_a, mask_b

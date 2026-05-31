import numpy as np
from core.geometry import estimate_homography, validate_homography, compute_reprojection_error, warp_images

def test_estimate_homography_identity():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float)
    homography, mask = estimate_homography(points, points, ransac_threshold=3.0)
    assert homography is not None
    assert mask is not None
    assert validate_homography(homography, mask, min_inliers=4, min_inlier_ratio=1.0)
    error = compute_reprojection_error(points, points, homography, mask)
    assert error < 1e-6

def test_warp_images_translation():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[5:15, 5:15] = 255
    homography = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 5.0], [0.0, 0.0, 1.0]], dtype=float)
    warped_a, warped_b, mask_a, mask_b = warp_images(image, image, homography)
    assert warped_a.shape == warped_b.shape
    assert mask_a.sum() > 0
    assert mask_b.sum() > 0

from typing import Dict

from core.blending import blend_panorama
from core.evaluation import compute_psnr, compute_ssim
from core.geometry import compute_reprojection_error, estimate_homography, validate_homography, warp_images
from client.preprocessing import preprocess_pair

def stitch_pair(image_a, image_b, matcher, config: Dict) -> Dict:
    image_a_proc, image_b_proc = preprocess_pair(image_a, image_b, config)
    points_a, points_b, matches = matcher.match(image_a_proc, image_b_proc)

    homography, mask = estimate_homography(points_a, points_b, config["geometry"]["ransac_threshold"])
    valid = validate_homography(
        homography, mask,
        config["geometry"]["min_inliers"],
        config["geometry"]["min_inlier_ratio"],
        max_condition_number=float(config["geometry"].get("max_condition_number", 20.0)),
    )
    reprojection = float("inf")
    panorama = None
    if valid:
        reprojection = compute_reprojection_error(points_a, points_b, homography, mask)
        if reprojection > config["geometry"]["max_reprojection_error"]:
            valid = False

    metrics = {
        "num_matches": len(matches),
        "num_inliers": int(mask.sum()) if mask is not None else 0,
        "reprojection_error": reprojection,
        "homography_valid": bool(valid),
    }

    if valid:
        warped_a, warped_b, mask_a, mask_b = warp_images(image_a, image_b, homography)
        panorama = blend_panorama([warped_a, warped_b], config)
        if config.get("evaluation", {}).get("compute_ssim", False):
            metrics["ssim"] = compute_ssim(warped_a, warped_b)
        if config.get("evaluation", {}).get("compute_psnr", False):
            metrics["psnr"] = compute_psnr(warped_a, warped_b)
    else:
        panorama = image_a.copy()
    metrics["panorama"] = panorama
    return metrics

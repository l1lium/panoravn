from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from bridge.ingestor import FrameRecord

def ground_footprint(
    alt_m: float,
    fov_deg: float,
    imu_pitch_rad: float,
    image_wh: tuple[int, int],
) -> tuple[float, float]:
    if alt_m <= 0:
        return 0.0, 0.0

    img_w, img_h = image_wh
    aspect = img_h / img_w if img_w else 1.0
    fov_h_rad = math.radians(fov_deg)
    fov_v_rad = fov_h_rad * aspect

    depression = math.pi / 2.0 + imu_pitch_rad

    cos_dep = max(math.cos(depression), 1e-6)
    slant_centre = alt_m / cos_dep

    width_m = 2.0 * slant_centre * math.tan(fov_h_rad / 2.0)

    angle_far  = depression + fov_v_rad / 2.0
    angle_near = depression - fov_v_rad / 2.0
    dist_far   = alt_m / max(math.cos(angle_far),  1e-6)
    dist_near  = alt_m / max(math.cos(angle_near), 1e-6) if angle_near > 0 else 0.0
    height_m   = dist_far - dist_near

    return width_m, height_m

def overlap_ratio(a: "FrameRecord", b: "FrameRecord") -> float:
    w_a, h_a = ground_footprint(a.gps_y, a.fov_deg, a.imu_pitch, a.image_wh)
    w_b, h_b = ground_footprint(b.gps_y, b.fov_deg, b.imu_pitch, b.image_wh)

    if w_a <= 0 or h_a <= 0:
        return 0.0

    ax, az = a.gps_x, a.gps_z
    bx, bz = b.gps_x, b.gps_z

    ha_x, ha_z = w_a / 2.0, h_a / 2.0
    hb_x, hb_z = w_b / 2.0, h_b / 2.0

    inter_x = max(0.0, min(ax + ha_x, bx + hb_x) - max(ax - ha_x, bx - hb_x))
    inter_z = max(0.0, min(az + ha_z, bz + hb_z) - max(az - ha_z, bz - hb_z))

    inter_area = inter_x * inter_z
    area_a = w_a * h_a
    return inter_area / area_a if area_a > 0 else 0.0

def gps_to_homography_prior(
    a: "FrameRecord",
    b: "FrameRecord",
) -> np.ndarray | None:
    if overlap_ratio(a, b) < 0.1:
        return None

    w_m, h_m = ground_footprint(a.gps_y, a.fov_deg, a.imu_pitch, a.image_wh)
    if w_m <= 0 or h_m <= 0:
        return None

    img_w, img_h = a.image_wh

    mpp_x = w_m / img_w
    mpp_z = h_m / img_h

    dx_world = b.gps_x - a.gps_x
    dz_world = b.gps_z - a.gps_z

    tx = dx_world / mpp_x if mpp_x > 0 else 0.0
    ty = dz_world / mpp_z if mpp_z > 0 else 0.0

    H = np.eye(3, dtype=np.float64)
    H[0, 2] = tx
    H[1, 2] = ty
    return H

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2.0 * R * math.asin(math.sqrt(a))

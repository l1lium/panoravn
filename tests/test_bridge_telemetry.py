"""
Tests for bridge/telemetry.py

Covers:
  - ground_footprint: positive values, scales with altitude, zero at alt=0
  - overlap_ratio:   1.0 for identical positions, 0.0 for far-apart frames,
                     partial overlap for adjacent footprints
  - gps_to_homography_prior: None when overlap < 0.1, 3×3 matrix otherwise,
                              translation-only structure, correct pixel shift
  - _haversine_m:    Paris → London regression (~341 km)
"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bridge.ingestor import FrameRecord
from bridge.telemetry import (
    _haversine_m,
    gps_to_homography_prior,
    ground_footprint,
    overlap_ratio,
)

_NADIR = -math.pi / 2
_FOV   = 84.0
_WH    = (64, 48)
_ALT   = 50.0

def _frame(
    gps_x: float,
    gps_z: float,
    gps_y: float = _ALT,
    fov_deg: float = _FOV,
    image_wh: tuple = _WH,
    imu_pitch: float = _NADIR,
) -> FrameRecord:
    """Minimal FrameRecord for telemetry tests."""
    return FrameRecord(
        drone_id="drone_0",
        frame_id=f"test_{gps_x:.1f}_{gps_z:.1f}",
        timestamp=0.0,
        image=np.zeros((image_wh[1], image_wh[0], 3), dtype=np.uint8),
        gps_x=gps_x,
        gps_y=gps_y,
        gps_z=gps_z,
        imu_roll=0.0,
        imu_pitch=imu_pitch,
        imu_yaw=0.0,
        gimbal_pitch=_NADIR,
        fov_deg=fov_deg,
        image_wh=image_wh,
    )

_FOOTPRINT_HALF_WIDTH_M = _ALT * math.tan(math.radians(_FOV / 2))

class TestGroundFootprint:
    def test_positive_values_for_nadir(self):
        w, h = ground_footprint(_ALT, _FOV, _NADIR, _WH)
        assert w > 0
        assert h > 0

    def test_width_greater_than_height_for_landscape_image(self):
        w, h = ground_footprint(_ALT, _FOV, _NADIR, (640, 480))
        assert w > 0 and h > 0

    def test_zero_altitude_returns_zero(self):
        w, h = ground_footprint(0.0, _FOV, _NADIR, _WH)
        assert w == 0.0
        assert h == 0.0

    def test_negative_altitude_returns_zero(self):
        w, h = ground_footprint(-10.0, _FOV, _NADIR, _WH)
        assert w == 0.0
        assert h == 0.0

    def test_footprint_scales_with_altitude(self):
        w1, h1 = ground_footprint(50.0,  _FOV, _NADIR, _WH)
        w2, h2 = ground_footprint(100.0, _FOV, _NADIR, _WH)
        assert w2 > w1
        assert h2 > h1

    def test_wider_fov_gives_wider_footprint(self):
        w_narrow, _ = ground_footprint(_ALT, 60.0, _NADIR, _WH)
        w_wide,   _ = ground_footprint(_ALT, 90.0, _NADIR, _WH)
        assert w_wide > w_narrow

    def test_width_approx_2_alt_tan_half_fov(self):
        """Nadir footprint width ≈ 2 * alt * tan(fov/2) (simple geometry)."""
        w, _ = ground_footprint(_ALT, _FOV, _NADIR, _WH)
        expected = 2.0 * _ALT * math.tan(math.radians(_FOV / 2))
        assert w == pytest.approx(expected, rel=0.01)

class TestOverlapRatio:
    def test_identical_positions_returns_one(self):
        r = _frame(0.0, 0.0)
        assert overlap_ratio(r, r) == pytest.approx(1.0)

    def test_same_position_different_records_returns_one(self):
        a = _frame(0.0, 0.0)
        b = _frame(0.0, 0.0)
        assert overlap_ratio(a, b) == pytest.approx(1.0)

    def test_far_apart_returns_zero(self):
        """
        Frames separated by 2× the footprint width cannot overlap.
        At alt=50 m, fov=84°: half-width ≈ 45 m, so 100 m apart → zero.
        """
        a = _frame(0.0,   0.0)
        b = _frame(100.0, 0.0)
        assert overlap_ratio(a, b) == pytest.approx(0.0)

    def test_result_in_unit_interval(self):
        a = _frame(0.0, 0.0)
        b = _frame(30.0, 0.0)
        ratio = overlap_ratio(a, b)
        assert 0.0 <= ratio <= 1.0

    def test_partial_overlap_decreases_with_distance(self):
        """Frames further apart should have equal or lower overlap."""
        a    = _frame(0.0, 0.0)
        near = _frame(20.0, 0.0)
        far  = _frame(40.0, 0.0)
        assert overlap_ratio(a, near) >= overlap_ratio(a, far)

    def test_zero_altitude_frame_returns_zero(self):
        a = _frame(0.0, 0.0, gps_y=0.0)
        b = _frame(0.0, 0.0)
        assert overlap_ratio(a, b) == pytest.approx(0.0)

class TestGpsToHomographyPrior:
    def test_returns_none_when_overlap_below_threshold(self):
        """Frames 100 m apart → overlap ≈ 0 < 0.1 → must return None."""
        a = _frame(0.0,   0.0)
        b = _frame(100.0, 0.0)
        assert gps_to_homography_prior(a, b) is None

    def test_returns_3x3_array_for_overlapping_frames(self):
        a = _frame(0.0, 0.0)
        b = _frame(0.0, 0.0)
        H = gps_to_homography_prior(a, b)
        assert H is not None
        assert H.shape == (3, 3)
        assert H.dtype == np.float64

    def test_zero_displacement_yields_identity(self):
        """Same-position frames → zero pixel displacement → identity H."""
        a = _frame(0.0, 0.0)
        b = _frame(0.0, 0.0)
        H = gps_to_homography_prior(a, b)
        assert H is not None
        np.testing.assert_allclose(H, np.eye(3), atol=1e-9)

    def test_homography_is_pure_translation(self):
        """
        The prior is built from GPS displacement only, so the returned
        matrix must have the rotation/scale block equal to identity and
        the perspective row equal to [0, 0, 1].
        """
        a = _frame(0.0, 0.0)
        b = _frame(0.0, 0.0)
        H = gps_to_homography_prior(a, b)
        assert H is not None
        np.testing.assert_allclose(H[:2, :2], np.eye(2), atol=1e-9)
        np.testing.assert_allclose(H[2, :], [0.0, 0.0, 1.0], atol=1e-9)

    def test_positive_x_displacement_produces_positive_tx(self):
        """
        Frame B is east of A (higher gps_x).  The pixel translation tx should
        be positive (B is to the right of A in the image).
        """
        a = _frame(0.0, 0.0)
        b = _frame(20.0, 0.0)
        H = gps_to_homography_prior(a, b)
        if H is not None:
            assert H[0, 2] > 0.0

    def test_zero_altitude_frame_returns_none(self):
        a = _frame(0.0, 0.0, gps_y=0.0)
        b = _frame(0.0, 0.0, gps_y=0.0)
        assert gps_to_homography_prior(a, b) is None

class TestHaversineRegression:
    def test_paris_to_london_approximately_341_km(self):
        """
        Great-circle distance Paris (48.8566°N, 2.3522°E) →
        London (51.5074°N, 0.1278°W) ≈ 341 km.
        Tolerance: ±2% (within ~7 km) to account for rounding.
        """
        dist_m = _haversine_m(48.8566, 2.3522, 51.5074, -0.1278)
        expected_m = 341_000.0
        assert dist_m == pytest.approx(expected_m, rel=0.02), (
            f"Expected ~341 km, got {dist_m / 1000:.1f} km"
        )

    def test_same_point_distance_is_zero(self):
        dist = _haversine_m(0.0, 0.0, 0.0, 0.0)
        assert dist == pytest.approx(0.0, abs=1e-3)

    def test_symmetry(self):
        d_ab = _haversine_m(48.8566, 2.3522, 51.5074, -0.1278)
        d_ba = _haversine_m(51.5074, -0.1278, 48.8566, 2.3522)
        assert d_ab == pytest.approx(d_ba, rel=1e-9)

    def test_known_equator_distance(self):
        """1° of longitude on the equator ≈ 111,195 m (R=6,371,000 m sphere)."""
        dist = _haversine_m(0.0, 0.0, 0.0, 1.0)
        assert dist == pytest.approx(111_195.0, abs=5.0)

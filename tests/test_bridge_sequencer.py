"""
Tests for bridge/sequencer.py

Covers:
  - xz_distance_m: zero for identical coords, correct Euclidean value
  - FrameSequencer.add_frame: out-of-order insertion keeps timestamp order
  - FrameSequencer.get_pairs: spatial distance filter (max_pair_distance_m)
  - FrameSequencer.get_pairs: temporal gap filter (max_pair_time_gap_sec)
  - FrameSequencer.get_pairs: min_frames_per_batch threshold
  - FrameSequencer.get_pairs(flush=True): emits all pending pairs
  - FrameSequencer.get_pairs(flush=True): no pair emitted twice
  - Frame appearance constraint: each frame in at most two pairs
"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bridge.ingestor import FrameRecord
from bridge.sequencer import FrameSequencer, xz_distance_m

def _rec(
    timestamp: float,
    gps_x: float,
    gps_z: float,
    gps_y: float = 50.0,
    drone: str = "drone_0",
) -> FrameRecord:
    """Create a minimal FrameRecord for sequencer tests."""
    return FrameRecord(
        drone_id=drone,
        frame_id=f"{drone}_{int(timestamp * 1_000):012d}",
        timestamp=timestamp,
        image=np.zeros((48, 64, 3), dtype=np.uint8),
        gps_x=gps_x,
        gps_y=gps_y,
        gps_z=gps_z,
        imu_roll=0.0,
        imu_pitch=-math.pi / 2,
        imu_yaw=0.0,
        gimbal_pitch=-math.pi / 2,
        fov_deg=84.0,
        image_wh=(64, 48),
    )

def _seq(**kwargs) -> FrameSequencer:
    """FrameSequencer with permissive defaults unless overridden."""
    defaults = dict(
        min_frames_per_batch=1,
        max_pair_distance_m=100.0,
        max_pair_time_gap_sec=100.0,
    )
    defaults.update(kwargs)
    return FrameSequencer(**defaults)

class TestXzDistance:
    def test_identical_positions_returns_zero(self):
        a = _rec(0.0, 5.0, -3.0)
        b = _rec(1.0, 5.0, -3.0)
        assert xz_distance_m(a, b) == pytest.approx(0.0)

    def test_known_3_4_5_triangle(self):
        a = _rec(0.0, 0.0, 0.0)
        b = _rec(1.0, 3.0, 4.0)
        assert xz_distance_m(a, b) == pytest.approx(5.0)

    def test_symmetry(self):
        a = _rec(0.0, 10.0, 0.0)
        b = _rec(1.0,  0.0, 10.0)
        assert xz_distance_m(a, b) == pytest.approx(xz_distance_m(b, a))

    def test_ignores_altitude_gps_y(self):
        """XZ distance must not consider altitude (gps_y)."""
        a = _rec(0.0, 0.0, 0.0, gps_y=10.0)
        b = _rec(1.0, 0.0, 0.0, gps_y=200.0)
        assert xz_distance_m(a, b) == pytest.approx(0.0)

class TestInsertionOrdering:
    def test_out_of_order_insertion_yields_sorted_pairs(self):
        """Frames added in reverse timestamp order must produce ascending pairs."""
        seq = _seq()
        records = [_rec(float(i), float(i) * 5, 0.0) for i in range(5, 0, -1)]
        for r in records:
            seq.add_frame(r)

        pairs = seq.get_pairs(flush=True)
        assert len(pairs) > 0
        for a, b in pairs:
            assert a.timestamp < b.timestamp

    def test_interleaved_drones_sorted_by_timestamp(self):
        """Frames from two drones interleaved by timestamp stay globally sorted."""
        seq = _seq()
        for t in range(6):
            seq.add_frame(_rec(float(t), float(t) * 3, 0.0,
                               drone=f"drone_{t % 2}"))

        pairs = seq.get_pairs(flush=True)
        for a, b in pairs:
            assert a.timestamp < b.timestamp

    def test_pending_count_reflects_buffer_size(self):
        seq = _seq(min_frames_per_batch=10)
        for i in range(4):
            seq.add_frame(_rec(float(i), 0.0, 0.0))
        assert seq.pending_count() == 4

class TestSpatialFilter:
    def test_frames_beyond_max_distance_not_paired(self):
        max_dist = 50.0
        seq = _seq(max_pair_distance_m=max_dist)
        seq.add_frame(_rec(0.0,          0.0, 0.0))
        seq.add_frame(_rec(1.0, max_dist + 1, 0.0))
        assert seq.get_pairs(flush=True) == []

    def test_frames_at_boundary_paired(self):
        max_dist = 50.0
        seq = _seq(max_pair_distance_m=max_dist)
        seq.add_frame(_rec(0.0,          0.0, 0.0))
        seq.add_frame(_rec(1.0, max_dist - 1, 0.0))
        pairs = seq.get_pairs(flush=True)
        assert len(pairs) == 1

    def test_diagonal_distance_respected(self):
        max_dist = 10.0
        seq = _seq(max_pair_distance_m=max_dist)
        seq.add_frame(_rec(0.0, 0.0, 0.0))
        seq.add_frame(_rec(1.0, 8.0, 8.0))
        assert seq.get_pairs(flush=True) == []

    def test_close_frames_paired_across_multiple_batches(self):
        seq = _seq(min_frames_per_batch=2, max_pair_distance_m=100.0)
        for i in range(4):
            seq.add_frame(_rec(float(i), float(i) * 5, 0.0))
        pairs = seq.get_pairs(flush=False)
        assert len(pairs) > 0

class TestTemporalFilter:
    def test_frames_exceeding_time_gap_not_paired(self):
        max_gap = 5.0
        seq = _seq(max_pair_time_gap_sec=max_gap)
        seq.add_frame(_rec(0.0, 0.0, 0.0))
        seq.add_frame(_rec(max_gap + 1.0, 5.0, 0.0))
        assert seq.get_pairs(flush=True) == []

    def test_frames_within_time_gap_paired(self):
        max_gap = 5.0
        seq = _seq(max_pair_time_gap_sec=max_gap)
        seq.add_frame(_rec(0.0, 0.0, 0.0))
        seq.add_frame(_rec(max_gap - 0.1, 5.0, 0.0))
        pairs = seq.get_pairs(flush=True)
        assert len(pairs) == 1

class TestBatchThreshold:
    def test_below_threshold_returns_empty(self):
        seq = _seq(min_frames_per_batch=6)
        for i in range(4):
            seq.add_frame(_rec(float(i), float(i) * 5, 0.0))
        assert seq.get_pairs(flush=False) == []

    def test_at_threshold_returns_pairs(self):
        seq = _seq(min_frames_per_batch=4)
        for i in range(4):
            seq.add_frame(_rec(float(i), float(i) * 5, 0.0))
        pairs = seq.get_pairs(flush=False)
        assert len(pairs) > 0

    def test_flush_bypasses_threshold(self):
        seq = _seq(min_frames_per_batch=100)
        for i in range(3):
            seq.add_frame(_rec(float(i), float(i) * 5, 0.0))
        assert seq.get_pairs(flush=False) == []
        pairs = seq.get_pairs(flush=True)
        assert len(pairs) == 2

class TestFlushAndNoDuplicates:
    def test_flush_emits_all_pending_pairs(self):
        seq = _seq(min_frames_per_batch=99)
        for i in range(5):
            seq.add_frame(_rec(float(i), float(i) * 5, 0.0))
        pairs = seq.get_pairs(flush=True)
        assert len(pairs) == 4

    def test_flush_does_not_re_emit_same_pairs(self):
        seq = _seq()
        for i in range(4):
            seq.add_frame(_rec(float(i), float(i) * 5, 0.0))
        first  = seq.get_pairs(flush=True)
        second = seq.get_pairs(flush=True)
        assert len(first) > 0
        assert second == []

    def test_pairs_not_duplicated_across_incremental_batches(self):
        """
        Add 6 frames in two batches of 3 and verify no pair is returned twice.
        """
        seq = _seq(min_frames_per_batch=3)
        seen: set[tuple[str, str]] = set()

        for i in range(3):
            seq.add_frame(_rec(float(i), float(i) * 5, 0.0))
        for a, b in seq.get_pairs(flush=False):
            key = (a.frame_id, b.frame_id)
            assert key not in seen, f"Duplicate pair: {key}"
            seen.add(key)

        for i in range(3, 6):
            seq.add_frame(_rec(float(i), float(i) * 5, 0.0))
        for a, b in seq.get_pairs(flush=True):
            key = (a.frame_id, b.frame_id)
            assert key not in seen, f"Duplicate pair: {key}"
            seen.add(key)

class TestFrameAppearanceConstraint:
    def test_each_frame_in_at_most_two_pairs(self):
        """
        Across all emitted pairs from a single get_pairs call, no frame ID
        may appear in more than two pairs (once as left, once as right).
        """
        seq = _seq()
        n = 10
        for i in range(n):
            seq.add_frame(_rec(float(i), float(i) * 5.0, 0.0))

        pairs = seq.get_pairs(flush=True)
        assert len(pairs) == n - 1

        appearances: dict[str, int] = {}
        for a, b in pairs:
            appearances[a.frame_id] = appearances.get(a.frame_id, 0) + 1
            appearances[b.frame_id] = appearances.get(b.frame_id, 0) + 1

        for fid, count in appearances.items():
            assert count <= 2, (
                f"Frame {fid} appears in {count} pairs (maximum is 2)"
            )

    def test_first_and_last_frames_appear_in_one_pair(self):
        """The first frame is only a left-partner; the last only a right-partner."""
        seq = _seq()
        records = [_rec(float(i), float(i) * 5, 0.0) for i in range(5)]
        for r in records:
            seq.add_frame(r)

        pairs = seq.get_pairs(flush=True)
        assert len(pairs) == 4

        all_left  = [a.frame_id for a, _ in pairs]
        all_right = [b.frame_id for _, b in pairs]

        first_id = records[0].frame_id
        last_id  = records[-1].frame_id

        assert all_left.count(first_id)  == 1
        assert all_right.count(first_id) == 0
        assert all_right.count(last_id)  == 1
        assert all_left.count(last_id)   == 0

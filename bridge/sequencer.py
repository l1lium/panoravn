from __future__ import annotations

import bisect
import math
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bridge.ingestor import FrameRecord

def xz_distance_m(a: "FrameRecord", b: "FrameRecord") -> float:
    return math.sqrt((a.gps_x - b.gps_x) ** 2 + (a.gps_z - b.gps_z) ** 2)

class FrameSequencer:
    def __init__(
        self,
        min_frames_per_batch: int = 6,
        max_pair_distance_m: float = 60.0,
        max_pair_time_gap_sec: float = 30.0,
    ) -> None:
        self._min_frames = min_frames_per_batch
        self._max_dist = max_pair_distance_m
        self._max_time_gap = max_pair_time_gap_sec

        self._buf: list[FrameRecord] = []
        self._emit_cursor: int = 0
        self._lock = threading.Lock()

    def add_frame(self, record: "FrameRecord") -> None:
        with self._lock:
            keys = [(_r.timestamp, _r.frame_id) for _r in self._buf]
            idx = bisect.bisect_left(keys, (record.timestamp, record.frame_id))
            self._buf.insert(idx, record)

    def get_pairs(
        self, flush: bool = False
    ) -> list[tuple["FrameRecord", "FrameRecord"]]:
        with self._lock:
            n = len(self._buf)
            if not flush and n < self._min_frames:
                return []

            pairs: list[tuple[FrameRecord, FrameRecord]] = []
            for i in range(self._emit_cursor, n - 1):
                a, b = self._buf[i], self._buf[i + 1]
                if self._qualifies(a, b):
                    pairs.append((a, b))

            if n > 1:
                self._emit_cursor = n - 1

            return pairs

    def pending_count(self) -> int:
        with self._lock:
            return len(self._buf)

    def _qualifies(self, a: "FrameRecord", b: "FrameRecord") -> bool:
        if abs(b.timestamp - a.timestamp) > self._max_time_gap:
            return False
        if xz_distance_m(a, b) > self._max_dist:
            return False
        return True

_default: FrameSequencer | None = None

def _get_default() -> FrameSequencer:
    global _default
    if _default is None:
        _default = FrameSequencer()
    return _default

def add_frame(record: "FrameRecord") -> None:
    _get_default().add_frame(record)

def get_pairs(flush: bool = False) -> list[tuple["FrameRecord", "FrameRecord"]]:
    return _get_default().get_pairs(flush=flush)

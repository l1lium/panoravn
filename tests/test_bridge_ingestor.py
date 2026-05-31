"""
Tests for bridge/ingestor.py

Covers:
  - parse_frame_record: happy-path field extraction and RGB conversion
  - parse_frame_record: all error branches (bad JSON, missing keys,
    unsupported encoding, corrupt JPEG, empty payload)
  - start_zmq_ingestor: callback count verified with a mocked zmq module
  - start_disk_ingestor: filesystem round-trip via a temporary directory
"""

import json
import os
import sys
import tempfile
import threading
import time
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bridge.ingestor import FrameRecord, parse_frame_record

def _make_jpeg(width: int = 64, height: int = 48) -> bytes:
    """Return valid JPEG bytes for a small synthetic image."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[height // 4 : height * 3 // 4, width // 4 : width * 3 // 4] = [100, 180, 50]
    ok, buf = cv2.imencode(".jpg", img)
    assert ok, "cv2.imencode failed in test fixture"
    return buf.tobytes()

def _make_meta(**overrides) -> bytes:
    """Return a valid telemetry JSON payload (bytes), with optional field overrides."""
    base = {
        "drone_id": "drone_0",
        "frame_id": "drone_0_000001",
        "timestamp": 1.0,
        "gps": {"x": 12.5, "y": 50.0, "z": -34.1},
        "imu": {"roll": 0.01, "pitch": -0.02, "yaw": 1.57},
        "gimbal_pitch": -1.5708,
        "fov_deg": 84.0,
        "image_wh": [64, 48],
        "encoding": "jpeg",
    }
    base.update(overrides)
    return json.dumps(base).encode()

class TestParseFrameRecordValid:
    def test_returns_frame_record_instance(self):
        rec = parse_frame_record(_make_jpeg(), _make_meta())
        assert isinstance(rec, FrameRecord)

    def test_drone_id_and_frame_id(self):
        rec = parse_frame_record(_make_jpeg(), _make_meta())
        assert rec.drone_id == "drone_0"
        assert rec.frame_id == "drone_0_000001"

    def test_timestamp(self):
        rec = parse_frame_record(_make_jpeg(), _make_meta(timestamp=42.5))
        assert rec.timestamp == pytest.approx(42.5)

    def test_gps_xyz(self):
        rec = parse_frame_record(_make_jpeg(), _make_meta())
        assert rec.gps_x == pytest.approx(12.5)
        assert rec.gps_y == pytest.approx(50.0)
        assert rec.gps_z == pytest.approx(-34.1)

    def test_imu_fields(self):
        rec = parse_frame_record(_make_jpeg(), _make_meta())
        assert rec.imu_roll  == pytest.approx(0.01)
        assert rec.imu_pitch == pytest.approx(-0.02)
        assert rec.imu_yaw   == pytest.approx(1.57)

    def test_image_dtype_and_channels(self):
        rec = parse_frame_record(_make_jpeg(64, 48), _make_meta())
        assert rec.image.dtype == np.uint8
        assert rec.image.ndim == 3
        assert rec.image.shape[2] == 3

    def test_image_decoded_as_rgb_not_bgr(self):
        img_bgr = np.zeros((48, 64, 3), dtype=np.uint8)
        img_bgr[:, :, 0] = 200
        ok, buf = cv2.imencode(".jpg", img_bgr)
        assert ok
        rec = parse_frame_record(buf.tobytes(), _make_meta())
        assert rec.image[:, :, 2].mean() > rec.image[:, :, 0].mean()

    def test_accepts_bytes_meta(self):
        """parse_frame_record must accept bytes as meta_json."""
        rec = parse_frame_record(_make_jpeg(), _make_meta())
        assert rec.fov_deg == pytest.approx(84.0)

    def test_accepts_str_meta(self):
        """parse_frame_record must accept str as meta_json."""
        rec = parse_frame_record(_make_jpeg(), _make_meta().decode())
        assert rec.drone_id == "drone_0"

    def test_encoding_jpg_alias_accepted(self):
        rec = parse_frame_record(_make_jpeg(), _make_meta(encoding="jpg"))
        assert isinstance(rec, FrameRecord)

    def test_image_wh_stored_as_tuple(self):
        rec = parse_frame_record(_make_jpeg(64, 48), _make_meta())
        assert rec.image_wh == (64, 48)

    def test_gimbal_pitch(self):
        rec = parse_frame_record(_make_jpeg(), _make_meta(gimbal_pitch=-1.5708))
        assert rec.gimbal_pitch == pytest.approx(-1.5708)

class TestParseFrameRecordErrors:
    def test_raises_on_malformed_json(self):
        with pytest.raises(ValueError, match="Malformed telemetry JSON"):
            parse_frame_record(_make_jpeg(), b"not {{ valid json")

    def test_raises_on_empty_json(self):
        with pytest.raises(ValueError):
            parse_frame_record(_make_jpeg(), b"")

    def test_raises_on_missing_top_level_key(self):
        meta = json.loads(_make_meta())
        del meta["gps"]
        with pytest.raises(ValueError, match="missing required keys"):
            parse_frame_record(_make_jpeg(), json.dumps(meta).encode())

    def test_raises_on_all_required_keys_present(self):
        """Sanity-check: correct payload should NOT raise."""
        parse_frame_record(_make_jpeg(), _make_meta())

    def test_raises_on_unsupported_encoding(self):
        with pytest.raises(ValueError, match="Unsupported encoding"):
            parse_frame_record(_make_jpeg(), _make_meta(encoding="png"))

    def test_raises_on_empty_jpeg(self):
        with pytest.raises(ValueError, match="Empty JPEG payload"):
            parse_frame_record(b"", _make_meta())

    def test_raises_on_corrupt_jpeg(self):
        with pytest.raises(ValueError, match="cv2.imdecode failed"):
            parse_frame_record(b"\x00\xff\xfe corrupt bytes", _make_meta())

    def test_raises_on_missing_gps_subkey(self):
        meta = json.loads(_make_meta())
        del meta["gps"]["x"]
        with pytest.raises(ValueError, match="Telemetry field extraction failed"):
            parse_frame_record(_make_jpeg(), json.dumps(meta).encode())

    def test_raises_on_missing_imu_subkey(self):
        meta = json.loads(_make_meta())
        del meta["imu"]["roll"]
        with pytest.raises(ValueError, match="Telemetry field extraction failed"):
            parse_frame_record(_make_jpeg(), json.dumps(meta).encode())

    def test_raises_on_missing_timestamp(self):
        meta = json.loads(_make_meta())
        del meta["timestamp"]
        with pytest.raises(ValueError, match="missing required keys"):
            parse_frame_record(_make_jpeg(), json.dumps(meta).encode())

def _build_zmq_mock(messages: list[list[bytes]], *, stop_after_n: int = 1):
    """
    Return (zmq_mock, socket_mock) where the mocked PULL socket yields
    exactly `stop_after_n` POLLIN events (each delivering one message from
    `messages`), then returns empty poll results so the stop_event loop exits.
    """
    zmq_real = pytest.importorskip("zmq")

    zmq_mock = MagicMock()
    zmq_mock.PULL   = zmq_real.PULL
    zmq_mock.POLLIN = zmq_real.POLLIN
    zmq_mock.RCVHWM = zmq_real.RCVHWM
    zmq_mock.LINGER = zmq_real.LINGER

    socket_mock = MagicMock()
    ctx_instance = MagicMock()
    ctx_instance.socket.return_value = socket_mock
    zmq_mock.Context.return_value = ctx_instance

    poller_mock = MagicMock()
    zmq_mock.Poller.return_value = poller_mock

    msg_iter   = iter(messages)
    pollin_left = [stop_after_n]

    def _poll(timeout):
        if pollin_left[0] > 0:
            pollin_left[0] -= 1
            return {socket_mock: zmq_real.POLLIN}
        return {}

    poller_mock.poll.side_effect = _poll
    socket_mock.recv_multipart.side_effect = lambda: next(msg_iter)

    return zmq_mock, socket_mock

class TestZmqIngestor:
    """Verify start_zmq_ingestor thread behaviour via a mocked zmq module."""

    def test_callback_called_once_for_valid_message(self):
        pytest.importorskip("zmq")
        from bridge.ingestor import start_zmq_ingestor

        jpeg = _make_jpeg()
        meta = _make_meta()
        zmq_mock, _ = _build_zmq_mock([[meta, jpeg]], stop_after_n=1)

        received = []
        stop = threading.Event()

        def _cb(rec):
            received.append(rec)
            stop.set()

        with patch.dict("sys.modules", {"zmq": zmq_mock}):
            t = start_zmq_ingestor("tcp://127.0.0.1:59995", _cb, stop_event=stop)
            t.join(timeout=3.0)

        assert len(received) == 1
        assert isinstance(received[0], FrameRecord)
        assert received[0].drone_id == "drone_0"

    def test_malformed_message_not_forwarded_to_callback(self):
        pytest.importorskip("zmq")
        from bridge.ingestor import start_zmq_ingestor

        zmq_mock, _ = _build_zmq_mock([[b"bad json", b"garbage"]], stop_after_n=1)

        received = []
        stop = threading.Event()

        with patch.dict("sys.modules", {"zmq": zmq_mock}):
            t = start_zmq_ingestor(
                "tcp://127.0.0.1:59994",
                lambda r: received.append(r),
                stop_event=stop,
            )
            time.sleep(0.25)
            stop.set()
            t.join(timeout=2.0)

        assert received == []

    def test_wrong_part_count_discarded(self):
        """A single-part message (not the expected 2) must be silently dropped."""
        pytest.importorskip("zmq")
        from bridge.ingestor import start_zmq_ingestor

        zmq_mock, _ = _build_zmq_mock([[b"only one part"]], stop_after_n=1)

        received = []
        stop = threading.Event()

        with patch.dict("sys.modules", {"zmq": zmq_mock}):
            t = start_zmq_ingestor(
                "tcp://127.0.0.1:59993",
                lambda r: received.append(r),
                stop_event=stop,
            )
            time.sleep(0.25)
            stop.set()
            t.join(timeout=2.0)

        assert received == []

class TestDiskIngestor:
    """Test _scan_once logic via start_disk_ingestor and a real temp directory."""

    @staticmethod
    def _write_pair(drone_dir: str, stem: str, meta: bytes, jpeg: bytes) -> None:
        os.makedirs(drone_dir, exist_ok=True)
        with open(os.path.join(drone_dir, f"{stem}.json"), "wb") as f:
            f.write(meta)
        with open(os.path.join(drone_dir, f"{stem}.jpg"), "wb") as f:
            f.write(jpeg)

    def test_valid_pair_delivered_to_callback(self):
        from bridge.ingestor import start_disk_ingestor

        with tempfile.TemporaryDirectory() as tmp:
            drone_dir = os.path.join(tmp, "drone_0")
            self._write_pair(drone_dir, "frame_000001", _make_meta(), _make_jpeg())

            received = []
            stop = threading.Event()

            def _cb(rec):
                received.append(rec)
                stop.set()

            t = start_disk_ingestor(tmp, _cb, poll_interval_s=0.04, stop_event=stop)
            stop.wait(timeout=3.0)
            stop.set()
            t.join(timeout=1.5)

        assert len(received) == 1
        assert received[0].drone_id == "drone_0"

    def test_corrupt_pair_not_delivered(self):
        from bridge.ingestor import start_disk_ingestor

        with tempfile.TemporaryDirectory() as tmp:
            drone_dir = os.path.join(tmp, "drone_0")
            self._write_pair(drone_dir, "frame_000001", b"bad json", b"bad jpeg")

            received = []
            stop = threading.Event()

            t = start_disk_ingestor(tmp, lambda r: received.append(r),
                                    poll_interval_s=0.04, stop_event=stop)
            time.sleep(0.25)
            stop.set()
            t.join(timeout=1.5)

        assert received == []

    def test_json_without_jpeg_not_delivered(self):
        """Incomplete pair (JSON present but JPEG missing) must not be processed."""
        from bridge.ingestor import start_disk_ingestor

        with tempfile.TemporaryDirectory() as tmp:
            drone_dir = os.path.join(tmp, "drone_0")
            os.makedirs(drone_dir)
            with open(os.path.join(drone_dir, "frame_000001.json"), "wb") as f:
                f.write(_make_meta())

            received = []
            stop = threading.Event()

            t = start_disk_ingestor(tmp, lambda r: received.append(r),
                                    poll_interval_s=0.04, stop_event=stop)
            time.sleep(0.25)
            stop.set()
            t.join(timeout=1.5)

        assert received == []

    def test_processed_frames_moved_to_subdirectory(self):
        """After a successful ingest the source files must be moved to processed/."""
        from bridge.ingestor import start_disk_ingestor

        with tempfile.TemporaryDirectory() as tmp:
            drone_dir = os.path.join(tmp, "drone_0")
            self._write_pair(drone_dir, "frame_000001", _make_meta(), _make_jpeg())

            stop = threading.Event()

            def _cb(rec):
                stop.set()

            t = start_disk_ingestor(tmp, _cb, poll_interval_s=0.04, stop_event=stop)
            stop.wait(timeout=3.0)
            stop.set()
            t.join(timeout=1.5)

            assert not os.path.exists(os.path.join(drone_dir, "frame_000001.json"))
            assert not os.path.exists(os.path.join(drone_dir, "frame_000001.jpg"))
            processed = os.path.join(drone_dir, "processed")
            assert os.path.isdir(processed)

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class FrameRecord:
    drone_id:     str
    frame_id:     str
    timestamp:    float
    image:        np.ndarray
    gps_x:        float
    gps_y:        float
    gps_z:        float
    imu_roll:     float
    imu_pitch:    float
    imu_yaw:      float
    gimbal_pitch: float
    fov_deg:      float
    image_wh:     tuple[int, int]

_REQUIRED_META_KEYS = frozenset({
    "drone_id", "frame_id", "timestamp", "gps", "imu",
    "gimbal_pitch", "fov_deg", "image_wh", "encoding",
})

def parse_frame_record(jpeg_bytes: bytes, meta_json: str | bytes) -> FrameRecord:
    if isinstance(meta_json, bytes):
        meta_json = meta_json.decode("utf-8")
    try:
        meta: dict = json.loads(meta_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed telemetry JSON: {exc}") from exc

    missing = _REQUIRED_META_KEYS - meta.keys()
    if missing:
        raise ValueError(f"Telemetry JSON missing required keys: {missing}")

    if meta.get("encoding") not in ("jpeg", "jpg"):
        raise ValueError(f"Unsupported encoding {meta.get('encoding')!r}; expected 'jpeg'")

    if not jpeg_bytes:
        raise ValueError("Empty JPEG payload")
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("cv2.imdecode failed — JPEG bytes may be corrupt")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    try:
        gps = meta["gps"]
        imu = meta["imu"]
        return FrameRecord(
            drone_id=str(meta["drone_id"]),
            frame_id=str(meta["frame_id"]),
            timestamp=float(meta["timestamp"]),
            image=rgb,
            gps_x=float(gps["x"]),
            gps_y=float(gps["y"]),
            gps_z=float(gps["z"]),
            imu_roll=float(imu["roll"]),
            imu_pitch=float(imu["pitch"]),
            imu_yaw=float(imu["yaw"]),
            gimbal_pitch=float(meta["gimbal_pitch"]),
            fov_deg=float(meta["fov_deg"]),
            image_wh=(int(meta["image_wh"][0]), int(meta["image_wh"][1])),
        )
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError(f"Telemetry field extraction failed: {exc}") from exc

def start_zmq_ingestor(
    endpoint: str,
    callback: Callable[[FrameRecord], None],
    hwm: int = 200,
    *,
    stop_event: threading.Event | None = None,
) -> threading.Thread:
    try:
        import zmq
    except ImportError:
        raise ImportError("pyzmq not found. Install it: pip install pyzmq")

    def _loop() -> None:
        context = zmq.Context()
        socket = context.socket(zmq.PULL)
        socket.setsockopt(zmq.RCVHWM, hwm)
        socket.setsockopt(zmq.LINGER, 0)
        socket.bind(endpoint)
        logger.info("ZMQ ingestor bound to %s (RCVHWM=%d)", endpoint, hwm)

        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)

        frames_received = 0
        frames_rejected = 0
        try:
            while True:
                if stop_event is not None and stop_event.is_set():
                    break
                socks = dict(poller.poll(timeout=100))
                if socket not in socks:
                    continue

                parts = socket.recv_multipart()
                if len(parts) != 2:
                    logger.warning(
                        "Unexpected multipart length %d (expected 2); discarding.", len(parts)
                    )
                    frames_rejected += 1
                    continue

                meta_bytes, jpeg_bytes = parts
                try:
                    record = parse_frame_record(jpeg_bytes, meta_bytes)
                except ValueError as exc:
                    logger.warning("Frame rejected: %s", exc)
                    frames_rejected += 1
                    continue

                frames_received += 1
                callback(record)
        finally:
            socket.close()
            context.term()
            logger.info(
                "ZMQ ingestor stopped. received=%d rejected=%d",
                frames_received, frames_rejected,
            )

    thread = threading.Thread(target=_loop, daemon=True, name="zmq-ingestor")
    thread.start()
    return thread

def start_disk_ingestor(
    frames_dir: str,
    callback: Callable[[FrameRecord], None],
    poll_interval_s: float = 0.1,
    *,
    stop_event: threading.Event | None = None,
) -> threading.Thread:
    def _loop() -> None:
        logger.info("Disk ingestor watching %s (poll=%.2fs)", frames_dir, poll_interval_s)
        while True:
            if stop_event is not None and stop_event.is_set():
                break
            _scan_once(frames_dir, callback)
            time.sleep(poll_interval_s)
        logger.info("Disk ingestor stopped.")

    thread = threading.Thread(target=_loop, daemon=True, name="disk-ingestor")
    thread.start()
    return thread

def _scan_once(
    frames_dir: str,
    callback: Callable[[FrameRecord], None],
) -> None:
    if not os.path.isdir(frames_dir):
        return

    for drone_entry in os.scandir(frames_dir):
        if not drone_entry.is_dir():
            continue
        drone_dir = drone_entry.path
        processed_dir = os.path.join(drone_dir, "processed")

        for entry in sorted(os.scandir(drone_dir), key=lambda e: e.name):
            if not entry.name.endswith(".json"):
                continue
            frame_stem = entry.name[:-5]
            jpeg_path = os.path.join(drone_dir, f"{frame_stem}.jpg")
            if not os.path.isfile(jpeg_path):
                continue

            try:
                with open(entry.path, "rb") as f:
                    meta_bytes = f.read()
                with open(jpeg_path, "rb") as f:
                    jpeg_bytes = f.read()
                record = parse_frame_record(jpeg_bytes, meta_bytes)
            except (OSError, ValueError) as exc:
                logger.warning("Disk frame %s rejected: %s", frame_stem, exc)
                _move_to_processed(entry.path, jpeg_path, processed_dir, tag="rejected")
                continue

            callback(record)
            _move_to_processed(entry.path, jpeg_path, processed_dir, tag="ok")

def _move_to_processed(json_path: str, jpeg_path: str, processed_dir: str, tag: str) -> None:
    os.makedirs(processed_dir, exist_ok=True)
    for src in (json_path, jpeg_path):
        if os.path.isfile(src):
            dst = os.path.join(processed_dir, f"{tag}_{os.path.basename(src)}")
            try:
                os.rename(src, dst)
            except OSError as exc:
                logger.debug("Could not move %s: %s", src, exc)

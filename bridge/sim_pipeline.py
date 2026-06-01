from __future__ import annotations

import argparse
import datetime
import logging
import os
import queue
import sys
import threading
import time

import cv2
import numpy as np
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bridge.ingestor import FrameRecord, start_disk_ingestor, start_zmq_ingestor
from bridge.sequencer import FrameSequencer
from bridge import telemetry
from core.stitching_pipeline import stitch_pair
from models.model_factory import create_matcher

logger = logging.getLogger(__name__)

def _load_yaml(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)

def _merge_configs(sim_cfg_path: str) -> tuple[dict, dict]:
    sim_cfg = _load_yaml(sim_cfg_path)
    stitch_cfg_rel: str = sim_cfg.get("stitching", {}).get("config_file", "configs/baseline.yaml")
    stitch_cfg_path = os.path.join(_ROOT, stitch_cfg_rel)
    stitch_cfg = _load_yaml(stitch_cfg_path)
    return sim_cfg, stitch_cfg

def _make_run_dir(base_dir: str, experiment_name: str) -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(_ROOT, base_dir, "runs", f"{ts}_{experiment_name}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def _save_panorama(panorama: np.ndarray, run_dir: str, pair_idx: int) -> None:
    path = os.path.join(run_dir, f"panorama_{pair_idx:05d}.jpg")
    cv2.imwrite(path, panorama if panorama.ndim == 3 else
                cv2.cvtColor(panorama, cv2.COLOR_GRAY2BGR))

def _save_metrics(metrics: dict, run_dir: str, pair_idx: int) -> None:
    import json
    safe = {k: v for k, v in metrics.items() if k != "panorama"}
    path = os.path.join(run_dir, f"metrics_{pair_idx:05d}.json")
    with open(path, "w") as fh:
        json.dump(safe, fh, indent=2)

def run_sim_pipeline(config: dict, sim_cfg_path: str, transport: str = "zmq") -> None:
    sim_cfg   = config["simulation"]
    map_cfg   = config["map"]
    trans_cfg = config.get("transport", {})
    bridge_cfg = config.get("bridge", {})
    out_cfg   = config.get("output", {})

    stitch_cfg_rel = config.get("stitching", {}).get("config_file", "configs/baseline.yaml")
    stitch_cfg_path = os.path.join(_ROOT, stitch_cfg_rel)
    stitch_cfg = _load_yaml(stitch_cfg_path)

    matcher_name = config.get("stitching", {}).get("matcher", stitch_cfg.get("matcher", {}).get("name", "sift"))
    stitch_cfg.setdefault("matcher", {})["name"] = matcher_name
    matcher = create_matcher(stitch_cfg["matcher"])

    run_dir = _make_run_dir(
        out_cfg.get("base_dir", "outputs"),
        out_cfg.get("experiment_name", "sim_baseline"),
    )
    logger.info("Run directory: %s", run_dir)

    seq = FrameSequencer(
        min_frames_per_batch=int(bridge_cfg.get("min_frames_per_batch", 6)),
        max_pair_distance_m=float(bridge_cfg.get("max_pair_distance_m", 60.0)),
        max_pair_time_gap_sec=float(bridge_cfg.get("max_pair_time_gap_sec", 30.0)),
    )

    frame_queue: queue.Queue[FrameRecord] = queue.Queue(maxsize=0)

    def _on_frame(record: FrameRecord) -> None:
        frame_queue.put_nowait(record)

    stop_event = threading.Event()

    if transport == "zmq":
        endpoint = trans_cfg.get("zmq_endpoint", "tcp://localhost:5555")
        bind_addr = endpoint.replace("localhost", "0.0.0.0").replace("127.0.0.1", "0.0.0.0")
        hwm = int(trans_cfg.get("zmq_rcvhwm", 200))
        ingestor_thread = start_zmq_ingestor(bind_addr, _on_frame, hwm=hwm, stop_event=stop_event)
        logger.info("ZMQ ingestor bound to %s", bind_addr)
    else:
        frames_dir = os.path.join(_ROOT, trans_cfg.get("disk_frames_dir", "data/sim_frames"))
        ingestor_thread = start_disk_ingestor(frames_dir, _on_frame, stop_event=stop_event)
        logger.info("Disk ingestor watching %s", frames_dir)

    flush_interval = float(bridge_cfg.get("flush_interval_sec", 10.0))
    use_telemetry_prior = bool(bridge_cfg.get("use_telemetry_prior", True))
    pair_idx = 0
    last_flush_t = time.monotonic()

    logger.info("Pipeline running. Ctrl-C to stop.")
    try:
        while True:
            drained = 0
            while not frame_queue.empty():
                try:
                    record = frame_queue.get_nowait()
                    seq.add_frame(record)
                    drained += 1
                except queue.Empty:
                    break
            if drained:
                logger.debug("Sequencer: drained %d frames (buffered=%d)", drained, seq.pending_count())

            now = time.monotonic()
            flush = (now - last_flush_t) >= flush_interval
            if flush:
                last_flush_t = now

            pairs = seq.get_pairs(flush=flush)
            for frame_a, frame_b in pairs:
                _process_pair(
                    frame_a, frame_b,
                    matcher=matcher,
                    stitch_cfg=stitch_cfg,
                    run_dir=run_dir,
                    pair_idx=pair_idx,
                    use_prior=use_telemetry_prior,
                )
                pair_idx += 1

            time.sleep(0.01)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt — flushing remaining pairs …")
        while not frame_queue.empty():
            try:
                seq.add_frame(frame_queue.get_nowait())
            except queue.Empty:
                break
        for frame_a, frame_b in seq.get_pairs(flush=True):
            _process_pair(
                frame_a, frame_b,
                matcher=matcher,
                stitch_cfg=stitch_cfg,
                run_dir=run_dir,
                pair_idx=pair_idx,
                use_prior=use_telemetry_prior,
            )
            pair_idx += 1
    finally:
        stop_event.set()
        ingestor_thread.join(timeout=3.0)
        logger.info("Pipeline stopped. %d pairs processed → %s", pair_idx, run_dir)

def _process_pair(
    frame_a: FrameRecord,
    frame_b: FrameRecord,
    *,
    matcher,
    stitch_cfg: dict,
    run_dir: str,
    pair_idx: int,
    use_prior: bool,
) -> None:
    logger.info(
        "Stitching pair %d: %s + %s (Δt=%.2fs, ΔXZ=%.1fm)",
        pair_idx,
        frame_a.frame_id,
        frame_b.frame_id,
        abs(frame_b.timestamp - frame_a.timestamp),
        telemetry.xz_distance_m(frame_a, frame_b) if hasattr(telemetry, "xz_distance_m") else 0.0,
    )

    img_a = cv2.cvtColor(frame_a.image, cv2.COLOR_RGB2BGR)
    img_b = cv2.cvtColor(frame_b.image, cv2.COLOR_RGB2BGR)

    try:
        metrics = stitch_pair(img_a, img_b, matcher, stitch_cfg)
    except Exception as exc:
        logger.warning("stitch_pair failed for pair %d: %s", pair_idx, exc)
        return

    panorama = metrics.get("panorama")
    if panorama is not None:
        _save_panorama(panorama, run_dir, pair_idx)
    _save_metrics(metrics, run_dir, pair_idx)

    logger.info(
        "  inliers=%d  reprojection=%.2f  valid=%s",
        metrics.get("num_inliers", 0),
        metrics.get("reprojection_error", float("inf")),
        metrics.get("homography_valid", False),
    )

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Drone swarm simulation → panorama pipeline")
    parser.add_argument(
        "--config",
        default="simulation/configs/simulation.yaml",
        help="Path to simulation.yaml (default: simulation/configs/simulation.yaml)",
    )
    parser.add_argument(
        "--transport",
        choices=["zmq", "disk"],
        default="zmq",
        help="Frame transport: zmq (default) or disk",
    )
    args = parser.parse_args()

    cfg_path = os.path.join(_ROOT, args.config) if not os.path.isabs(args.config) else args.config
    if not os.path.isfile(cfg_path):
        parser.error(f"Config file not found: {cfg_path!r}")

    config = _load_yaml(cfg_path)
    run_sim_pipeline(config, sim_cfg_path=cfg_path, transport=args.transport)

if __name__ == "__main__":
    main()

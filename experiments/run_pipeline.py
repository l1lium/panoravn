from __future__ import annotations

import argparse
import copy
import datetime
import json
import logging
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from string import Template
from typing import Any

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")

_SIM_FRAMES = _ROOT / "data" / "sim_frames"
_CLIENTS    = _ROOT / "data" / "clients"
_RUNS       = _ROOT / "outputs" / "runs"
_VENV_PY    = _ROOT / "venv" / "bin" / "python"

ALL_PHASES = ["partition", "baseline", "neural", "federated", "compare"]

def _detect_num_drones(sim_frames_dir: Path) -> int:
    if not sim_frames_dir.is_dir():
        return 0
    return sum(
        1 for d in sim_frames_dir.iterdir()
        if d.is_dir() and d.name.startswith("drone_")
    )

def _make_run_dir(tag: str) -> Path:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    d = _RUNS / f"{ts}_{tag}"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _save_metrics(metrics: dict[str, Any], run_dir: Path, pair_idx: int) -> None:
    def _clean(v: Any) -> Any:
        if isinstance(v, float) and (not math.isfinite(v)):
            return None
        return v

    safe = {k: _clean(v) for k, v in metrics.items() if k != "panorama"}
    path = run_dir / f"metrics_{pair_idx:05d}.json"
    with path.open("w") as fh:
        json.dump(safe, fh, indent=2)

def _save_panorama(panorama, run_dir: Path, pair_idx: int) -> None:
    import cv2
    if panorama is None:
        return
    out = run_dir / f"panorama_{pair_idx:05d}.jpg"
    img = panorama if panorama.ndim == 3 else cv2.cvtColor(panorama, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(out), img)

def phase_partition(num_clients: int) -> None:
    log.info("=" * 60)
    log.info("Phase 1: Partition  (%d clients)", num_clients)
    log.info("=" * 60)

    existing = [_CLIENTS / f"client_{i}" for i in range(num_clients)]
    if all(d.exists() for d in existing):
        log.info("Client datasets already present in %s — skipping.", _CLIENTS)
        return

    from experiments.split_clients import (
        _assign_scenes_to_clients,
        _build_client,
        _collect_scene_scenes,
        _collect_sim_scenes,
        _is_sim_directory,
    )

    if not _SIM_FRAMES.is_dir():
        log.error("sim_frames not found: %s", _SIM_FRAMES)
        sys.exit(1)

    mode = "sim" if _is_sim_directory(_SIM_FRAMES) else "scene"
    log.info("Detected mode: %s", mode.upper())

    scenes = _collect_sim_scenes(_SIM_FRAMES) if mode == "sim" else _collect_scene_scenes(_SIM_FRAMES)
    if not scenes:
        log.error("No frames found in %s", _SIM_FRAMES)
        sys.exit(1)

    total_frames = sum(len(s.frames) for s in scenes)
    log.info("Found %d scene(s), %d total frames.", len(scenes), total_frames)

    buckets = _assign_scenes_to_clients(scenes, num_clients)
    for idx, client_scenes in enumerate(buckets):
        client_dir = _CLIENTS / f"client_{idx}"
        n_pairs = _build_client(client_dir, client_scenes, copy=False)
        log.info(
            "  client_%d: %d scene(s), %d frame(s), %d pair(s)",
            idx, len(client_scenes),
            sum(len(s.frames) for s in client_scenes),
            n_pairs,
        )

    log.info("Partition complete → %s", _CLIENTS)

def phase_batch_run(matcher_cfg: dict, base_stitch_cfg: dict, tag: str) -> Path:
    log.info("=" * 60)
    log.info("Phase: Batch run  [%s]", tag)
    log.info("=" * 60)

    from client.dataset import load_image, load_pairs_from_csv
    from core.io import save_config
    from core.stitching_pipeline import stitch_pair
    from models.model_factory import create_matcher

    run_dir = _make_run_dir(f"pipeline_{tag}")

    stitch_cfg = copy.deepcopy(base_stitch_cfg)
    stitch_cfg["matcher"] = copy.deepcopy(matcher_cfg)
    save_config(stitch_cfg, str(run_dir / "used_config.yaml"))

    client_dirs = sorted(_CLIENTS.glob("client_*"))
    if not client_dirs:
        log.error(
            "No client directories found in %s. "
            "Run the partition phase first.",
            _CLIENTS,
        )
        return run_dir

    try:
        matcher = create_matcher(matcher_cfg)
        log.info("Matcher ready: %s", matcher_cfg.get("name"))
    except Exception as exc:
        log.warning(
            "Cannot initialise %s matcher: %s — skipping batch run.",
            matcher_cfg.get("name"), exc,
        )
        return run_dir

    pair_idx   = 0
    n_valid    = 0
    n_skipped  = 0

    for client_dir in client_dirs:
        csv_path = client_dir / "pairs.csv"
        if not csv_path.exists():
            log.warning("No pairs.csv in %s — skipping.", client_dir)
            continue

        try:
            pairs = load_pairs_from_csv(str(client_dir), "pairs.csv")
        except Exception as exc:
            log.warning("Failed to load pairs from %s: %s", client_dir, exc)
            continue

        log.info("Processing %d pairs from %s …", len(pairs), client_dir.name)

        for pair in pairs:
            try:
                img_a = load_image(pair["image_a_path"])
                img_b = load_image(pair["image_b_path"])
            except FileNotFoundError:
                n_skipped += 1
                pair_idx  += 1
                continue

            try:
                metrics = stitch_pair(img_a, img_b, matcher, stitch_cfg)
            except Exception as exc:
                log.debug("stitch_pair failed (pair %d): %s", pair_idx, exc)
                metrics = {
                    "num_matches": 0,
                    "num_inliers": 0,
                    "homography_valid": False,
                    "reprojection_error": float("inf"),
                }

            if metrics.get("homography_valid"):
                n_valid += 1
                _save_panorama(metrics.get("panorama"), run_dir, pair_idx)

            _save_metrics(metrics, run_dir, pair_idx)
            pair_idx += 1

            if pair_idx % 25 == 0:
                log.info(
                    "  %d pairs processed  (valid %d, skipped %d)",
                    pair_idx, n_valid, n_skipped,
                )

    log.info(
        "%s complete: %d pairs total, %d valid, %d skipped → %s",
        tag, pair_idx, n_valid, n_skipped, run_dir,
    )
    return run_dir

_FL_SCRIPT = Template("""\
#!/usr/bin/env python3
import json, logging, os, sys
import numpy as _np
from pathlib import Path

ROOT = Path(r"$root")
sys.path.insert(0, str(ROOT))

os.environ.setdefault("RAY_memory_usage_threshold", "0.95")
os.environ.setdefault("RAY_memory_monitor_refresh_ms", "0")

_pp = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = str(ROOT) + (os.pathsep + _pp if _pp else "")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("fl_sim")

try:
    import flwr as fl
    from flwr.simulation import run_simulation
    from flwr.client import ClientApp
    log.info("flwr version: %s", fl.__version__)
except ImportError:
    sys.exit("ERROR: flwr not installed in this Python. "
             "Activate the venv and run: pip install flwr")

from client.client_app import create_client
from server.server_app import build_server_app

NUM_CLIENTS = $num_clients
NUM_ROUNDS  = $num_rounds
CLIENTS_DIR = ROOT / "data" / "clients"
OUTPUT_DIR  = Path(r"$output_dir")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model_cfg = dict(
    name="loftr",
    checkpoint_path=None,
    device=None,
    confidence_threshold=0.1,
    lora=dict(enabled=True, rank=4, alpha=8.0,
              target_modules=["q_proj", "v_proj"]),
)
training_cfg = dict(
    local_epochs=1,
    learning_rate=1e-4,
    weight_decay=1e-4,
    batch_size=2,
)

def client_fn(cid: str) -> fl.client.Client:
    client_dir = str(CLIENTS_DIR / f"client_{cid}")
    config = dict(
        model=model_cfg,
        data=dict(
            data_dir=client_dir,
            train_csv="pairs.csv",
            val_csv="pairs.csv",
        ),
        training=training_cfg,
    )
    return create_client(cid, config).to_client()

round_losses: list = []

def fit_metrics_agg(metrics):
    losses = [m.get("train_loss", m.get("loss", 0.0)) for _, m in metrics]
    avg = sum(losses) / len(losses) if losses else 0.0
    round_losses.append(avg)
    log.info("Round %d avg train_loss=%.4f", len(round_losses), avg)
    return {"train_loss": avg}

fl_config = dict(
    model=model_cfg,
    training=training_cfg,
    num_rounds=NUM_ROUNDS,
    output_dir=str(OUTPUT_DIR),
    strategy=dict(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=min(NUM_CLIENTS, 2),
        min_available_clients=NUM_CLIENTS,
    ),
)
server_app = build_server_app(fl_config, fit_metrics_aggregation_fn=fit_metrics_agg)

_BACKEND_CONFIG = {
    "client_resources": {"num_cpus": 4, "num_gpus": 0},
    "init_args": {
        "num_cpus": 4,
        "num_gpus": 0,
        "object_store_memory": 1_500_000_000,
        "log_to_driver": True,
    },
}

log.info(
    "Starting FL simulation: %d clients x %d rounds  "
    "[sequential: 1 client at a time, object_store=1.5 GB]",
    NUM_CLIENTS, NUM_ROUNDS,
)

client_app = ClientApp(client_fn=client_fn)

try:
    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=NUM_CLIENTS,
        backend_config=_BACKEND_CONFIG,
    )
except Exception as exc:
    log.error("FL simulation failed: %s", exc)
    with open(OUTPUT_DIR / "fl_error.txt", "w") as f:
        import traceback
        traceback.print_exc(file=f)
    sys.exit(1)

out = dict(
    num_rounds=NUM_ROUNDS,
    num_clients=NUM_CLIENTS,
    losses_distributed=list(enumerate(round_losses, start=1)),
    metrics_distributed={},
)
metrics_path = OUTPUT_DIR / "fl_round_metrics.json"
with open(metrics_path, "w") as fh:
    json.dump(out, fh, indent=2, default=str)

log.info("FL simulation complete. Metrics saved to %s", metrics_path)

print("\\n--- FL Round Summary ---")
for rnd, loss in enumerate(round_losses, start=1):
    print(f"  Round {rnd:2d}: loss = {loss:.4f}")
""")

def phase_federated(num_clients: int, fl_rounds: int, base_stitch_cfg: dict) -> Path:
    log.info("=" * 60)
    log.info("Phase 4: Federated Learning  (%d clients, %d rounds)", num_clients, fl_rounds)
    log.info("=" * 60)

    run_dir = _make_run_dir("pipeline_federated")

    if not _VENV_PY.exists():
        log.warning(
            "venv not found at %s — cannot run FL phase.\n"
            "  Fix: python -m venv venv && source venv/bin/activate && pip install -r requirements.txt",
            _VENV_PY,
        )
        return run_dir

    script_src = _FL_SCRIPT.substitute(
        root=str(_ROOT),
        num_clients=num_clients,
        num_rounds=fl_rounds,
        output_dir=str(run_dir),
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix="_fl_sim.py", delete=False, dir=str(_ROOT / "outputs"),
    ) as tmp:
        tmp.write(script_src)
        tmp_path = tmp.name

    try:
        log.info("Spawning FL subprocess via %s …", _VENV_PY)
        result = subprocess.run(
            [str(_VENV_PY), tmp_path],
            cwd=str(_ROOT),
            timeout=3600,
        )
        if result.returncode == 0:
            log.info("FL simulation succeeded → %s", run_dir)
            phase_post_fl_inference(run_dir, base_stitch_cfg)
        else:
            log.warning(
                "FL simulation exited with code %d. "
                "Check %s/fl_error.txt for the traceback.",
                result.returncode, run_dir,
            )
    except subprocess.TimeoutExpired:
        log.warning("FL simulation timed out after 3600 s.")
    except Exception as exc:
        log.warning("FL subprocess failed to start: %s", exc)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return run_dir

def phase_post_fl_inference(fl_run_dir: Path, base_stitch_cfg: dict) -> Path | None:
    log.info("=" * 60)
    log.info("Phase 4b: Post-FL Inference  (LoFTR + LoRA fine-tuned)")
    log.info("=" * 60)

    weights_path = fl_run_dir / "final_lora_weights.npz"
    if not weights_path.exists():
        log.warning(
            "final_lora_weights.npz not found in %s — skipping post-FL inference.",
            fl_run_dir,
        )
        return None

    import copy
    import numpy as np
    from client.dataset import load_image, load_pairs_from_csv
    from core.io import save_config
    from core.stitching_pipeline import stitch_pair
    from models.loftr_matcher import LoFTRMatcher

    matcher = LoFTRMatcher({"confidence_threshold": 0.1})
    matcher.enable_lora(rank=4, alpha=8.0, target_modules=["q_proj", "v_proj"])

    data = np.load(str(weights_path))
    params = [data[k] for k in sorted(data.files)]
    matcher.set_lora_parameters(params)
    log.info("Loaded %d LoRA weight arrays from %s", len(params), weights_path)

    run_dir = _make_run_dir("pipeline_loftr_fl")

    stitch_cfg = copy.deepcopy(base_stitch_cfg)
    stitch_cfg["matcher"] = {"name": "loftr_fl", "confidence_threshold": 0.1}
    save_config(stitch_cfg, str(run_dir / "used_config.yaml"))

    client_dirs = sorted(_CLIENTS.glob("client_*"))
    if not client_dirs:
        log.error("No client directories found in %s.", _CLIENTS)
        return run_dir

    pair_idx = 0
    n_valid  = 0
    n_skipped = 0

    for client_dir in client_dirs:
        if not (client_dir / "pairs.csv").exists():
            log.warning("No pairs.csv in %s — skipping.", client_dir)
            continue

        try:
            pairs = load_pairs_from_csv(str(client_dir), "pairs.csv")
        except Exception as exc:
            log.warning("Failed to load pairs from %s: %s", client_dir, exc)
            continue

        log.info("Processing %d pairs from %s …", len(pairs), client_dir.name)

        for pair in pairs:
            try:
                img_a = load_image(pair["image_a_path"])
                img_b = load_image(pair["image_b_path"])
            except FileNotFoundError:
                n_skipped += 1
                pair_idx  += 1
                continue

            try:
                metrics = stitch_pair(img_a, img_b, matcher, stitch_cfg)
            except Exception as exc:
                log.debug("stitch_pair failed (pair %d): %s", pair_idx, exc)
                metrics = {
                    "num_matches": 0,
                    "num_inliers": 0,
                    "homography_valid": False,
                    "reprojection_error": float("inf"),
                }

            if metrics.get("homography_valid"):
                n_valid += 1
                _save_panorama(metrics.get("panorama"), run_dir, pair_idx)

            _save_metrics(metrics, run_dir, pair_idx)
            pair_idx += 1

            if pair_idx % 25 == 0:
                log.info(
                    "  %d pairs processed  (valid %d, skipped %d)",
                    pair_idx, n_valid, n_skipped,
                )

    log.info(
        "loftr_fl complete: %d pairs total, %d valid, %d skipped → %s",
        pair_idx, n_valid, n_skipped, run_dir,
    )
    return run_dir

def phase_compare() -> None:
    log.info("=" * 60)
    log.info("Phase 5: Compare Results")
    log.info("=" * 60)

    from experiments.compare_results import main as compare_main
    compare_main(["--scan", str(_RUNS)])

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="End-to-end panoravn pipeline orchestrator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=ALL_PHASES,
        default=ALL_PHASES,
        metavar="PHASE",
        help="Run only these phases (default: all). Choices: " + " ".join(ALL_PHASES),
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        choices=ALL_PHASES,
        default=[],
        metavar="PHASE",
        help="Skip these phases.",
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=None,
        help=(
            "Number of FL client partitions. "
            "Defaults to the number of drone_* directories found in data/sim_frames/."
        ),
    )
    parser.add_argument(
        "--fl-rounds",
        type=int,
        default=3,
        help="Number of Flower FL rounds (default: 3).",
    )
    args = parser.parse_args(argv)

    phases = [p for p in args.phases if p not in args.skip]
    if not phases:
        log.error("No phases left to run after applying --skip.")
        return 1

    if args.num_clients is not None:
        num_clients = args.num_clients
    else:
        n_drones = _detect_num_drones(_SIM_FRAMES)
        num_clients = n_drones if n_drones > 0 else 4
        log.info(
            "Auto-detected %d drone_* director%s in %s → using %d client(s).",
            num_clients,
            "y" if num_clients == 1 else "ies",
            _SIM_FRAMES,
            num_clients,
        )

    log.info("Panoravn pipeline — phases: %s", " → ".join(phases))
    log.info("Project root: %s", _ROOT)
    log.info("Clients: %d  (sim_frames=%s  clients=%s)", num_clients, _SIM_FRAMES, _CLIENTS)
    log.info("Output: %s", _RUNS)

    from core.io import load_config
    base_stitch_cfg = load_config(str(_ROOT / "configs" / "baseline.yaml"))

    sift_cfg = {
        "name": "sift",
        "type": "classical",
        "detector": "sift",
        "ratio_test": 0.75,
        "cross_check": False,
    }
    loftr_cfg = {
        "name": "loftr",
        "checkpoint_path": None,
        "device": None,
        "confidence_threshold": 0.1,
    }

    if "partition" in phases:
        phase_partition(num_clients)

    if "baseline" in phases:
        phase_batch_run(sift_cfg, base_stitch_cfg, "sift")

    if "neural" in phases:
        phase_batch_run(loftr_cfg, base_stitch_cfg, "loftr")

    if "federated" in phases:
        phase_federated(num_clients, args.fl_rounds, base_stitch_cfg)

    if "compare" in phases:
        phase_compare()

    log.info("Pipeline complete.")
    return 0

if __name__ == "__main__":
    sys.exit(main())

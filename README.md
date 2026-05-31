# Panoravn

> Research system for privacy-preserving aerial panorama stitching using federated learning across a drone swarm.

---

## Author

- **Name**: Liliia Lyashkevych
- **Group**: FeP-43
- **Supervisor**: Roman Shuvar
- **Date**: 31.05.2026

---

## General Information

- **Project type**: Research / Python system
- **Programming language**: Python 3.12 / 3.13
- **Frameworks / Libraries**: PyTorch, Kornia (LoFTR), Flower (flwr), Ray, OpenCV, scikit-image, pyzmq, PyYAML, Webots R2023b

---

## System Description

Panoravn combines drone swarm simulation, real-time image ingestion, classical and deep-learning-based image matching, and federated learning (FL) to produce aerial panoramas. The core privacy constraint: raw images never leave the client node — only LoRA adapter weight updates (~160 KB per round) are transmitted.

The pipeline operates in five sequential phases:

| Phase | Name | Description |
|-------|------|-------------|
| 1 | Partition | Drone footage split into per-client datasets |
| 2 | Baseline | Classical SIFT matching on all image pairs |
| 3 | Neural | Pre-trained LoFTR matching on all pairs |
| 4 | Federated | LoFTR fine-tuned via FL with LoRA adapters |
| 4b | Post-FL | Fine-tuned LoFTR+LoRA inference |
| 5 | Compare | Quantitative evaluation across all runs |

---

## Architecture

The system is divided into seven layers with strict isolation boundaries. Lower layers have no knowledge of upper layers.

```
Layer 1: simulation/    — Webots drone swarm, procedural world generation
Layer 2: bridge/        — Frame ingestion from simulation (ZMQ or disk)
Layer 3: models/        — Matcher interface, SIFT/ORB, LoFTR, LoRA
Layer 4: core/          — Shared stitching pipeline (geometry, blending, evaluation)
Layer 5: client/        — Flower FL client, local training, privacy enforcement
Layer 6: server/        — Flower FL server, FedAvg strategy, checkpointing
Layer 7: experiments/   — Pipeline orchestration, data partitioning, comparison
```

---

## Key Files

| File | Purpose |
|------|---------|
| `experiments/run_pipeline.py` | Master orchestrator — runs all phases end-to-end |
| `experiments/split_clients.py` | Partitions sim frames into per-client datasets with pairs.csv |
| `experiments/compare_results.py` | Aggregates and prints metrics across all runs |
| `simulation/worlds/generate_world.py` | Generates Webots world file procedurally (Perlin terrain) |
| `simulation/worlds/drone_swarm.wbt` | Webots world definition used by the simulator |
| `simulation/controllers/swarm_supervisor/swarm_supervisor.py` | Injects drones, computes lawnmower waypoints |
| `simulation/controllers/mavic_controller/mavic_controller.py` | Per-drone flight controller, camera capture, ZMQ/disk export |
| `bridge/ingestor.py` | ZMQ PULL receiver and disk watcher; emits FrameRecord objects |
| `bridge/sequencer.py` | Timestamp-sorted frame buffer; emits qualifying image pairs |
| `bridge/telemetry.py` | Camera footprint geometry, overlap ratio, GPS-to-homography prior |
| `bridge/sim_pipeline.py` | CLI entry point for live bridge operation |
| `models/base_matcher.py` | Abstract `MatcherInterface` — `match(img_a, img_b)` |
| `models/classical_matcher.py` | SIFT/ORB matching with Lowe's ratio test |
| `models/loftr_matcher.py` | LoFTR dense matching, self-supervised loss, LoRA integration |
| `models/lora.py` | LoRALinear wrapper, injection, weight serialisation |
| `models/model_factory.py` | Factory: `create_matcher(config)` |
| `core/stitching_pipeline.py` | `stitch_pair()` — preprocessing, matching, homography, warp, blend, evaluate |
| `core/geometry.py` | RANSAC homography, validation (5-gate), reprojection error, warp |
| `core/blending.py` | Alpha blending and multi-band (Laplacian pyramid) blending |
| `core/evaluation.py` | SSIM and PSNR computation |
| `client/client_app.py` | `PanoramaMatcherClient` — Flower NumPyClient |
| `client/local_training.py` | Local Adam training loop with per-epoch eval |
| `client/dataset.py` | `FederatedMatcherDataset`, pairs.csv loader, dataloader factory |
| `client/privacy.py` | Recursive payload validator — blocks str/bytes at client boundary |
| `server/strategies.py` | `LoRAFedAvg` — FedAvg with checkpointing hook |
| `server/server_app.py` | `build_server_app()` factory — creates ServerApp |
| `server/model_registry.py` | Per-round .pth checkpoints, final_lora_weights.npz export |
| `configs/baseline.yaml` | Preprocessing, geometry thresholds, blending, evaluation settings |
| `configs/federated.yaml` | FL model, FedAvg strategy, training hyperparameters |
| `configs/loftr.yaml` | LoFTR confidence threshold and checkpoint path |
| `simulation/configs/simulation.yaml` | Drone count, altitude, FOV, transport mode, pairing parameters |

---

## Full Setup and Run Guide

The complete workflow has four stages: environment setup, world generation, simulation, and pipeline execution. Follow all steps in order on a fresh machine.

### Stage 1 — Environment Setup

**Step 1. Install prerequisites**

- [Python 3.13](https://www.python.org/) (system / Miniconda) — used by phases 1–3, 5 and tests
- [Python 3.12](https://www.python.org/) — used by the FL subprocess (Ray does not support 3.13)
- [Webots R2023b](https://cyberbotics.com/) — the robotics simulator; install from the official site

**Step 2. Clone the repository**

```bash
git clone https://github.com/your-user/panoravn.git
cd panoravn
```

**Step 3. Install system Python dependencies**

```bash
pip install -r requirements.txt
```

**Step 4. Create the FL virtual environment**

The federated learning phase (phase 4) and the ZMQ bridge require a separate Python 3.12 environment with Flower and Ray:

```bash
python3.12 -m venv venv
venv/bin/pip install flwr ray torch kornia opencv-python numpy pyzmq
```

---

### Stage 2 — Webots World Generation

The repository already includes a ready-to-use world file at `simulation/worlds/drone_swarm.wbt`. This step is only needed if you want to regenerate the terrain (e.g. after changing map parameters in `generate_world.py`).

**Step 5. (Optional) Regenerate the Webots world file**

```bash
python simulation/worlds/generate_world.py
```

This writes a new `drone_swarm.wbt` with a procedurally generated 800×800 m terrain (Perlin fBm hills, road corridor, forest patches, building clusters). The script prints the coordinate bounds and suggested `simulation.yaml` values on completion.

Key constants inside `generate_world.py` you may want to adjust before regenerating:

| Constant | Default | Meaning |
|----------|---------|---------|
| `GRID_N` | 201 | ElevationGrid resolution (201×201 cells) |
| `MAP_HALF` | 400 | Half-extent of the map in metres |
| `HILL_AMP` | 12 | Maximum terrain hill height in metres |

---

### Stage 3 — Running the Simulation

There are two transport modes. **Disk mode** is recommended for reproducibility; ZMQ mode is for real-time processing while the simulation runs.

**Step 6a. Disk mode (recommended)**

Open Webots R2023b and load the world:

```
File → Open World → simulation/worlds/drone_swarm.wbt
```

Before starting, verify `simulation/configs/simulation.yaml` has `transport.mode: disk`. Then press **Play** in Webots. Each drone controller will write captured frames as JPEG + JSON telemetry pairs into:

```
data/sim_frames/drone_N/processed/ok_<drone_id>_<frame_id>.jpg
data/sim_frames/drone_N/processed/ok_<drone_id>_<frame_id>.json
```

Wait until all drones complete their lawnmower survey and the Webots console prints `[supervisor] All drones complete. Pausing simulation.`

**Step 6b. ZMQ mode (live bridge, alternative)**

Set `transport.mode: zmq` in `simulation/configs/simulation.yaml`, then start the bridge in one terminal and Webots in another:

```bash
# Terminal 1 — start the bridge first
python bridge/sim_pipeline.py \
    --config simulation/configs/simulation.yaml \
    --transport zmq

# Terminal 2 — then start Webots and press Play
```

The bridge binds a ZMQ PULL socket on `tcp://localhost:5555` (configurable via `transport.zmq_endpoint`) and stitches image pairs in real time as they arrive.

---

### Stage 4 — Pipeline Execution

**Step 7. Partition frames into per-client datasets**

This step is handled automatically by `run_pipeline.py` (phase 1), but can also be run standalone:

```bash
python experiments/split_clients.py \
    --input data/sim_frames \
    --output data/clients \
    --num_clients 4
```

Each client gets a subdirectory under `data/clients/client_N/` containing symlinks to its assigned frames and a `pairs.csv` file listing all consecutive image pairs for that drone strip.

**Step 8. Run the full pipeline**

```bash
python experiments/run_pipeline.py
```

By default this runs all five phases with 4 clients and 3 FL rounds. Use flags to customise:

```bash
python experiments/run_pipeline.py \
    --phases partition baseline neural federated post_fl compare \
    --num-clients 4 \
    --fl-rounds 5
```

To skip one or more phases (e.g. if frames are already partitioned):

```bash
python experiments/run_pipeline.py --skip partition
```

Available phase names: `partition`, `baseline`, `neural`, `federated`, `post_fl`, `compare`.

**Step 9. Run the test suite**

```bash
pytest
```

---

## Python Environment Split

The project uses two Python environments due to a Ray / Python 3.13 incompatibility:

| Environment | Python | Used for |
|-------------|--------|----------|
| System (Miniconda) | 3.13 | Phases 1, 2, 3, 5; pytest |
| `venv/` | 3.12 | Phase 4 FL subprocess; ZMQ bridge |

`run_pipeline.py` automatically invokes `venv/bin/python` for the FL subprocess. No manual switching is required.

---

## Outputs

Results are written to `outputs/runs/` with timestamped directories:

| File | Content |
|------|---------|
| `metrics_NNNNN.json` | Per-pair stitching metrics: inliers, reprojection error, SSIM, PSNR |
| `panorama_NNNNN.jpg` | Stitched panorama image |
| `lora_round_NNN.pth` | Per-round LoRA checkpoint (state dict + metadata) |
| `final_lora_weights.npz` | Final adapter weights used by phase 4b inference |
| `federated_metrics.json` | Per-round FL training loss |
| `fl_round_metrics.json` | Round-by-round loss summary written by the FL subprocess |

---

## Federated Learning Protocol

- **Algorithm**: FedAvg (McMahan et al., 2017) via Flower
- **Parameter exchange**: LoRA adapters only (~160 KB/round vs. 84 MB full model)
- **Privacy**: `client/privacy.py` blocks any `str` or `bytes` from leaving the client
- **Loss**: Self-supervised from LoFTR's internal confidence matrix — no ground-truth labels needed
- **Default hyperparameters**: Adam, lr=1e-4, weight_decay=1e-4, batch size=2, 1 local epoch/round

---

## Known Issues and Solutions

| Problem | Solution |
|---------|---------|
| Ray does not support Python 3.13 | Phase 4 runs automatically in `venv/bin/python` (Python 3.12); no manual action needed |
| `flwr` or `zmq` not found in system Python | Install them only inside `venv/` — system Python intentionally does not have them |
| Canvas size error during warping | Degenerate homography; increase `min_inliers` or `ransac_threshold` in `configs/baseline.yaml` |
| LoFTR `KeyError` in coarse matching during training | Expected behaviour — `loftr_matcher.py` keeps coarse/fine modules in eval mode during `fit()` |
| Webots not available | Place drone footage manually in `data/sim_frames/drone_N/processed/` and run from step 7 |
| ZMQ frames being dropped | Normal under high load (ADR-001: non-blocking send); reduce drone speed or increase `zmq_sndhwm` in `simulation.yaml` |

---

## Sources

- Flower (flwr) documentation — https://flower.ai/docs/
- LoFTR: Detector-Free Local Feature Matching — Sun et al., CVPR 2021
- LoRA: Low-Rank Adaptation of Large Language Models — Hu et al., 2021
- FedAvg: Communication-Efficient Learning of Deep Networks — McMahan et al., 2017
- Kornia computer vision library — https://kornia.readthedocs.io/
- OpenCV documentation — https://docs.opencv.org/
- Webots robotics simulator — https://cyberbotics.com/
- PyTorch documentation — https://pytorch.org/docs/

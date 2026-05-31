from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path
from typing import NamedTuple

class FrameEntry(NamedTuple):
    src_path: Path
    frame_index: int

class SceneDescriptor(NamedTuple):
    name: str
    frames: list[FrameEntry]

_JPEG_EXTS = {".jpg", ".jpeg"}
_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

def _is_sim_directory(path: Path) -> bool:
    for child in path.iterdir():
        if not child.is_dir():
            continue
        candidates = [child, child / "processed"]
        for candidate in candidates:
            if candidate.is_dir():
                for f in candidate.iterdir():
                    if f.suffix.lower() in _JPEG_EXTS and (
                        f.name.startswith("frame_")
                        or f.name.startswith("ok_")
                        or f.name.startswith("rejected_")
                    ):
                        return True
    return False

def _frame_index_from_name(name: str) -> int:
    stem = Path(name).stem
    for token in reversed(stem.split("_")):
        if token.isdigit():
            return int(token)
    return 0

def _collect_sim_scenes(input_dir: Path) -> list[SceneDescriptor]:
    scenes: list[SceneDescriptor] = []
    for drone_dir in sorted(input_dir.iterdir()):
        if not drone_dir.is_dir():
            continue
        frames: list[FrameEntry] = []
        search_dirs = [drone_dir]
        processed = drone_dir / "processed"
        if processed.is_dir():
            search_dirs.append(processed)
        for search in search_dirs:
            for f in search.iterdir():
                if f.suffix.lower() in _JPEG_EXTS and not f.is_dir():
                    frames.append(FrameEntry(
                        src_path=f.resolve(),
                        frame_index=_frame_index_from_name(f.name),
                    ))
        if not frames:
            continue
        frames.sort(key=lambda e: e.frame_index)
        scenes.append(SceneDescriptor(name=f"scene_{drone_dir.name}", frames=frames))
    return scenes

def _collect_scene_scenes(input_dir: Path) -> list[SceneDescriptor]:
    scenes: list[SceneDescriptor] = []
    for scene_dir in sorted(input_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        images = sorted(
            f for f in scene_dir.iterdir()
            if f.suffix.lower() in _IMAGE_EXTS and f.is_file()
        )
        if not images:
            continue
        frames = [
            FrameEntry(src_path=f.resolve(), frame_index=i)
            for i, f in enumerate(images)
        ]
        scenes.append(SceneDescriptor(name=scene_dir.name, frames=frames))
    return scenes

def _assign_scenes_to_clients(
    scenes: list[SceneDescriptor],
    num_clients: int,
) -> list[list[SceneDescriptor]]:
    buckets: list[list[SceneDescriptor]] = [[] for _ in range(num_clients)]
    for i, scene in enumerate(scenes):
        buckets[i % num_clients].append(scene)
    return buckets

def _link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copy2(src, dst)
    else:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)

def _build_client(
    client_dir: Path,
    scenes: list[SceneDescriptor],
    copy: bool,
) -> int:
    client_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = client_dir / "pairs.csv"

    total_pairs = 0
    with pairs_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["image_a", "image_b", "scene_id"])
        writer.writeheader()

        for scene in scenes:
            scene_out = client_dir / scene.name
            scene_out.mkdir(parents=True, exist_ok=True)

            renamed: list[str] = []
            for entry in scene.frames:
                dst_name = f"img_{entry.frame_index:06d}.jpg"
                _link_or_copy(entry.src_path, scene_out / dst_name, copy)
                renamed.append(dst_name)

            for a_name, b_name in zip(renamed, renamed[1:]):
                writer.writerow({
                    "image_a":  a_name,
                    "image_b":  b_name,
                    "scene_id": scene.name,
                })
                total_pairs += 1

    return total_pairs

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Partition drone frames or scenes into per-client FL datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Source directory (data/sim_frames for sim mode, data/raw for scene mode).",
    )
    parser.add_argument(
        "--output", "-o",
        default="data/clients",
        help="Destination root directory (default: data/clients).",
    )
    parser.add_argument(
        "--num_clients", "-n",
        type=int,
        default=None,
        help=(
            "Number of client partitions to create. "
            "Defaults to the number of drone_* directories found in the input path (SIM mode) or 4."
        ),
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of creating symlinks (slower but self-contained).",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "sim", "scene"],
        default="auto",
        help="Force a specific input format (default: auto-detect).",
    )
    args = parser.parse_args(argv)

    input_dir  = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()

    if not input_dir.is_dir():
        print(f"ERROR: input directory does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)

    mode = args.mode
    if mode == "auto":
        mode = "sim" if _is_sim_directory(input_dir) else "scene"

    if args.num_clients is not None:
        num_clients = args.num_clients
    elif mode == "sim":
        num_clients = sum(
            1 for d in input_dir.iterdir()
            if d.is_dir() and d.name.startswith("drone_")
        ) or 4
        print(f"Auto-detected {num_clients} drone_* director{'y' if num_clients == 1 else 'ies'} → {num_clients} client(s).")
    else:
        num_clients = 4

    if num_clients < 1:
        print("ERROR: --num_clients must be >= 1", file=sys.stderr)
        sys.exit(1)

    print(f"Input:    {input_dir}  [{mode.upper()} mode]")
    print(f"Output:   {output_dir}")
    print(f"Clients:  {num_clients}")
    print(f"Strategy: {'copy' if args.copy else 'symlink'}")
    print()

    if mode == "sim":
        scenes = _collect_sim_scenes(input_dir)
    else:
        scenes = _collect_scene_scenes(input_dir)

    if not scenes:
        print("WARNING: no frames or scenes found in the input directory.", file=sys.stderr)
        print("  SIM mode: expects subdirectories containing frame_*.jpg or ok_*.jpg files.")
        print("  SCENE mode: expects subdirectories containing .jpg/.png image files.")
        sys.exit(0)

    total_frames = sum(len(s.frames) for s in scenes)
    print(f"Found {len(scenes)} scene(s) / drone strip(s), {total_frames} total frames.")

    buckets = _assign_scenes_to_clients(scenes, num_clients)
    for idx, client_scenes in enumerate(buckets):
        client_dir = output_dir / f"client_{idx}"
        n_pairs = _build_client(client_dir, client_scenes, copy=args.copy)
        n_frames = sum(len(s.frames) for s in client_scenes)
        n_scenes = len(client_scenes)
        print(
            f"  client_{idx}: {n_scenes} scene(s), "
            f"{n_frames} frame(s), {n_pairs} pair(s) → {client_dir}"
        )

    print("\nDone.")

if __name__ == "__main__":
    main()

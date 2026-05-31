from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

def _load_metrics_from_dir(run_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for f in sorted(run_dir.glob("metrics_*.json")):
        try:
            with f.open() as fh:
                records.append(json.load(fh))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  WARNING: skipping {f.name}: {exc}", file=sys.stderr)
    return records

def _read_matcher_from_config(run_dir: Path) -> str:
    config_path = run_dir / "used_config.yaml"
    if not config_path.exists():
        return "unknown"
    try:
        import yaml
        with config_path.open() as fh:
            cfg = yaml.safe_load(fh)
        return cfg.get("matcher", {}).get("name", "unknown")
    except Exception:
        return "unknown"

def _discover_run_dirs(scan_root: Path) -> list[Path]:
    dirs: list[Path] = []
    for candidate in sorted(scan_root.rglob("metrics_*.json")):
        parent = candidate.parent
        if parent not in dirs:
            dirs.append(parent)
    return dirs

def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "n_pairs": 0,
            "success_rate": float("nan"),
            "mean_matches": float("nan"),
            "mean_inliers": float("nan"),
            "mean_inlier_ratio": float("nan"),
            "mean_reprojection_error": float("nan"),
            "mean_ssim": float("nan"),
            "mean_psnr": float("nan"),
        }

    n = len(records)

    def _mean(key: str, fallback: float = float("nan")) -> float:
        vals = [
            r[key] for r in records
            if key in r and r[key] is not None and r[key] != float("inf")
        ]
        if not vals:
            return fallback
        return sum(vals) / len(vals)

    n_success = sum(1 for r in records if r.get("homography_valid", False))
    matches    = [r["num_matches"]  for r in records if "num_matches"  in r]
    inliers    = [r["num_inliers"]  for r in records if "num_inliers"  in r]

    inlier_ratios = []
    for r in records:
        nm = r.get("num_matches", 0)
        ni = r.get("num_inliers", 0)
        if nm and nm > 0:
            inlier_ratios.append(ni / nm)

    return {
        "n_pairs":               n,
        "success_rate":          n_success / n,
        "mean_matches":          sum(matches)  / len(matches)  if matches  else float("nan"),
        "mean_inliers":          sum(inliers)  / len(inliers)  if inliers  else float("nan"),
        "mean_inlier_ratio":     sum(inlier_ratios) / len(inlier_ratios) if inlier_ratios else float("nan"),
        "mean_reprojection_error": _mean("reprojection_error"),
        "mean_ssim":             _mean("ssim"),
        "mean_psnr":             _mean("psnr"),
    }

_HEADER = (
    "Experiment", "Matcher", "Pairs",
    "Success%", "Matches", "Inliers", "Inlier%",
    "Reproj.Err", "SSIM", "PSNR",
)

_COL_WIDTHS = (28, 9, 6, 9, 9, 9, 8, 11, 7, 7)

def _fmt(val: Any, fmt: str = "", na: str = "n/a") -> str:
    if isinstance(val, float) and (val != val):
        return na
    if isinstance(val, float) and val == float("inf"):
        return "∞"
    try:
        return format(val, fmt)
    except (ValueError, TypeError):
        return str(val)

def _row(cells: tuple) -> str:
    parts = []
    for cell, width in zip(cells, _COL_WIDTHS):
        parts.append(str(cell).ljust(width)[:width])
    return "  ".join(parts)

def _print_table(rows: list[tuple], title: str = "") -> None:
    sep = "  ".join("─" * w for w in _COL_WIDTHS)
    if title:
        total = sum(_COL_WIDTHS) + 2 * (len(_COL_WIDTHS) - 1)
        print(f"\n{'═' * total}")
        print(title.center(total))
        print(f"{'═' * total}")
    print(_row(_HEADER))
    print(sep)
    for row in rows:
        print(_row(row))
    print()

def _build_row(label: str, matcher: str, stats: dict[str, Any]) -> tuple:
    n   = stats["n_pairs"]
    suc = _fmt(stats["success_rate"] * 100, ".1f") + "%" if n else "n/a"
    return (
        label[:28],
        matcher[:9],
        str(n),
        suc,
        _fmt(stats["mean_matches"],            ".1f"),
        _fmt(stats["mean_inliers"],             ".1f"),
        _fmt(stats["mean_inlier_ratio"] * 100,  ".1f") + "%" if n else "n/a",
        _fmt(stats["mean_reprojection_error"],  ".3f"),
        _fmt(stats["mean_ssim"],                ".4f"),
        _fmt(stats["mean_psnr"],                ".2f"),
    )

def _save_comparison(entries: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment", "matcher", "n_pairs", "success_rate",
        "mean_matches", "mean_inliers", "mean_inlier_ratio",
        "mean_reprojection_error", "mean_ssim", "mean_psnr",
    ]
    with output_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            writer.writerow({k: entry.get(k, "") for k in fieldnames})
    print(f"Comparison saved to {output_path}")

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare stitching quality metrics across experiment runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--baseline",
        metavar="DIR",
        help="Run directory for the classical baseline (SIFT/ORB).",
    )
    parser.add_argument(
        "--neural",
        metavar="DIR",
        help="Run directory for the centralized neural baseline (LoFTR).",
    )
    parser.add_argument(
        "--federated",
        metavar="DIR",
        help="Run directory for the FL-trained model evaluation.",
    )
    parser.add_argument(
        "--scan",
        metavar="DIR",
        help="Auto-discover all run directories inside DIR.",
    )
    parser.add_argument(
        "--save",
        metavar="FILE",
        help="Save the aggregated comparison to a CSV file.",
    )
    args = parser.parse_args(argv)

    if not any([args.baseline, args.neural, args.federated, args.scan]):
        parser.print_help()
        return 1

    named_experiments: list[tuple[str, str, Path]] = []
    if args.baseline:
        named_experiments.append(("Baseline",  "baseline",  Path(args.baseline)))
    if args.neural:
        named_experiments.append(("Neural",    "neural",    Path(args.neural)))
    if args.federated:
        named_experiments.append(("Federated", "federated", Path(args.federated)))

    scan_experiments: list[tuple[str, str, Path]] = []
    if args.scan:
        scan_root = Path(args.scan)
        if not scan_root.is_dir():
            print(f"ERROR: --scan directory does not exist: {scan_root}", file=sys.stderr)
            return 1
        discovered = _discover_run_dirs(scan_root)
        for d in discovered:
            label = d.name[:28]
            scan_experiments.append((label, "auto", d))
        if not discovered:
            print(f"No run directories with metrics_*.json found under {scan_root}",
                  file=sys.stderr)

    all_experiments = named_experiments + scan_experiments
    if not all_experiments:
        print("Nothing to compare.", file=sys.stderr)
        return 1

    table_rows: list[tuple]       = []
    csv_entries: list[dict]       = []

    for label, tag, run_dir in all_experiments:
        if not run_dir.is_dir():
            print(f"WARNING: directory not found: {run_dir}", file=sys.stderr)
            continue

        records = _load_metrics_from_dir(run_dir)
        matcher = _read_matcher_from_config(run_dir)
        stats   = _aggregate(records)

        print(f"  {label:28s}  {run_dir}  ({len(records)} pairs)")
        table_rows.append(_build_row(label, matcher, stats))
        csv_entries.append({
            "experiment":             label,
            "matcher":                matcher,
            "n_pairs":                stats["n_pairs"],
            "success_rate":           stats["success_rate"],
            "mean_matches":           stats["mean_matches"],
            "mean_inliers":           stats["mean_inliers"],
            "mean_inlier_ratio":      stats["mean_inlier_ratio"],
            "mean_reprojection_error": stats["mean_reprojection_error"],
            "mean_ssim":              stats["mean_ssim"],
            "mean_psnr":              stats["mean_psnr"],
        })

    _print_table(
        table_rows,
        title="Panorama Stitching — Experiment Comparison",
    )

    if len(named_experiments) >= 2 and len(table_rows) >= 2:
        _print_delta_summary(csv_entries[:len(named_experiments)])

    if args.save:
        _save_comparison(csv_entries, Path(args.save))

    return 0

def _print_delta_summary(entries: list[dict]) -> None:
    if len(entries) < 2:
        return
    baseline = entries[0]
    print("Delta vs baseline:")
    print(f"  {'Experiment':<28}  {'Success%':>9}  {'Matches':>9}  {'Inliers':>9}  {'SSIM':>7}")
    print("  " + "─" * 72)
    for entry in entries[1:]:
        def _delta(key: str, scale: float = 1.0) -> str:
            bv = baseline.get(key, float("nan"))
            ev = entry.get(key, float("nan"))
            if bv != bv or ev != ev:
                return "    n/a"
            d = (ev - bv) * scale
            sign = "+" if d >= 0 else ""
            return f"{sign}{d:.2f}"

        suc_b = baseline.get("success_rate", float("nan"))
        suc_e = entry.get("success_rate", float("nan"))
        suc_d = "" if suc_b != suc_b or suc_e != suc_e else (
            f"{'+' if (suc_e-suc_b)>=0 else ''}{(suc_e-suc_b)*100:.1f}pp"
        )
        print(
            f"  {entry['experiment']:<28}  "
            f"{suc_d:>9}  "
            f"{_delta('mean_matches'):>9}  "
            f"{_delta('mean_inliers'):>9}  "
            f"{_delta('mean_ssim'):>7}"
        )
    print()

if __name__ == "__main__":
    sys.exit(main())

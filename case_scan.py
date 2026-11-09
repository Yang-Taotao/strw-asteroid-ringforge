"""
Select parameter-scan cases and merge their impact-scan rows

`select` groups cases by impact parameter: `count / repeats` targets
    with `repeats` nearby realisations per target

`merge` reads the per-case JSON rows
    and draws the mean with min/max error bars at each target
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import main as main_module
import numpy as np
import param_utils


def select_cases(
    scan_dir: Path,
    count: int,
    seed: int,
    out: Path,
    repeats: int,
) -> None:
    """
    Write `count` reproducible cases grouped by impact parameter to `out`

    The table has columns `target_r2`, `index`, and `b_r2`
    Targets are evenly spaced across the successful scan range
        `repeats` nearest unused cases are selected at each target
    """
    if count < 1 or repeats < 1 or count % repeats:
        raise SystemExit("--count must be positive and divisible by --repeats")

    data, _ = param_utils.load_chunks(scan_dir)
    success = np.asarray(data["success"], dtype=bool)
    index = np.asarray(data["index"], dtype=np.int64)
    b = np.asarray(data["b_contact"], dtype=float) / np.asarray(
        data["R_b"], dtype=float
    )
    keep = success & np.isfinite(b)
    if not np.any(keep):
        raise SystemExit(f"no successful cases in {scan_dir}")

    available_b = b[keep]
    available_idx = index[keep]
    groups = count // repeats
    targets = np.linspace(available_b.min(), available_b.max(), groups)
    rng = np.random.default_rng(seed)

    spacing = (available_b.max() - available_b.min()) / max(groups - 1, 1)
    targets += rng.uniform(-0.1, 0.1, groups) * spacing
    targets = np.clip(targets, available_b.min(), available_b.max())

    chosen: list[tuple[float, int, float]] = []
    unused = np.ones(available_b.size, dtype=bool)
    for target in targets:
        for _ in range(repeats):
            candidates = np.flatnonzero(unused)
            if candidates.size == 0:
                raise SystemExit(f"only {available_b.size} successful cases available")
            pick = candidates[np.argmin(np.abs(available_b[candidates] - target))]
            unused[pick] = False
            chosen.append(
                (float(target), int(available_idx[pick]), float(available_b[pick]))
            )

    chosen.sort(key=lambda row: row[2])
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# target_r2\tindex\tb_r2"]
    lines += [f"{target:.6f}\t{idx}\t{value:.6f}" for target, idx, value in chosen]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"Selected {len(chosen)} cases ({groups} targets x {repeats}) "
        f"from {int(keep.sum())} successful rows -> {out}"
    )


def merge_cases(case_dir: Path, count: int, out: Path, json_out: Path) -> None:
    """Merge per-case json rows into the impact-scan image and data file."""
    paths = sorted(Path(case_dir).glob("case_*.json"))
    if len(paths) != count:
        raise SystemExit(
            f"expected {count} case rows, found {len(paths)} in {case_dir}"
        )

    payloads: list[dict[str, Any]] = [
        json.loads(path.read_text(encoding="utf-8")) for path in paths
    ]
    rows = [payload["impact_row"] for payload in payloads]
    rows.sort(
        key=lambda row: (
            float(row.get("impact_target_r2") or -1.0),
            float(row["impact_parameter_r2"]),
        )
    )

    secondary_radius = float(payloads[0]["secondary_radius_m"])
    bullet_radius = float(payloads[0]["bullet_radius_m"])
    for payload in payloads[1:]:
        if not np.isclose(
            float(payload["secondary_radius_m"]), secondary_radius, rtol=1e-9
        ):
            raise SystemExit("secondary radii differ between cases")
        if not np.isclose(float(payload["bullet_radius_m"]), bullet_radius, rtol=1e-9):
            raise SystemExit("bullet radii differ between cases")

    out.parent.mkdir(parents=True, exist_ok=True)
    main_module.plot_impact_scan(
        rows,
        out,
        secondary_radius=secondary_radius,
        bullet_radius=bullet_radius,
    )
    json_out.parent.mkdir(parents=True, exist_ok=True)
    main_module.atomic_write_json(
        json_out,
        {
            "case_count": len(rows),
            "secondary_radius_m": secondary_radius,
            "bullet_radius_m": bullet_radius,
            "rows": rows,
        },
    )
    print(f"Merged {len(rows)} cases -> {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    pick = sub.add_parser("select", help="pick reproducible successful scan indices")
    pick.add_argument("--scan-dir", type=Path, default=Path("data/param_scan_test"))
    pick.add_argument("--count", type=int, default=24)
    pick.add_argument("--repeats", type=int, default=3)
    pick.add_argument("--seed", type=int, default=42)
    pick.add_argument("--out", type=Path, required=True)

    merge = sub.add_parser("merge", help="plot and save finished case rows")
    merge.add_argument("--case-dir", type=Path, required=True)
    merge.add_argument("--count", type=int, default=24)
    merge.add_argument("--out", type=Path, default=Path("media/impact_scan.png"))
    merge.add_argument(
        "--json-out",
        type=Path,
        default=Path("data/impact_scan_run_cases.json"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "select":
        select_cases(args.scan_dir, args.count, args.seed, args.out, args.repeats)
    else:
        merge_cases(args.case_dir, args.count, args.out, args.json_out)


if __name__ == "__main__":
    main()

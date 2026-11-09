"""
Utils -> Select parameter-scan cases and merge their impact-scan rows
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import main as main_module
import numpy as np


def select_cases(scan_dir: Path, count: int, seed: int, out: Path) -> None:
    """Write `count` reproducible, successful scan idx to `out`"""
    if count < 1:
        raise SystemExit("--count must be positive")

    paths = sorted(Path(scan_dir).glob("chunk_*.npz"))
    if not paths:
        raise SystemExit(f"no chunk_*.npz files under {scan_dir}")

    indices: set[int] = set()
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            success = np.asarray(data["success"], dtype=bool)
            indices.update(int(value) for value in np.asarray(data["index"])[success])
    available = sorted(indices)
    if len(available) < count:
        raise SystemExit(
            f"only {len(available)} successful cases in {scan_dir}; need {count}"
        )

    rng = np.random.default_rng(seed)
    chosen = np.sort(
        rng.choice(np.asarray(available, dtype=np.int64), count, replace=False)
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "\n".join(str(int(value)) for value in chosen) + "\n", encoding="utf-8"
    )
    print(f"Selected {count} of {len(available)} successful scan rows -> {out}")


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
    rows.sort(key=lambda row: float(row["impact_parameter_r2"]))

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
        select_cases(args.scan_dir, args.count, args.seed, args.out)
    else:
        merge_cases(args.case_dir, args.count, args.out, args.json_out)


if __name__ == "__main__":
    main()

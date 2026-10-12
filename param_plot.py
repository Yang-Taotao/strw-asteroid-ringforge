"""
Visualize param_scan results with plots and diagnostics. Work in progress.

CLI
    # typical usage, quick 10000 points scatter from total num of realisations
    python param_plot.py data/run_00 --out media/origins.png --n-scater 10000

    # for only head-on collisions (cos-vrel -0.5 threshold)
    python param_plot.py data/run_00 --out media/head_on_only.png --cos-vrel :-0.5

    # for only rear-end collisions (cos-vrel 0.5 threshold)
    python param_plot.py data/run_00 --out media/rear_end_only.png --cos-vrel 0.5:

    # for sampling diagnostics
    python param_plot.py data/run_00 --diagnostics media/diag.png

"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401

plt.style.use(["science"])

# ==================================================================================== #
# Display formatter
# ==================================================================================== #


def fmt_num(v: object) -> str:
    """
    Format a number for display.

    Parameters
        v
            The value to format.

    Returns
        The formatted text.
    """
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        v = float(v)
        if v == 0.0:
            return "0.0"
        if 1e-3 <= abs(v) < 1e3:
            return f"{v:g}"
        return f"{v:.3e}"
    if isinstance(v, np.ndarray):
        return "[" + ", ".join(fmt_num(x) for x in v.tolist()) + "]"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(fmt_num(x) for x in v) + "]"
    return str(v)


# ==================================================================================== #
# Chunks load
# ==================================================================================== #


def load_chunks(directory: str) -> tuple[dict[str, np.ndarray], int]:
    """
    Join every chunk_*.npz in a directory along axis 0.

    Parameters
        directory
            The directory that holds the chunk files.

    Returns
        A tuple (data, n_files).
        The data dict holds the joined arrays.
        The integer is the number of chunk files.

    Raises
        FileNotFoundError
            If the directory holds no chunk_*.npz files.
    """
    files = sorted(str(p) for p in Path(directory).glob("chunk_*.npz"))
    if not files:
        raise FileNotFoundError(f"no chunk_*.npz files in {directory}")

    scalar_data: dict[str, np.ndarray] = {}
    list_data: dict[str, list[np.ndarray]] = {}

    for f in files:
        with np.load(f, allow_pickle=False) as d:
            for k in d.files:
                arr = d[k]
                if k == "lookback_periods":
                    # Do not join this array. Keep the first copy.
                    scalar_data[k] = arr
                else:
                    list_data.setdefault(k, []).append(arr)

    out: dict[str, np.ndarray] = dict(scalar_data)
    for k, v in list_data.items():
        out[k] = np.concatenate(v, axis=0)
    return out, len(files)


def parse_cos_vrel(spec: str | None) -> tuple[float | None, float | None] | None:
    """
    Read a cos_vrel range from text.

    Examples
        '0.5:' gives (0.5, None).
        ':-0.5' gives (None, -0.5).
        '0.5:1.0' gives (0.5, 1.0).

    Parameters
        spec
            The text from the CLI, or None.

    Returns
        The (lo, hi) pair, or None if spec is None.

    Raises
        ValueError
            If spec holds no ':' character.
    """
    if spec is None:
        return None
    if ":" not in spec:
        raise ValueError("--cos-vrel must contain ':' e.g. '0.5:' or ':-0.5'")
    lo_s, hi_s = spec.split(":")
    lo = float(lo_s) if lo_s else None
    hi = float(hi_s) if hi_s else None
    return lo, hi


def apply_cos_vrel_filter(
    data: dict[str, np.ndarray],
    spec: tuple[float | None, float | None] | None,
) -> tuple[dict[str, np.ndarray], str]:
    """
    Keep only the realisations inside the cos_vrel range.

    Parameters
        data
            The data dict from load_chunks().
        spec
            The (lo, hi) pair from parse_cos_vrel(), or None.

    Returns
        A tuple (filtered_data, label).
    """
    if spec is None:
        return data, "all"
    lo, hi = spec
    cv = data["cos_vrel"]
    mask = data["success"].copy()
    if lo is not None:
        mask &= cv >= lo
    if hi is not None:
        mask &= cv <= hi

    lo_text = lo if lo is not None else "-inf"
    hi_text = hi if hi is not None else "+inf"
    label = f"cos_vrel in ({lo_text}, {hi_text})"
    out = {k: (v if k == "lookback_periods" else v[mask]) for k, v in data.items()}
    return out, label


# ==================================================================================== #
# Figure 1: origins in space and radius
# ==================================================================================== #


def choose_unit(max_meters: float) -> tuple[float, str]:
    """
    Pick a length unit that gives easy axis values.

    Parameters
        max_meters
            The largest length in m.

    Returns
        A tuple (scale_factor, unit_name). The scale factor converts m into that unit.
    """
    for scale, name in [
        (1.0, "m"),
        (1e3, "km"),
        (1e6, "Mm"),
        (1e9, "Gm"),
        (1.496e11, "AU"),
    ]:
        if max_meters / scale < 1e4:
            return scale, name
    return 1.496e11, "AU"


def plot_origins(
    data: dict[str, np.ndarray],
    out_png: str,
    label: str = "all",
    n_scatter: int = 5000,
    seed: int = 0,
) -> None:
    """
    Show the bullet origins in space and in radius.

    Parameters
        data
            The data dict from load_chunks().
        out_png
            The output PNG path.
        label
            A short text for the title.
        n_scatter
            The largest number of points in each xy panel.
        seed
            The RNG seed for the scatter subsample.
    """
    ok = data["success"]
    snaps = data["snapshots"][ok]  # (n, nl, 3, 6)
    v_rel = data["v_rel"][ok]
    lbp = data["lookback_periods"]  # (nl,)
    n_ok = snaps.shape[0]

    if n_ok == 0:
        raise RuntimeError("no successful realisations to plot")

    sep_mc_ma = np.linalg.norm(snaps[:, :, 2, :3] - snaps[:, :, 0, :3], axis=-1)
    rel_pos = snaps[:, :, 2, :3] - snaps[:, :, 0, :3]  # (n, nl, 3) in m
    r_max = np.nanmax(sep_mc_ma)
    scale, unit = choose_unit(float(r_max))

    # Subsample for the scatter. The histograms use all points.
    rng = np.random.default_rng(seed)
    sel = rng.choice(n_ok, size=min(n_scatter, n_ok), replace=False)

    classes = np.unique(v_rel)
    colors = {c: f"C{i}" for i, c in enumerate(classes)}
    labels = {c: f"{fmt_num(c)} m/s" for c in classes}

    n_lb = len(lbp)
    fig, axes = plt.subplots(2, n_lb, figsize=(4 * n_lb, 8), squeeze=False)

    # Row 1: bullet position relative to the primary at eachlookback time.
    for i, lb in enumerate(lbp):
        ax = axes[0, i]
        xy = rel_pos[sel, i, :2] / scale
        cls = v_rel[sel]
        for c in classes:
            m = cls == c
            if not m.any():
                continue
            ax.scatter(
                xy[m, 0],
                xy[m, 1],
                s=1,
                alpha=0.2,
                c=colors[c],
                label=labels[c] if i == 0 else None,
            )
        ax.plot(0, 0, marker="P", color="k", ms=12, alpha=0.8, label="Host binary")
        ax.set_aspect("equal")
        ax.set_title(f"Bullet position at t = -{int(lb)} T_bin")
        ax.set_xlabel(f"x_mc - x_ma  [{unit}]")
        ax.set_ylabel(f"y_mc - y_ma  [{unit}]")
        if i == 0:
            ax.legend(
                loc="lower left",
                title="Bullet impact speed",
                frameon=True,
                edgecolor="black",
                facecolor="white",
                framealpha=0.8,
            )

    # Row 2: histogram of the bullet-primary distance over the full dataset.
    for i, lb in enumerate(lbp):
        ax = axes[1, i]
        r = sep_mc_ma[:, i] / scale

        rmin = max(r[r > 0].min(), 1e-3 * r.max()) if np.any(r > 0) else 1e-3
        rmax = r.max() if r.max() > rmin else rmin * 10
        bins = np.logspace(np.log10(rmin), np.log10(rmax), 100)

        for c in classes:
            m = v_rel == c
            if not m.any():
                continue
            ax.hist(
                r[m],
                bins=bins,
                alpha=0.6,
                color=colors[c],
                label=labels[c] if i == 0 else None,
            )

        ax.set_title(f"Separation at t = -{int(lb)} T_bin")
        ax.set_xlabel(f"Dist. between mc and ma  [{unit}]")
        ax.set_ylabel("Count")
        if i == 0:
            ax.legend(
                loc="lower left",
                title="Bullet impact speed",
                frameon=True,
                edgecolor="black",
                facecolor="white",
                framealpha=0.8,
            )

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}  ({n_ok} cases, axes in {unit})")


# ==================================================================================== #
# Figure 2: sampling diagnostics
# ==================================================================================== #


def plot_diagnostics(
    data: dict[str, np.ndarray],
    out_png: str,
    label: str = "all",
) -> None:
    """
    Show the sampling diagnostics.

    Parameters
        data
            The data dict from load_chunks().
        out_png
            The output PNG path.
        label
            A short text for the title.
    """
    ok = data["success"]
    if not ok.any():
        raise RuntimeError("no successful realisations")

    v_rel = data["v_rel"][ok]
    cos_vrel = data["cos_vrel"][ok]
    cos_contact = data["cos_contact"][ok]
    offset_dir = data["offset_dir"][ok]

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    # Panel (a): realised v_rel.
    ax = axes[0, 0]
    classes = np.unique(v_rel)
    ax.hist(
        v_rel,
        bins=np.arange(classes.min() - 10, classes.max() + 20, 5),
        alpha=0.8,
    )
    ax.set_xlabel("v_rel [m/s]")
    ax.set_ylabel("Count")
    ax.set_title("Realised bullet speed")

    # Panel (b): cos_vrel, the impact geometry against the mb velocity.
    ax = axes[0, 1]
    ax.hist(cos_vrel, bins=100, alpha=0.8)
    ax.axvline(0.5, color="r", ls="--", lw=1, label="rear-end cutoff")
    ax.axvline(-0.5, color="r", ls=":", lw=1, label="head-on  cutoff")
    ax.set_xlabel("cos(v_dir, v_mb)")
    ax.set_ylabel("Count")
    ax.set_title("Approach direction vs mb velocity")
    ax.legend(
        loc="lower left",
        frameon=True,
        edgecolor="black",
        facecolor="white",
        framealpha=0.8,
    )

    # Panel (c): cos_contact, the impact point on mb.
    ax = axes[1, 0]
    ax.hist(cos_contact, bins=100, alpha=0.8)
    ax.set_xlabel("cos(offset_dir, v_mb)")
    ax.set_ylabel("Count")
    ax.set_title("Contact point on mb vs its velocity")

    # Panel (d): sky map of offset_dir.
    ax = axes[1, 1]
    sc = ax.scatter(
        offset_dir[:, 0],
        offset_dir[:, 1],
        s=2,
        alpha=0.3,
        c=cos_vrel,
        cmap="bwr",
    )
    ax.set_aspect("equal")
    ax.set_xlabel("Offset_dir_x")
    ax.set_ylabel("Offset_dir_y")
    ax.set_title("Contact direction (cbar = cos_vrel)")
    fig.colorbar(sc, ax=ax, fraction=0.0475, pad=0)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}")


# ==================================================================================== #
# Summary printout
# ==================================================================================== #


def print_summary(data: dict[str, np.ndarray], label: str) -> None:
    """
    Print a short summary of the data.

    Parameters
        data
            The data dict from load_chunks().
        label
            A short text for the header line.
    """
    ok = data["success"]
    n = int(ok.sum())
    if n == 0:
        print(f"[{label}] no successful realisations")
        return

    v_rel = data["v_rel"][ok]
    cos_vrel = data["cos_vrel"][ok]
    cos_cont = data["cos_contact"][ok]
    t_bin = data["T_bin"][ok]
    a_bin = data["a_bin"][ok]

    classes = np.unique(v_rel)
    print(f"[{label}]  n_success = {n}")
    print(f"  a_bin          : median {fmt_num(np.median(a_bin))} m")
    print(f"  T_bin          : median {fmt_num(np.median(t_bin))} s")
    print(f"  v_rel classes  : {fmt_num(classes.tolist())} m/s")
    for c in classes:
        print(
            f"    v_rel={fmt_num(c):>8s} m/s : "
            f"n={int((v_rel == c).sum()):>7d}   "
            f"cos_vrel mean={cos_vrel[v_rel == c].mean():+.3f}"
        )
    rear = int((cos_vrel > 0.5).sum())
    head = int((cos_vrel < -0.5).sum())
    graz = n - rear - head
    print(f"  geometry       : rear-end={rear}  head-on={head}  grazing={graz}")
    print(f"  cos_contact    : mean {cos_cont.mean():+.3f}  std {cos_cont.std():.3f}")


# ==================================================================================== #
# Main
# ==================================================================================== #


def main() -> None:
    """
    Read the CLI arguments and run the requested figures.

    Returns
        None.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=str, help="directory containing chunk_*.npz files")
    ap.add_argument(
        "--out",
        type=str,
        default="media/param_scan_test.png",
        help="output PNG for the origins figure",
    )
    ap.add_argument(
        "--diagnostics",
        type=str,
        default=None,
        help="optional output PNG for the sampling diagnostics",
    )
    ap.add_argument(
        "--cos-vrel",
        type=str,
        default=None,
        help="filter realisations, e.g. '0.5:' for rear-end only",
    )
    ap.add_argument(
        "--n-scatter",
        type=int,
        default=10000,
        help="max points per panel in the xy scatter",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for scatter subsampling",
    )
    args = ap.parse_args()

    data, n_files = load_chunks(args.dir)
    print(f"loaded {n_files} chunk file(s) from {args.dir}")

    spec = parse_cos_vrel(args.cos_vrel)
    data_f, label = apply_cos_vrel_filter(data, spec)

    print_summary(data_f, label)
    print()

    if data_f["success"].any():
        plot_origins(
            data_f,
            args.out,
            label=label,
            n_scatter=args.n_scatter,
            seed=args.seed,
        )
    else:
        print("no cases pass the filter, so the origins figure is skipped")

    if args.diagnostics:
        plot_diagnostics(data_f, args.diagnostics, label=label)


if __name__ == "__main__":
    main()

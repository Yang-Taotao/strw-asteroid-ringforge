"""
Earth-Moon-Apophis flyby with a rubble-pile 99942 Apophis

CLI:
    python apophis.py  --n 10000 --settle-time 100000 --dt-settle 10.0 --no-progress
"""

# pyright: reportAttributeAccessIssue=false
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import main as rf
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import rebound
from lumberyard import setup_logging
from matplotlib.backends.backend_agg import FigureCanvasAgg
from scipy.constants import G as _G
from tqdm import tqdm

G: float = float(_G)

EARTH_MASS: float = 5.9722e24
EARTH_RADIUS: float = 6.371e6
MOON_MASS: float = 7.342e22
MOON_RADIUS: float = 1.7374e6
MOON_A: float = 3.844e8
MOON_E: float = 0.0549
MOON_INC: float = math.radians(5.145)

APOPHIS_MASS: float = 6.1e10
APOPHIS_DENSITY: float = 3200.0
APOPHIS_N: int = 10_000
APOPHIS_SLOPE: float = -3.0
APOPHIS_SEED: int = 42
APOPHIS_SETTLE_TIME: float = 100_000.0
APOPHIS_DT_SETTLE: float = 2.0
APOPHIS_SOFTENING: float = 2.0
APOPHIS_V_INF: float = 7_400.0
APOPHIS_PERIGEE: float = 3.74e7  # m from Earth COM
APOPHIS_INC: float = math.radians(3.34)
APOPHIS_INCOMING_SEPARATION: float = 2.0
ENCOUNTER_HALF_WINDOW: float = 6.0 * 3600.0
CACHE_VERSION: str = "apophis_v2"

LOGGER = logging.getLogger("ringforge.apophis")


def log_info(message: str) -> None:
    """Log at INFO; the console handler is configured in `main`"""
    LOGGER.info("%s", message)


def _com_state(
    masses: np.ndarray, positions: np.ndarray, velocities: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    total = float(masses.sum())
    if total <= 0:
        raise ValueError("total mass must be positive")
    pos = (masses[:, None] * positions).sum(axis=0) / total
    vel = (masses[:, None] * velocities).sum(axis=0) / total
    return np.asarray(pos), np.asarray(vel)


def _state_array(sim: rebound.Simulation, start: int = 0) -> np.ndarray:
    rows = []
    for index in range(start, sim.N):
        particle = sim.particles[index]
        rows.append(
            [
                particle.x,
                particle.y,
                particle.z,
                particle.vx,
                particle.vy,
                particle.vz,
                particle.r,
            ]
        )
    return np.asarray(rows, float)


def _point_state(sim: rebound.Simulation, index: int) -> np.ndarray:
    particle = sim.particles[index]
    return np.asarray(
        [
            particle.x,
            particle.y,
            particle.z,
            particle.vx,
            particle.vy,
            particle.vz,
        ],
        float,
    )


def apophis_radius(mass: float, density: float) -> float:
    """Equivalent spherical radius for a solid body"""
    return float((3.0 * mass / (4.0 * np.pi * density)) ** (1.0 / 3.0))


def cache_path(args: argparse.Namespace) -> Path:
    """Deterministic `settled_apophis_N<n>_<hash>.npz` path"""
    existing = getattr(args, "cache_file", None)
    if existing is not None:
        return Path(existing)
    ratio = rf.auto_radius_ratio(
        args.n,
        slope=APOPHIS_SLOPE,
        packing_fraction=rf.PACKING_FRACTION,
    )
    key = "|".join(
        str(value)
        for value in (
            CACHE_VERSION,
            args.mass,
            args.n,
            args.seed,
            args.density,
            APOPHIS_SLOPE,
            ratio,
            args.softening,
            rf.PARTICLE_BUFFER,
            rf.SETTLE_ROOT_SIZE,
            rf.PACKING_FRACTION,
            G,
        )
    )
    digest = hashlib.sha1(key.encode()).hexdigest()[:10]
    return Path(args.cache_dir) / f"settled_apophis_N{args.n}_{digest}.npz"


def ensure_settled(
    args: argparse.Namespace,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Build/settle Apophis if the cache is none"""
    path = cache_path(args)
    expected = {
        "target_mass": float(args.mass),
        "target_n": int(args.n),
        "slope": APOPHIS_SLOPE,
        "density": float(args.density),
        "seed": int(args.seed),
    }
    if not args.overwrite_settled:
        candidates = [path]
        candidates.extend(
            sorted(
                Path(args.cache_dir).glob("settled_apophis_*.npz"),
                key=lambda item: item.stat().st_mtime,
                reverse=True,
            )
        )
        seen: set[Path] = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            try:
                state, extent, meta = rf.load_settled_asteroid(
                    candidate, expected=expected, density=args.density
                )
            except Exception as exc:
                if candidate == path and candidate.exists():
                    log_info(f"cache {path.name} unusable ({exc})")
                continue
            stored_softening = meta.get("softening")
            if stored_softening is not None and not np.isclose(
                float(stored_softening), args.softening
            ):
                continue
            args.cache_file = str(candidate)
            log_info(
                f"reusing {candidate.name} "
                f"(settle={meta.get('settle_time')} s, dt={meta.get('dt')} s)"
            )
            if (
                meta.get("settle_time") != args.settle_time
                or meta.get("dt") != args.dt_settle
            ):
                LOGGER.warning(
                    "  cache was settled with settle=%s s, dt=%s s; "
                    "requested settle=%s s, dt=%s s",
                    meta.get("settle_time"),
                    meta.get("dt"),
                    args.settle_time,
                    args.dt_settle,
                )
            return state, extent, meta

    args.cache_file = str(path)
    log_info(f"settling {path.name}")
    job = {
        "target_mass": float(args.mass),
        "target_n": int(args.n),
        "settled_file": path,
        "seed": int(args.seed),
        "radius_ratio": rf.auto_radius_ratio(
            args.n, slope=APOPHIS_SLOPE, packing_fraction=rf.PACKING_FRACTION
        ),
        "slope": APOPHIS_SLOPE,
        "density": float(args.density),
        "settle_time": float(args.settle_time),
        "dt": float(args.dt_settle),
        "integrator": "leapfrog",
        "softening": float(args.softening),
        "buffer": rf.PARTICLE_BUFFER,
        "root_size": rf.SETTLE_ROOT_SIZE,
    }
    rf.process_asteroid(job)
    state, extent, meta = rf.load_settled_asteroid(
        path, expected=expected, density=args.density
    )
    return state, extent, meta


def hyperbolic_elements(
    v_inf: float, perigee: float, total_mass: float
) -> tuple[float, float]:
    """Return `(a, e)` for a hyperbolic Earth encounter"""
    mu = G * total_mass
    a = -mu / v_inf**2
    e = 1.0 + perigee * v_inf**2 / mu
    return float(a), float(e)


def mean_anomaly_at_lead(a: float, total_mass: float, lead: float) -> float:
    """Mean anomaly at `lead` seconds before periapsis (negative)"""
    n = math.sqrt(G * total_mass / abs(a) ** 3)
    return float(-n * lead)


def lead_time_for_separation(
    radius: float, a: float, e: float, total_mass: float
) -> float:
    """Time from a given Earth distance to periapsis on the incoming branch"""
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    p = a * (1.0 - e * e)
    cos_f = (p / radius - 1.0) / e
    if not -1.0 <= cos_f <= 1.0:
        raise ValueError("radius is outside the hyperbolic orbit")
    f = math.acos(cos_f)
    h = 2.0 * math.atanh(math.sqrt((e - 1.0) / (e + 1.0)) * math.tan(f / 2.0))
    mean = e * math.sinh(h) - h
    n = math.sqrt(G * total_mass / abs(a) ** 3)
    return float(mean / n)


def build_scene(
    state: np.ndarray,
    args: argparse.Namespace,
    apophis_extent: float,
) -> tuple[Any, dict[str, Any]]:
    """Construct the Earth-Moon-Apophis scene and return its initial state"""
    sim = rf.make_fresh_simulation(
        dt=args.dt,
        softening=args.softening,
        root_size=2.0e9,
        integrator="leapfrog",
    )
    sim.add(
        m=EARTH_MASS,
        r=EARTH_RADIUS,
        x=0.0,
        y=0.0,
        z=0.0,
        vx=0.0,
        vy=0.0,
        vz=0.0,
    )
    sim.add(m=MOON_MASS, a=MOON_A, e=MOON_E, inc=MOON_INC, l=0.0)
    sim.move_to_com()

    total = EARTH_MASS + MOON_MASS + args.mass
    a_hyp, e_hyp = hyperbolic_elements(args.v_inf, args.perigee, total)
    mean_anomaly = mean_anomaly_at_lead(a_hyp, total, args.lead_seconds)

    apophis_r = apophis_radius(args.mass, args.density)
    sim.add(
        m=args.mass,
        r=apophis_r,
        primary=sim.particles[0],
        a=a_hyp,
        e=e_hyp,
        inc=APOPHIS_INC,
        Omega=0.0,
        omega=0.0,
        l=mean_anomaly,
    )
    apophis = _point_state(sim, 2)
    sim.remove(2)

    rf.add_settled_at_point_state(
        sim,
        state,
        apophis,
        target_mass=args.mass,
        density=args.density,
        mass_mode="scale",
    )
    rf.align_root_size(sim, horizon=args.frame_every)

    final_earth = _point_state(sim, 0)
    final_moon = _point_state(sim, 1)
    apophis_state = _state_array(sim, start=2)
    apophis_masses = np.asarray([sim.particles[i].m for i in range(2, sim.N)], float)
    apophis_com, apophis_vel = _com_state(
        apophis_masses,
        apophis_state[:, :3],
        apophis_state[:, 3:6],
    )
    initial = {
        "earth_state": final_earth,
        "moon_state": final_moon,
        "apophis_state": np.concatenate([apophis_com, apophis_vel]),
        "earth_mass": EARTH_MASS,
        "moon_mass": MOON_MASS,
        "apophis_mass": float(args.mass),
        "earth_radius": EARTH_RADIUS,
        "moon_radius": MOON_RADIUS,
        "apophis_radius": apophis_r,
        "a_moon": MOON_A,
        "e_moon": MOON_E,
        "inc_moon": MOON_INC,
        "a_hyp": a_hyp,
        "e_hyp": e_hyp,
        "inc_hyp": APOPHIS_INC,
        "v_inf": float(args.v_inf),
        "perigee": float(args.perigee),
        "lead_seconds": float(args.lead_seconds),
        "apophis_extent": float(apophis_extent),
        "cache": str(cache_path(args)),
    }
    return sim, initial


def save_initial(path: Path, initial: dict[str, Any], args: argparse.Namespace) -> None:
    """Write `data/apophis_initial.npz` and a json sidecar"""
    meta = {
        "description": "constructed Earth-Moon-Apophis encounter; SI units",
        "created_utc": datetime.now(UTC).isoformat(),
        "cache": initial["cache"],
        "args": vars(args),
        "a_hyp": initial["a_hyp"],
        "e_hyp": initial["e_hyp"],
        "v_inf": initial["v_inf"],
        "perigee_m": initial["perigee"],
        "lead_seconds": initial["lead_seconds"],
    }
    arrays = {
        key: np.asarray(value, float)
        for key, value in initial.items()
        if key not in ("cache",)
    }
    arrays["meta_json"] = np.asarray(json.dumps(meta, sort_keys=True))
    rf.atomic_savez(path, **arrays)
    rf.atomic_write_json(path.with_suffix(".json"), meta)
    log_info(f"initial state -> {path}")


class Renderer:
    """Two-panel renderer: 1) wide Earth-Moon view, 2) Apophis zoomed in"""

    def __init__(
        self,
        sim: rebound.Simulation,
        args: argparse.Namespace,
        apophis_extent: float,
    ) -> None:
        self.fig, (self.ax_wide, self.ax_zoom) = plt.subplots(
            1,
            2,
            figsize=(args.fig_width, 0.5 * args.fig_width),
            dpi=args.render_dpi,
        )
        self.fig.subplots_adjust(
            left=0.10, right=0.98, top=0.86, bottom=0.12, wspace=0.28
        )
        self.fig.suptitle(
            "Constructed 2029-style Apophis flyby (not an ephemeris)",
            fontsize=9,
        )
        self.canvas = FigureCanvasAgg(self.fig)
        self.unit = MOON_A  # metres per Earth-Moon separation
        half = 2.0
        self.ax_wide.set_aspect("equal")
        self.ax_wide.set_xlim(-half, half)
        self.ax_wide.set_ylim(-half, half)
        self.ax_wide.set_xlabel("x [Earth-Moon separation]")
        self.ax_wide.set_ylabel("y [Earth-Moon separation]")
        self.ax_wide.set_title("Earth-Moon")

        self.particle_radii = np.asarray(
            [sim.particles[i].r for i in range(2, sim.N)], float
        )
        self.base_zoom = max(
            args.zoom_factor * float(apophis_extent),
            1.2 * float(self.particle_radii.max())
            if self.particle_radii.size
            else 200.0,
            100.0,
        )
        self.zoom = self.base_zoom
        self.lead_seconds = float(args.lead_seconds)
        self.ax_zoom.set_aspect("equal")
        self.ax_zoom.set_xlim(-self.zoom, self.zoom)
        self.ax_zoom.set_ylim(-self.zoom, self.zoom)
        self.ax_zoom.set_xlabel("x - x$_{COM}$ [m]")
        self.ax_zoom.set_ylabel("y - y$_{COM}$ [m]")
        self.ax_zoom.set_title("Apophis")

        gray = args.color_mode == "gray"
        earth_color = "0.40" if gray else "tab:blue"
        moon_color = "0.60" if gray else "0.5"
        self.earth = patches.Circle(
            (0.0, 0.0),
            radius=EARTH_RADIUS / self.unit,
            facecolor=earth_color,
            edgecolor="black",
            linewidth=0.5,
            zorder=5,
        )
        self.moon = patches.Circle(
            (0.0, 0.0),
            radius=MOON_RADIUS / self.unit,
            facecolor=moon_color,
            edgecolor="black",
            linewidth=0.4,
            zorder=4,
        )
        self.ax_wide.add_patch(self.earth)
        self.ax_wide.add_patch(self.moon)
        self.ax_wide.text(
            -half * 0.95,
            half * 0.90,
            "Earth",
            fontsize=8,
            color="black",
        )
        self.moon_label = self.ax_wide.text(
            0.0,
            0.0,
            "Moon",
            fontsize=6,
            color="black",
            clip_on=True,
        )
        (self.apophis_marker,) = self.ax_wide.plot(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=4,
            color="black" if gray else "tab:red",
            zorder=7,
            label="Apophis",
        )
        (self.trail,) = self.ax_wide.plot([], [], color="0.25", lw=0.7, zorder=3)
        self.ax_wide.legend(loc="upper right", fontsize=7)
        self.info_wide = self.ax_wide.text(
            0.02,
            0.02,
            "",
            transform=self.ax_wide.transAxes,
            va="bottom",
            fontsize=7,
            family="monospace",
        )

        self.particles: list[patches.Circle] = []
        radii = np.asarray([sim.particles[i].r for i in range(2, sim.N)], float)
        r_max = float(radii.max()) if radii.size else 1.0
        for radius in radii:
            shade = 0.30 + 0.55 * float(radius / r_max)
            self.particles.append(
                patches.Circle(
                    (0.0, 0.0),
                    radius=float(radius),
                    facecolor=f"{shade:.3f}" if gray else "0.55",
                    edgecolor="none",
                    alpha=0.9,
                )
            )
        for circle in self.particles:
            self.ax_zoom.add_patch(circle)
        self.info_zoom = self.ax_zoom.text(
            0.02,
            0.02,
            "",
            transform=self.ax_zoom.transAxes,
            va="bottom",
            fontsize=7,
            family="monospace",
        )
        self.grayscale = gray
        self.trail_x: list[float] = []
        self.trail_y: list[float] = []

    def frame(self, sim: rebound.Simulation, t: float, index: int) -> np.ndarray:
        earth = _point_state(sim, 0)
        moon = _point_state(sim, 1)
        self.earth.center = (earth[0] / self.unit, earth[1] / self.unit)
        self.moon.center = (moon[0] / self.unit, moon[1] / self.unit)
        self.moon_label.set_position(
            (moon[0] / self.unit + 0.015, moon[1] / self.unit + 0.015)
        )

        masses = np.asarray([sim.particles[i].m for i in range(2, sim.N)], float)
        positions = np.asarray(
            [
                [sim.particles[i].x, sim.particles[i].y, sim.particles[i].z]
                for i in range(2, sim.N)
            ],
            float,
        )
        velocities = np.asarray(
            [
                [sim.particles[i].vx, sim.particles[i].vy, sim.particles[i].vz]
                for i in range(2, sim.N)
            ],
            float,
        )
        com, vel = _com_state(masses, positions, velocities)
        rel = positions - com
        for circle, row in zip(self.particles, rel, strict=True):
            circle.center = (float(row[0]), float(row[1]))

        radial = np.linalg.norm(rel[:, :2], axis=1)
        outer = radial + self.particle_radii if self.particle_radii.size else radial
        zoom = max(self.base_zoom, 1.1 * float(outer.max()) if outer.size else 0.0)
        if zoom > self.zoom:
            self.zoom = zoom
            self.ax_zoom.set_xlim(-zoom, zoom)
            self.ax_zoom.set_ylim(-zoom, zoom)

        x = float(com[0])
        y = float(com[1])
        self.trail_x.append(x / self.unit)
        self.trail_y.append(y / self.unit)
        self.trail.set_data(self.trail_x, self.trail_y)
        self.apophis_marker.set_data([x / self.unit], [y / self.unit])

        r = float(np.linalg.norm(com - earth[:3]))
        v_rel = float(np.linalg.norm(vel - earth[3:6]))
        hours = t / 3600.0
        if t < self.lead_seconds - ENCOUNTER_HALF_WINDOW:
            stage = "incoming"
        elif t <= self.lead_seconds + ENCOUNTER_HALF_WINDOW:
            stage = "encounter"
        else:
            stage = "outgoing"
        r_moon = r / self.unit
        self.ax_wide.set_title(f"Earth-Moon | {stage}")
        self.ax_zoom.set_title(f"Apophis | {stage}")
        self.info_wide.set_text(
            f"frame {index:05d}\n"
            f"stage = {stage}\n"
            f"t = {hours:8.3f} h\n"
            f"r_E = {r_moon:6.3f} a_Moon\n"
            f"v_rel = {v_rel / 1e3:6.2f} km/s"
        )
        self.info_zoom.set_text(f"N = {sim.N - 2}\nzoom = ±{self.zoom:.0f} m")
        self.canvas.draw()
        rgba = np.asarray(self.canvas.buffer_rgba())
        if self.grayscale:
            return rgba[..., 0].copy()
        return rgba[..., :3].copy()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Earth-Moon-Apophis rubble-pile flyby")
    parser.add_argument("--n", type=int, default=APOPHIS_N)
    parser.add_argument("--mass", type=float, default=APOPHIS_MASS)
    parser.add_argument("--density", type=float, default=APOPHIS_DENSITY)
    parser.add_argument("--seed", type=int, default=APOPHIS_SEED)
    parser.add_argument("--settle-time", type=float, default=APOPHIS_SETTLE_TIME)
    parser.add_argument("--dt-settle", type=float, default=APOPHIS_DT_SETTLE)
    parser.add_argument("--softening", type=float, default=APOPHIS_SOFTENING)
    parser.add_argument("--overwrite-settled", action="store_true")
    parser.add_argument("--cache-dir", type=str, default="data/apophis")
    parser.add_argument("--initial-path", type=str, default="data/apophis/initial.npz")
    parser.add_argument("--duration-days", type=float, default=None)
    parser.add_argument("--lead-hours", type=float, default=None)
    parser.add_argument(
        "--incoming-separation", type=float, default=APOPHIS_INCOMING_SEPARATION
    )
    parser.add_argument("--dt", type=float, default=10.0)
    parser.add_argument("--frame-every", type=float, default=120.0)
    parser.add_argument("--v-inf", type=float, default=APOPHIS_V_INF)
    parser.add_argument("--perigee-km", type=float, default=APOPHIS_PERIGEE / 1e3)
    parser.add_argument("--fig-width", type=float, default=10.0)
    parser.add_argument("--render-dpi", type=int, default=100)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--crf", type=int, default=20)
    parser.add_argument("--color-mode", choices=("gray", "rgb"), default="gray")
    parser.add_argument("--zoom-factor", type=float, default=1.2)
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.n < 17:
        raise SystemExit("--n must be at least 17")
    if args.mass <= 0 or args.density <= 0:
        raise SystemExit("--mass and --density must be positive")
    if args.dt <= 0 or args.frame_every <= 0:
        raise SystemExit("--dt and --frame-every must be positive")
    if args.duration_days is not None and args.duration_days <= 0:
        raise SystemExit("--duration-days must be positive")
    if args.lead_hours is not None and args.lead_hours <= 0:
        raise SystemExit("--lead-hours must be positive")
    if args.incoming_separation <= 0:
        raise SystemExit("--incoming-separation must be positive")
    if args.perigee_km <= EARTH_RADIUS / 1e3:
        raise SystemExit("--perigee-km must exceed the Earth radius")
    if args.fig_width <= 0 or args.render_dpi <= 0:
        raise SystemExit("--fig-width and --render-dpi must be positive")
    if args.fps <= 0 or not 0 <= args.crf <= 51:
        raise SystemExit("--fps must be positive and --crf in [0, 51]")
    if args.zoom_factor <= 0:
        raise SystemExit("zoom factor must be positive")
    ratio = rf.auto_radius_ratio(
        args.n, slope=APOPHIS_SLOPE, packing_fraction=rf.PACKING_FRACTION
    )
    if ratio <= 1.0:
        raise SystemExit("invalid particle size ratio")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.dt = float(args.dt)
    args.frame_every = float(args.frame_every)
    args.perigee = float(args.perigee_km) * 1e3
    validate_args(args)

    total_mass = EARTH_MASS + MOON_MASS + args.mass
    a_hyp, e_hyp = hyperbolic_elements(args.v_inf, args.perigee, total_mass)
    if args.lead_hours is None:
        args.lead_seconds = lead_time_for_separation(
            args.incoming_separation * MOON_A, a_hyp, e_hyp, total_mass
        )
    else:
        args.lead_seconds = float(args.lead_hours) * 3600.0
    if args.duration_days is None:
        args.duration_seconds = 2.0 * args.lead_seconds
    else:
        args.duration_seconds = float(args.duration_days) * 86400.0

    _, log_path = setup_logging(
        "ringforge.apophis",
        log_dir=args.log_dir,
        console_level=logging.WARNING if args.quiet else logging.INFO,
    )
    rf.LOGGER = LOGGER
    show_progress = not (args.quiet or args.no_progress)
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    rf.VID_DIR.mkdir(parents=True, exist_ok=True)
    run_id = rf.make_run_id()
    log_info(f"run_id={run_id}")
    log_info(f"log={log_path}")

    state, extent, meta = ensure_settled(args)
    log_info(
        f"Apophis: N={args.n} M={args.mass:.3e} kg "
        f"extent={extent:.1f} m radius={apophis_radius(args.mass, args.density):.1f} m"
    )
    r_lo = meta.get("r_lo")
    r_hi = meta.get("r_hi")
    rf.check_powerlaw(
        state[:, 6],
        APOPHIS_SLOPE,
        float(r_lo if r_lo is not None else state[:, 6].min()),
        float(r_hi if r_hi is not None else state[:, 6].max()),
        label="apophis",
    )
    rf.state_diagnostics(state, density=args.density, label="apophis")

    sim, initial = build_scene(state, args, extent)
    initial_path = Path(args.initial_path)
    save_initial(initial_path, initial, args)

    video_path = (
        Path(args.video)
        if args.video is not None
        else rf.VID_DIR / f"apophis_{run_id}.mp4"
    )
    manifest = {
        "run_id": run_id,
        "created_utc": datetime.now(UTC).isoformat(),
        "args": vars(args),
        "cache": str(cache_path(args)),
        "initial": str(initial_path),
        "video": str(video_path),
        "earth_state": initial["earth_state"].tolist(),
        "moon_state": initial["moon_state"].tolist(),
        "apophis_state": initial["apophis_state"].tolist(),
        "a_hyp": initial["a_hyp"],
        "e_hyp": initial["e_hyp"],
        "v_inf": initial["v_inf"],
        "perigee_m": initial["perigee"],
        "lead_seconds": initial["lead_seconds"],
        "duration_seconds": args.duration_seconds,
    }
    manifest_path = Path(args.cache_dir) / f"apophis_run_{run_id}.json"
    rf.atomic_write_json(manifest_path, manifest)

    n_frames = math.ceil(args.duration_seconds / args.frame_every) + 1
    log_info(
        f"scene: dt={args.dt:g} s  duration={args.duration_seconds / 86400.0:g} d  "
        f"lead={args.lead_seconds / 3600.0:g} h  frames~{n_frames}  "
        f"dt/frame={args.frame_every:g} s"
    )
    if n_frames > 5000:
        log_info(
            "warning: long render; increase --frame-every or shorten --duration-days"
        )

    renderer = Renderer(sim, args, extent)
    frame0 = renderer.frame(sim, 0.0, 0)
    writer = rf.FFmpegVideoWriter(
        video_path,
        int(frame0.shape[1]),
        int(frame0.shape[0]),
        fps=args.fps,
        crf=args.crf,
        grayscale=args.color_mode == "gray",
    )
    writer.write(frame0)
    t0 = time.time()
    frame_index = 1
    progress = tqdm(
        total=math.ceil(args.duration_seconds),
        unit="s",
        desc="apophis",
        ncols=88,
        disable=not show_progress,
    )
    last_t = 0.0
    try:
        while sim.t < args.duration_seconds - 1e-9:
            t_next = min(sim.t + args.frame_every, args.duration_seconds)
            sim.integrate(t_next)
            rf.align_root_size(sim, horizon=args.frame_every)
            delta = max(0.0, float(sim.t) - last_t)
            progress.update(min(delta, float(progress.total) - progress.n))
            last_t = float(sim.t)
            writer.write(renderer.frame(sim, float(sim.t), frame_index))
            frame_index += 1
    except Exception:
        writer.abort()
        raise
    finally:
        progress.close()
        plt.close(renderer.fig)
    writer.close()
    log_info(
        f"wrote {video_path} ({frame_index} frames, wall {time.time() - t0:.1f} s)"
    )
    manifest["frames"] = frame_index
    manifest["wall_time_s"] = time.time() - t0
    manifest["video"] = str(video_path)
    rf.atomic_write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()

"""
REBOUND rubble-pile asteroids collision script

Stages:
    settle -> cache -> build scene -> integrate -> render

Notes:
    - Always in SI units
    - Particle states are `(N, 7)` arrays of `x, y, z, vx, vy, vz, r`
    - masses follow from radius and density, with radii rescaled to match target mass

cfgs:
    Parse from `config.toml` (read through `config.py`)

CLI:
    python main.py \
        --param-path data/param_scan_test/ --param-index 42 \
        --particles-per-body 1000 --look-back 0.01 
    python main.py --rebuild-gif data/collision_frames_XXX
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import config as ring_config
import matplotlib
import numpy as np
import rebound

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import param_utils
from lumberyard import configure_worker_logging, setup_logging
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image
from scipy.constants import G as _G
from tqdm import tqdm

# logger init
LOGGER = logging.getLogger("ringforge.main")


def log_info(message: str) -> None:
    LOGGER.info("%s", message)


def log_debug(message: str) -> None:
    LOGGER.debug("%s", message)


def log_warning(message: str) -> None:
    LOGGER.warning("%s", message)


# ==================================================================================== #
# Configs
# ==================================================================================== #

_CFG = ring_config.load_config()


def _cfg[T](section: str, key: str, default: T) -> T:
    """Read cfg from `config.toml` with defaults fallback"""
    return cast(T, ring_config.value(_CFG, section, key, default))


# ==================================================================================== #
# Physical constants
# ==================================================================================== #

G: float = float(_G)  # m^3 / (kg s^2), scipy const G
PARTICLE_DENSITY: float = float(_cfg("bodies", "density", 3000.0))  # kg/m^3
PACKING_FRACTION: float = float(_cfg("bodies", "packing_fraction", 0.45))
MAX_RMAX_OVER_RBALL: float = float(_cfg("bodies", "max_rmax_over_rball", 0.30))
SOFTENING_DEFAULT: float = float(_cfg("settle", "softening", 20.0))  # m
PARTICLE_BUFFER: float = 2.0  # m

# ==================================================================================== #
# Asteroids
# ==================================================================================== #

M1: float = float(_cfg("bodies", "ma", 1.0e15))  # kg, host 1 (ma)
M2: float = float(_cfg("bodies", "mb", 1.0e13))  # kg, host 2 (mb)
M3: float = float(_cfg("bodies", "mc", 1.0e12))  # kg, bullet (mc)

# Unused particle based asterpods builder
N1: int = int(_cfg("bodies", "n1", 2100))
N2: int = int(_cfg("bodies", "n2", 900))
N3: int = int(_cfg("bodies", "n3", 300))

# Base power-law range at the M1 mass scale.
R_MIN: float = float(_cfg("bodies", "r_min", 50.0))  # m
R_MAX: float = float(_cfg("bodies", "r_max", 1000.0))  # m
SIZE_SLOPE: float = float(_cfg("bodies", "size_slope", -3.0))
RUN_SEED: int = int(_cfg("run", "seed", 42))
PLACEMENT_SEED: int = int(_cfg("run", "placement_seed", 42))
PLACEMENT_MAX_TRIES: int = 20000

BALL_RADIUS_REF: float = float(
    (3.0 * M1 / (4.0 * np.pi * PACKING_FRACTION * PARTICLE_DENSITY)) ** (1.0 / 3.0)
)

# ==================================================================================== #
# Time control
# ==================================================================================== #

SETTLE_TIME: float = float(_cfg("settle", "settle_time", 108000.0))  # s
DT_SETTLE: float = float(_cfg("settle", "dt", 10.0))  # s
DT_HIT: float = float(_cfg("impact", "dt", 1.0))  # s

SETTLE_CHUNKS: int = int(_cfg("settle", "chunks", 10))
SETTLE_MAX_SUBSTEPS: int = int(_cfg("settle", "max_substeps", 100))
SETTLE_DT_SAFETY: float = float(_cfg("settle", "dt_safety", 0.10))
SETTLE_DT_MIN: float = 1.0e-3

SETTLE_ROOT_SIZE: float = 1.0e7  # m, grows via align_root_size
HIT_ROOT_SIZE: float = 1.0e8  # m, grows for lookback starts

SETTLE_INTEGRATOR: str = str(_cfg("settle", "integrator", "leapfrog"))
HIT_INTEGRATOR: str = str(_cfg("impact", "integrator", "leapfrog"))

STEPS_PER_FRAME_HIT: int = int(_cfg("impact", "steps_per_frame", 5))

# ==================================================================================== #
# Impact geometry (fixed-geometry path only)
# ==================================================================================== #

SEPARATION_FACTOR: float = 6.0
BULLET_SPEED: float = float(_cfg("impact", "bullet_speed", 100.0))  # m/s
BOX_SIZE_FACTOR: float = 32.0
POST_IMPACT_TAIL: float = float(_cfg("impact", "post_impact_tail", 18000.0))  # s

# Pseudo bin and bullet placement cfg
OFFSET_1 = np.asarray([+0.00, 0.25, 0.00])
BLKVEL_1 = np.asarray([-0.01, 0.00, 0.00])
OFFSET_2 = np.asarray([+0.00, 0.00, 0.00])
BLKVEL_2 = np.asarray([+0.01, 0.00, 0.00])
OFFSET_3 = np.asarray([-1.00, 0.00, 0.00])
BLKVEL_3 = np.asarray([+0.20, 0.00, 0.00])

# ==================================================================================== #
# I/O and rendering
# ==================================================================================== #

DATA_DIR: Path = Path(str(_cfg("run", "data_dir", "data")))
MEDIA_DIR: Path = Path(str(_cfg("run", "media_dir", "media")))
GIF_DIR: Path = MEDIA_DIR / "gif"
VID_DIR: Path = MEDIA_DIR / "vid"

SETTLED_STEM: str = "settled"
GIF_STEM: str = "gif"
VID_STEM: str = "vid"

MEDIA_FORMAT: str = str(_cfg("render", "media_format", "mp4"))
COLOR_MODE: str = str(_cfg("render", "color_mode", "gray"))
VIDEO_FPS: int = int(_cfg("render", "video_fps", 20))
VIDEO_CRF: int = int(_cfg("render", "video_crf", 20))

FRAMES_CHUNK_SIZE: int = 100
DURATION_MS: int = int(_cfg("render", "gif_duration_ms", 50))
GIF_QUANTIZE_COLORS: int = int(_cfg("render", "gif_quantize_colors", 128))

FIG_SIZE_IN: float = float(_cfg("render", "fig_size_in", 8.0))
RENDER_DPI: int = int(_cfg("render", "render_dpi", 100))
VIEW_PERCENTILE: float = float(_cfg("render", "view_percentile", 95.0))

MIN_FRAME_HALF_KM: float = float(_cfg("render", "min_half_extent_km", 10.0))
MAX_HALF_EXTENT_A_BIN: float = float(_cfg("render", "max_half_extent_a_bin", 2.0))
BOUND_RADIUS_A_BIN: float = float(_cfg("impact", "bound_radius_a_bin", 1.0))
BOUND_CHECK_EVERY: int = int(_cfg("impact", "bound_check_every", 100))
MIN_SETTLED_BOUND_FRACTION: float = float(_cfg("settle", "min_bound_fraction", 0.98))
MAX_SETTLED_EXTENT_FACTOR: float = float(_cfg("settle", "max_extent_factor", 3.0))

VERSION: str = "0.0.1"


# ==================================================================================== #
# Small helpers
# ==================================================================================== #


def format_hms(seconds: float) -> str:
    """Format a duration as `HH:MM:SS`"""
    seconds = int(seconds)
    sign = "-" if seconds < 0 else ""
    seconds = abs(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{sign}{hours:02d}:{minutes:02d}:{secs:02d}"


def format_clock(epoch_seconds: float) -> str:
    """Format an absolute epoch time as local `HH:MM:SS`"""
    return time.strftime("%H:%M:%S", time.localtime(epoch_seconds))


def _utc_now() -> str:
    """Get UTC time now"""
    return datetime.now(UTC).isoformat()


def _git_hash() -> str:
    """Check local git hash"""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def make_run_id(param_index: int | None = None) -> str:
    """
    Return a unique run identifier

    Includes the param index (if present), time in ms, and pid
    Parallel `run_cases.sh` jobs can start in the same second without
        colliding on manifests, chunk directories, GIFs, or videos
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    tag = f"_p{param_index:03d}" if param_index is not None else ""
    return f"{stamp}{tag}_{os.getpid()}"


def collect_metadata() -> dict[str, Any]:
    """Reproducibility metadata stored beside every cached state and run"""
    return {
        "created_utc": _utc_now(),
        "git_hash": _git_hash(),
        "rebound": rebound.__version__,
        "numpy": np.__version__,
        "python": sys.version.split()[0],
        "cwd": str(Path.cwd()),
        "argv": list(sys.argv),
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write to JSON atomically"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    """Write an `npz` without pickle"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, allow_pickle=False, **arrays)
    tmp.replace(path)


# ==================================================================================== #
# Power-law size distribution and packing geometry
# ==================================================================================== #


def powerlaw_sizes(
    rng: np.random.Generator,
    slope: float,
    r_min: float,
    r_max: float,
    n: int,
) -> np.ndarray:
    """Draw `n` radii from p(r) propto r^slope on `[r_min, r_max]`"""
    if n <= 0:
        raise ValueError("n must be positive")
    if r_min <= 0 or r_max <= r_min:
        raise ValueError("require 0 < r_min < r_max")
    exponent = float(slope) + 1.0
    if np.isclose(exponent, 0.0):
        raise ValueError("slope = -1 is not supported by this power-law sampler")
    u = rng.uniform(size=int(n))
    p_max = r_max**exponent
    p_min = r_min**exponent
    return np.asarray(((p_max - p_min) * u + p_min) ** (1.0 / exponent), float)


def mass_from_radius(
    radius: float | np.ndarray,
    density: float = PARTICLE_DENSITY,
) -> float | np.ndarray:
    """Mass of a solid sphere"""
    return density * (4.0 / 3.0) * np.pi * np.asarray(radius, float) ** 3


def expected_particle_mass(
    density: float = PARTICLE_DENSITY,
    r_min: float = R_MIN,
    r_max: float = R_MAX,
    slope: float = SIZE_SLOPE,
) -> float:
    """Expected mass for p(r) propto r^slope on `[r_min, r_max]`"""
    if density <= 0 or r_min <= 0 or r_max <= r_min:
        raise ValueError("require density > 0 and 0 < r_min < r_max")
    numerator_power = slope + 4.0
    denominator_power = slope + 1.0
    numerator = (
        np.log(r_max / r_min)
        if np.isclose(numerator_power, 0.0)
        else (r_max**numerator_power - r_min**numerator_power) / numerator_power
    )
    denominator = (
        np.log(r_max / r_min)
        if np.isclose(denominator_power, 0.0)
        else (r_max**denominator_power - r_min**denominator_power) / denominator_power
    )
    return float(density * (4.0 / 3.0) * np.pi * numerator / denominator)


def particle_count_for_mass(
    target_mass: float,
    density: float = PARTICLE_DENSITY,
    r_min: float = R_MIN,
    r_max: float = R_MAX,
    slope: float = SIZE_SLOPE,
) -> int:
    """Expected particle count for a target mass and a radius law"""
    if not np.isfinite(target_mass) or target_mass <= 0:
        raise ValueError("target_mass must be finite and positive")
    return max(
        1,
        round(
            target_mass
            / expected_particle_mass(
                density=density, r_min=r_min, r_max=r_max, slope=slope
            )
        ),
    )


def ball_radius_for_mass(
    target_mass: float,
    packing_fraction: float = PACKING_FRACTION,
    density: float = PARTICLE_DENSITY,
) -> float:
    """Ball radius holding the target mass at the requested packing fraction"""
    if density <= 0:
        raise ValueError("density must be positive")
    return float(
        (3.0 * target_mass / (4.0 * np.pi * packing_fraction * density)) ** (1.0 / 3.0)
    )


def mean_r3_factor(rho: float, slope: float) -> float:
    """
    Set `<r^3> / r_min^3` for p(r) propto r^slope on `[r_min, rho*r_min]` over hard -3.
    """
    if rho <= 0:
        raise ValueError("rho must be positive")
    if np.isclose(rho, 1.0):
        return 1.0
    a = slope + 4.0
    b = slope + 1.0
    numerator = np.log(rho) if np.isclose(a, 0.0) else (rho**a - 1.0) / a
    denominator = np.log(rho) if np.isclose(b, 0.0) else (rho**b - 1.0) / b
    return float(numerator / denominator)


def auto_radius_ratio(
    n_particles: int,
    *,
    slope: float = SIZE_SLOPE,
    packing_fraction: float = PACKING_FRACTION,
    max_rmax_over_rball: float = MAX_RMAX_OVER_RBALL,
) -> float:
    """
    Largest `r_max/r_min` such that `r_max/R_ball <= max_rmax_over_rball`.

    The derivation uses the exact `<r^3>` of the truncated power law
    Raises error if the requested particle count cannot match the cap at `r_max=r_min`
    """
    if n_particles < 1:
        raise ValueError("n_particles must be at least 1")
    if not (0.0 < packing_fraction < 1.0):
        raise ValueError("packing_fraction must be in (0, 1)")

    def rmax_over_rball(rho: float) -> float:
        # mean_r3 = eta * R_ball^3 / N; r_max^3 = mean_r3 * rho^3 / g(rho)
        mean_r3_over_rball3 = packing_fraction / float(n_particles)
        value = mean_r3_over_rball3 * rho**3 / mean_r3_factor(rho, slope)
        return float(value ** (1.0 / 3.0))

    if rmax_over_rball(1.0) > max_rmax_over_rball:
        raise ValueError(
            f"N={n_particles} is too small for r_max/R_ball <= "
            f"{max_rmax_over_rball:.2f} (even r_max=r_min gives "
            f"{rmax_over_rball(1.0):.3f}); increase N or the cap"
        )

    lo, hi = 1.0, 1.0e6
    if rmax_over_rball(hi) <= max_rmax_over_rball:
        return hi  # the cap is never reached; a very broad law is allowed

    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if rmax_over_rball(mid) < max_rmax_over_rball:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def scaled_radius_range(
    target_mass: float,
    target_n: int,
    *,
    size_scaling: str = "mass",
    radius_ratio: float | None = None,
    slope: float = SIZE_SLOPE,
    density: float = PARTICLE_DENSITY,
) -> tuple[float, float]:
    """
    Return the `(r_min, r_max)` used to draw the radii for this body

    `size_scaling`=
        `'mass'` scales the base `[R_MIN, R_MAX]` by`(M/M1)^(1/3)`
        `'fixed'` keeps the base range

    The target N then rescales the range so the expected particle mass is M/N

    An explicit `radius_ratio` reconstructs the range from the same mean `r^3` while
        forcing `r_max/r_min = radius_ratio`
    """
    if not np.isfinite(target_mass) or target_mass <= 0:
        raise ValueError("target_mass must be finite and positive")
    if target_n < 1:
        raise ValueError("target_n must be at least 1")
    if not (radius_ratio is None or radius_ratio >= 1.0):
        raise ValueError("radius_ratio must be >= 1")

    if size_scaling == "mass":
        scale = (target_mass / M1) ** (1.0 / 3.0)
        base_r_min, base_r_max = R_MIN * scale, R_MAX * scale
    elif size_scaling == "fixed":
        base_r_min, base_r_max = R_MIN, R_MAX
    else:
        raise ValueError(f"unknown size_scaling: {size_scaling!r}")

    base_mean_m = expected_particle_mass(
        density=density,
        r_min=base_r_min,
        r_max=base_r_max,
        slope=slope,
    )
    scale_n = (target_mass / (target_n * base_mean_m)) ** (1.0 / 3.0)
    base_r_min *= scale_n
    base_r_max *= scale_n

    if radius_ratio is None:
        return float(base_r_min), float(base_r_max)

    rho = float(radius_ratio)
    mean_r3 = expected_particle_mass(
        density=1.0,
        r_min=base_r_min,
        r_max=base_r_max,
        slope=slope,
    ) / (4.0 / 3.0 * np.pi)
    r_min_new = (mean_r3 / mean_r3_factor(rho, slope)) ** (1.0 / 3.0)
    return float(r_min_new), float(r_min_new * rho)


def describe_body(
    label: str,
    target_mass: float,
    target_n: int,
    radius_ratio: float,
    *,
    density: float = PARTICLE_DENSITY,
    slope: float = SIZE_SLOPE,
) -> None:
    """Print the particle budget and the key placement ratios"""
    r_lo, r_hi = scaled_radius_range(
        target_mass,
        target_n,
        radius_ratio=radius_ratio,
        slope=slope,
        density=density,
    )
    r_ball = ball_radius_for_mass(target_mass, density=density)
    n_exp = particle_count_for_mass(
        target_mass, density=density, r_min=r_lo, r_max=r_hi, slope=slope
    )
    eta = target_mass / (density * (4.0 / 3.0) * np.pi * r_ball**3)
    frac = r_hi / r_ball
    flag = "  <-- WARN r_max/R_ball > 0.30" if frac > MAX_RMAX_OVER_RBALL + 1e-6 else ""
    log_info(
        f"         {label:8s} M={target_mass:.3e} kg  "
        f"r=[{r_lo:8.1f}, {r_hi:8.1f}] m  "
        f"R_ball={r_ball:7.0f} m  eta={eta:.3f}  "
        f"r_max/R_ball={frac:.3f}  N={target_n:5d} (derived {n_exp:5d}){flag}",
    )


# ==================================================================================== #
# Deterministic cache paths
# ==================================================================================== #


def _fingerlog_info(*parts: object) -> str:
    key = "|".join(str(part) for part in parts)
    return hashlib.sha1(key.encode()).hexdigest()[:10]


def settled_path(
    target_mass: float,
    target_n: int,
    *,
    seed: int,
    radius_ratio: float,
    slope: float = SIZE_SLOPE,
    density: float = PARTICLE_DENSITY,
    packing_fraction: float = PACKING_FRACTION,
    settle_time: float = SETTLE_TIME,
    dt: float = DT_SETTLE,
    integrator: str = SETTLE_INTEGRATOR,
    softening: float = SOFTENING_DEFAULT,
    buffer: float = PARTICLE_BUFFER,
    root_size: float = SETTLE_ROOT_SIZE,
    size_scaling: str = "mass",
    r_min: float = R_MIN,
    r_max: float = R_MAX,
) -> Path:
    """Deterministic cache path for one settled body"""
    r_ball = ball_radius_for_mass(target_mass, density=density)
    digest = _fingerlog_info(
        VERSION,
        target_mass,
        target_n,
        seed,
        radius_ratio,
        slope,
        density,
        packing_fraction,
        settle_time,
        dt,
        integrator,
        softening,
        buffer,
        root_size,
        size_scaling,
        r_min,
        r_max,
        r_ball,
        G,
    )
    return DATA_DIR / f"{SETTLED_STEM}_M{target_mass:.6e}_N{target_n}_{digest}.npz"


# ==================================================================================== #
# REBOUND factory and particle placement
# ==================================================================================== #


def coefficient_of_restitution(radius: float, speed: float) -> float:
    """Constant coefficient of restitution for hard-sphere collisions"""
    return 0.5


def make_fresh_simulation(
    *,
    dt: float,
    softening: float = SOFTENING_DEFAULT,
    root_size: float = SETTLE_ROOT_SIZE,
    integrator: str = SETTLE_INTEGRATOR,
) -> rebound.Simulation:
    """Create new tree code grav and hard-sphere REBOUND sim"""
    sim = rebound.Simulation()
    sim.G = G
    sim.softening = softening
    sim.dt = dt
    sim.integrator = integrator
    sim.gravity = "tree"
    sim.root_size = root_size
    sim.opening_angle2 = 0.5
    sim.collision = "tree"
    sim.coefficient_of_restitution = coefficient_of_restitution
    sim.collision_resolve = "hardsphere"
    return sim


def _fits(
    sim: rebound.Simulation,
    x: float,
    y: float,
    z: float,
    radius: float,
    buffer: float,
) -> bool:
    """True if a sphere at `(x, y, z)` clears every existing particle"""
    for particle in sim.particles:
        dx = particle.x - x
        dy = particle.y - y
        dz = particle.z - z
        if dx * dx + dy * dy + dz * dz < (particle.r + radius + buffer) ** 2:
            return False
    return True


def place_radii(
    sim: rebound.Simulation,
    radii: np.ndarray,
    *,
    ball_radius: float,
    rng: np.random.Generator,
    buffer: float,
    max_tries: int,
    density: float = PARTICLE_DENSITY,
) -> int:
    """Place radii largest-first inside a ball, return the number placed"""
    placed = 0
    for radius in radii:
        mass = float(mass_from_radius(radius, density))
        for _ in range(max_tries):
            u = rng.uniform()
            r = ball_radius * u ** (1.0 / 3.0)
            ct = rng.uniform(-1.0, 1.0)
            st = np.sqrt(max(0.0, 1.0 - ct * ct))
            phi = rng.uniform(0.0, 2.0 * np.pi)
            x = r * st * np.cos(phi)
            y = r * st * np.sin(phi)
            z = r * ct
            if _fits(sim, x, y, z, float(radius), buffer):
                sim.add(
                    m=mass,
                    r=float(radius),
                    x=x,
                    y=y,
                    z=z,
                    vx=0.0,
                    vy=0.0,
                    vz=0.0,
                )
                placed += 1
                break
    return placed


def build_asteroid(
    target_mass: float,
    n_particles: int,
    *,
    seed: int,
    radius_ratio: float | None = None,
    slope: float = SIZE_SLOPE,
    density: float = PARTICLE_DENSITY,
    buffer: float = PARTICLE_BUFFER,
    max_tries: int = PLACEMENT_MAX_TRIES,
    dt: float = DT_SETTLE,
    softening: float = SOFTENING_DEFAULT,
    root_size: float = SETTLE_ROOT_SIZE,
    integrator: str = SETTLE_INTEGRATOR,
    size_scaling: str = "mass",
) -> tuple[rebound.Simulation, dict[str, Any]]:
    """
    Build one rubble pile with `n_particles` at `target_mass`

    The radii are drawn from the truncated power law, then multiplied by a
        single constant so that `sum(m_i) = target_mass` exactly
    A constant rescaling preserves the slope and the shape of the distribution.
    The particles are then placed largest-first by rejection
        if a packing attempt cannot place all of them,
        the placement sphere is opened by 10% in volume and the attempt is repeated
    Failing after all attempts raise error
    """
    if radius_ratio is None:
        radius_ratio = auto_radius_ratio(
            n_particles, slope=slope, packing_fraction=PACKING_FRACTION
        )
    r_lo, r_hi = scaled_radius_range(
        target_mass,
        n_particles,
        size_scaling=size_scaling,
        radius_ratio=radius_ratio,
        slope=slope,
        density=density,
    )

    rng = np.random.default_rng(seed)
    radii = powerlaw_sizes(rng, slope, r_lo, r_hi, n_particles)
    raw_mass = float(np.asarray(mass_from_radius(radii, density)).sum())
    if raw_mass <= 0:
        raise RuntimeError("drawn radii have zero mass")
    mass_scale = (target_mass / raw_mass) ** (1.0 / 3.0)
    radii *= mass_scale
    r_lo_used = float(r_lo * mass_scale)
    r_hi_used = float(r_hi * mass_scale)

    # Largest first gives the best packing for a polydisperse law.
    radii.sort()
    radii = radii[::-1]

    r_ball_base = ball_radius_for_mass(target_mass, density=density)
    best: tuple[rebound.Simulation, int, float] | None = None
    for attempt in range(8):
        sim = make_fresh_simulation(
            dt=dt,
            softening=softening,
            root_size=root_size,
            integrator=integrator,
        )
        ball_radius = r_ball_base * (1.0 + 0.10 * attempt) ** (1.0 / 3.0)
        place_rng = np.random.default_rng(seed + 10_000 + attempt)
        placed = place_radii(
            sim,
            radii,
            ball_radius=ball_radius,
            rng=place_rng,
            buffer=buffer,
            max_tries=max_tries,
            density=density,
        )
        if best is None or placed > best[1]:
            best = (sim, placed, ball_radius)
        if placed == n_particles:
            break

    if best is None or best[1] != n_particles:
        placed = 0 if best is None else best[1]
        raise RuntimeError(
            f"could not place all {n_particles} particles for "
            f"M={target_mass:.3e} kg (best {placed}); increase N, lower the "
            f"packing fraction, or raise MAX_RMAX_OVER_RBALL"
        )

    sim, placed, ball_radius = best
    sim.move_to_com()
    positions = np.array(
        [[particle.x, particle.y, particle.z] for particle in sim.particles]
    )
    initial_extent = state_extent(positions)
    volumes = (4.0 / 3.0) * np.pi * radii**3
    packing_actual = float(volumes.sum() / ((4.0 / 3.0) * np.pi * ball_radius**3))
    total_mass = float(sum(particle.m for particle in sim.particles))
    info: dict[str, Any] = {
        "target_mass": float(target_mass),
        "target_n": int(n_particles),
        "placed": int(placed),
        "total_mass": total_mass,
        "mass_error": total_mass / target_mass - 1.0,
        "mass_scale": float(mass_scale),
        "r_lo": r_lo_used,
        "r_hi": r_hi_used,
        "r_lo_base": float(r_lo),
        "r_hi_base": float(r_hi),
        "radius_ratio": float(radius_ratio),
        "ball_radius": float(ball_radius),
        "packing_actual": packing_actual,
        "initial_extent": initial_extent,
        "slope": float(slope),
        "density": float(density),
        "seed": int(seed),
        "size_scaling": size_scaling,
    }
    return sim, info


# ==================================================================================== #
# Diagnostics
# ==================================================================================== #


def state_extent(positions: np.ndarray) -> float:
    """Return the largest side length of the particle bounding box"""
    return float(np.max(np.ptp(positions, axis=0)))


def _com_state(
    masses: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    total = float(masses.sum())
    if total <= 0:
        raise ValueError("total mass must be positive")
    pos = (masses[:, None] * positions).sum(axis=0) / total
    vel = (masses[:, None] * velocities).sum(axis=0) / total
    return np.asarray(pos), np.asarray(vel), total


def _bound_fraction(
    masses: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    *,
    total_mass: float,
) -> tuple[float, float, float]:
    """Return `bound mass fraction, bound number fraction, v_rms`"""
    com_pos, com_vel, _ = _com_state(masses, positions, velocities)
    rel_r = positions - com_pos
    rel_v = velocities - com_vel
    r = np.linalg.norm(rel_r, axis=1)
    v2 = (rel_v**2).sum(axis=1)
    energy = 0.5 * v2 - G * total_mass / np.maximum(r, 1e-9)
    bound = energy < 0.0
    mass_fraction = float(masses[bound].sum() / total_mass)
    number_fraction = float(bound.mean()) if bound.size else 0.0
    v_rms = float(np.sqrt(np.mean(v2))) if v2.size else 0.0
    return mass_fraction, number_fraction, v_rms


def body_diagnostics(
    sim: rebound.Simulation,
    label: str,
    *,
    compute_energy: bool = True,
) -> dict[str, float]:
    """Print and return the stability diagnostics of one body"""
    if sim.N == 0:
        log_debug(f"         [{label}] N=0")
        return {"n": 0.0}

    masses = np.array([particle.m for particle in sim.particles])
    positions = np.array(
        [[particle.x, particle.y, particle.z] for particle in sim.particles]
    )
    velocities = np.array(
        [[particle.vx, particle.vy, particle.vz] for particle in sim.particles]
    )
    radii = np.array([particle.r for particle in sim.particles])
    total_mass = float(masses.sum())
    com_pos, com_vel, _ = _com_state(masses, positions, velocities)
    rel_r = positions - com_pos
    rel_v = velocities - com_vel
    r = np.linalg.norm(rel_r, axis=1)
    bound_mass_fraction, bound_number_fraction, v_rms = _bound_fraction(
        masses, positions, velocities, total_mass=total_mass
    )
    extent = float(np.max(np.ptp(positions, axis=0)))
    r_max = float(r.max()) if r.size else 0.0
    v_esc = float(np.sqrt(2.0 * G * total_mass / r_max)) if r_max > 0 else float("inf")
    angular_momentum = (masses[:, None] * np.cross(rel_r, rel_v)).sum(axis=0)
    if compute_energy:
        try:
            energy = float(sim.energy())
        except Exception:
            energy = float("nan")
    else:
        energy = float("nan")

    energy_text = f"  E={energy:.3e} J" if np.isfinite(energy) else ""
    log_debug(
        f"         [{label}] N={sim.N}  M={total_mass:.3e} kg  "
        f"v_rms={v_rms:.3f} m/s  v_esc={v_esc:.3f} m/s  "
        f"bound(mass)={bound_mass_fraction:.3f}  "
        f"bound(N)={bound_number_fraction:.3f}  "
        f"r=[{radii.min():.1f}, {radii.max():.1f}] m  "
        f"extent={extent:.0f} m  |L|={np.linalg.norm(angular_momentum):.3e} kg m^2/s"
        f"{energy_text}",
    )
    if bound_mass_fraction < 0.90:
        log_warning(
            f"         only {bound_mass_fraction:.1%} of the mass is "
            f"bound after {label}; the pile may be disrupted",
        )
    return {
        "n": float(sim.N),
        "mass": total_mass,
        "v_rms": v_rms,
        "v_esc": v_esc,
        "bound_mass_fraction": bound_mass_fraction,
        "bound_number_fraction": bound_number_fraction,
        "extent": extent,
        "energy": energy,
    }


def state_diagnostics(
    state: np.ndarray,
    *,
    density: float,
    label: str,
) -> dict[str, float]:
    """Stability diagnostics for a cached `(N, 7)` state"""
    if state.ndim != 2 or state.shape[1] != 7:
        raise ValueError("state must have shape (N, 7)")
    radii = state[:, 6]
    masses = np.asarray(mass_from_radius(radii, density), float)
    positions = state[:, :3]
    velocities = state[:, 3:6]
    total_mass = float(masses.sum())
    com_pos, com_vel, _ = _com_state(masses, positions, velocities)
    rel_r = positions - com_pos
    rel_v = velocities - com_vel
    r = np.linalg.norm(rel_r, axis=1)
    v2 = (rel_v**2).sum(axis=1)
    energy = 0.5 * v2 - G * total_mass / np.maximum(r, 1e-9)
    bound_mass_fraction = float(masses[energy < 0].sum() / total_mass)
    v_rms = float(np.sqrt(np.mean(v2)))
    r_max = float(r.max())
    v_esc = float(np.sqrt(2.0 * G * total_mass / r_max))
    extent = state_extent(positions)
    log_debug(
        f"         [{label}] N={len(radii)}  M={total_mass:.3e} kg  "
        f"v_rms={v_rms:.3f} m/s  v_esc={v_esc:.3f} m/s  "
        f"bound(mass)={bound_mass_fraction:.3f}  "
        f"r=[{radii.min():.1f}, {radii.max():.1f}] m  extent={extent:.0f} m",
    )
    if bound_mass_fraction < 0.90:
        log_warning(
            f"         cached state has only {bound_mass_fraction:.1%} bound mass",
        )
    return {
        "mass": total_mass,
        "v_rms": v_rms,
        "v_esc": v_esc,
        "bound_mass_fraction": bound_mass_fraction,
        "extent": extent,
    }


# ==================================================================================== #
# Power-law validation
# ==================================================================================== #


def powerlaw_cdf(
    radius: np.ndarray,
    slope: float,
    r_min: float,
    r_max: float,
) -> np.ndarray:
    """CDF of p(r) propto r^slope on `[r_min, r_max]`"""
    x = np.asarray(radius, float)
    exponent = slope + 1.0
    if np.isclose(exponent, 0.0):
        value = np.log(x / r_min) / np.log(r_max / r_min)
    else:
        value = (x**exponent - r_min**exponent) / (r_max**exponent - r_min**exponent)
    return np.clip(value, 0.0, 1.0)


def check_powerlaw(
    radii: np.ndarray,
    slope_input: float,
    r_min: float,
    r_max: float,
    *,
    label: str = "",
) -> dict[str, Any] | None:
    """
    Compare an empirical radius sample with p(r) propto r^slope

    Prints the fitted PDF and survival-function slopes and runs a one-sample
        KS test against the truncated input law
    A p-value below 0.01 is reported as a warning
    """
    from scipy.stats import kstest

    r = np.asarray(radii, float)
    n = r.size
    log_debug("")
    log_debug(f"Power-law check: {label}")
    log_debug(f"         N sampled       : {n}")
    log_debug(f"         r range (input) : [{r_min:8.1f}, {r_max:8.1f}] m")
    log_debug(f"         r range (actual): [{r.min():8.1f}, {r.max():8.1f}] m")
    log_debug(f"         input slope     : {slope_input:+.3f}")
    if n < 20:
        log_warning("         too few samples for a fit")
        return None

    nbins = 15
    edges = np.logspace(np.log10(r.min() * 0.99), np.log10(r.max() * 1.01), nbins + 1)
    counts, _ = np.histogram(r, bins=edges)
    widths = np.diff(edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    pdf = counts / widths / n
    ok = counts > 0
    slope_pdf = float("nan")
    if ok.sum() >= 2:
        slope_pdf = float(np.polyfit(np.log10(centers[ok]), np.log10(pdf[ok]), 1)[0])

    r_sorted = np.sort(r)
    survival = (n - np.arange(n)) / n
    lo, hi = int(0.1 * n), int(0.9 * n)
    slope_sf = float(
        np.polyfit(np.log10(r_sorted[lo:hi]), np.log10(survival[lo:hi]), 1)[0]
    )

    result = kstest(r, lambda x: powerlaw_cdf(x, slope_input, r_min, r_max))
    log_info(
        f"         fit  pdf slope : {slope_pdf:+.3f} (expected {slope_input:+.3f})"
    )
    log_info(
        f"         fit  SF  slope : {slope_sf:+.3f} (expected {slope_input + 1:+.3f})"
    )
    log_info(
        f"         KS vs input law: D={result.statistic:.5f}, p={result.pvalue:.5g}"
    )
    if result.pvalue < 0.01:
        log_warning(
            "         radius sample is inconsistent with the input "
            "power law at p < 0.01",
        )
    return {
        "n": n,
        "slope_input": slope_input,
        "slope_pdf": slope_pdf,
        "slope_sf": slope_sf,
        "ks_d": float(result.statistic),
        "ks_p": float(result.pvalue),
        "r_lo": float(r.min()),
        "r_hi": float(r.max()),
    }


# ==================================================================================== #
# Settling
# ==================================================================================== #


def align_root_size(
    sim: rebound.Simulation,
    *,
    minimum: float = 1.0e6,
    horizon: float | None = None,
) -> None:
    """
    Make the tree root box enclose every particle

    `horizon` is the time until the next root-box update
    """
    if sim.N == 0:
        return
    max_r = max(
        max(abs(particle.x), abs(particle.y), abs(particle.z))
        for particle in sim.particles
    )
    v_max = max(
        float(np.sqrt(particle.vx**2 + particle.vy**2 + particle.vz**2))
        for particle in sim.particles
    )
    margin_time = sim.dt if horizon is None else float(horizon)
    needed = 2.0 * (max_r + v_max * margin_time) * 1.5
    if sim.root_size < needed:
        sim.root_size = max(needed, minimum)


def adaptive_settle_dt(
    sim: rebound.Simulation,
    *,
    dt_max: float,
    safety: float = SETTLE_DT_SAFETY,
    dt_min: float = SETTLE_DT_MIN,
) -> float:
    """
    A conservative fixed-step safety control for `leapfrog` integrator
    """
    if sim.N == 0:
        return dt_max
    r_min = min(float(particle.r) for particle in sim.particles)
    v_max = max(
        float(np.sqrt(particle.vx**2 + particle.vy**2 + particle.vz**2))
        for particle in sim.particles
    )
    if v_max <= 0.0:
        return dt_max
    return min(dt_max, max(safety * r_min / v_max, dt_min))


def process_asteroid(job: dict[str, Any]) -> dict[str, Any]:
    """
    Build, settle, validate, and cache one asteroid -> top level func
    """
    target_mass = float(job["target_mass"])
    target_n = int(job["target_n"])
    settled_file = Path(job["settled_file"])
    settle_time = float(job["settle_time"])
    dt = float(job["dt"])
    integrator = str(job["integrator"])
    density = float(job["density"])
    radius_ratio = float(job["radius_ratio"])
    slope = float(job["slope"])
    softening = float(job["softening"])
    buffer = float(job["buffer"])
    root_size = float(job["root_size"])
    seed = int(job["seed"])

    log_info(f"\n          Process Asteroid - {settled_file.name}")
    log_info(
        f"         M={target_mass:.3e} kg  N={target_n}  seed={seed}  "
        f"integrator={integrator}  dt={dt:g} s  settle={settle_time:g} s",
    )

    sim, info = build_asteroid(
        target_mass,
        target_n,
        seed=seed,
        radius_ratio=radius_ratio,
        slope=slope,
        density=density,
        buffer=buffer,
        dt=dt,
        softening=softening,
        root_size=root_size,
        integrator=integrator,
    )
    body_diagnostics(sim, "after placement", compute_energy=False)
    log_debug(
        f"         placement: ball R={info['ball_radius']:.0f} m  "
        f"packing={info['packing_actual']:.3f}  "
        f"r=[{info['r_lo']:.1f}, {info['r_hi']:.1f}] m  "
        f"mass error={info['mass_error']:+.2e}",
    )

    energy_before = float(sim.energy())
    t0 = time.time()
    dt_min_used = float(dt)
    for chunk in range(SETTLE_CHUNKS):
        target_time = settle_time * (chunk + 1) / SETTLE_CHUNKS
        while sim.t < target_time - 1e-9:
            if integrator == "leapfrog":
                sim.dt = adaptive_settle_dt(sim, dt_max=dt)
                dt_min_used = min(dt_min_used, float(sim.dt))
                horizon = SETTLE_MAX_SUBSTEPS * sim.dt
            else:
                # IAS15 is adaptive; dt is irrelevant, but the root box still
                # needs a finite update horizon.
                horizon = max(1000.0, SETTLE_MAX_SUBSTEPS * dt)
            # Include the whole sub-chunk in the root-box margin before
            # integrating, then align again afterwards.
            align_root_size(sim, horizon=horizon)
            t_next = min(sim.t + horizon, target_time)
            sim.integrate(t_next)
            align_root_size(sim)
        body_diagnostics(
            sim,
            f"settle {chunk + 1}/{SETTLE_CHUNKS} (dt={sim.dt:.3g} s)",
            compute_energy=False,
        )
    wall_time = time.time() - t0
    energy_after = float(sim.energy())
    if np.isfinite(energy_before) and np.isfinite(energy_after):
        relative_change = (
            (energy_after - energy_before) / abs(energy_before)
            if energy_before != 0.0
            else float("nan")
        )
        if np.isfinite(relative_change) and relative_change > 1.0e-2:
            log_warning(
                "         tree-energy estimate increased by more than 1% "
                "during settling; check dt/softening/collision settings",
            )
        log_debug(
            f"         energy change: {energy_after - energy_before:+.3e} J "
            f"({relative_change:+.3e} relative)",
        )
    log_debug(
        f"         settle wall time: {wall_time:.1f} s "
        f"({wall_time / max(settle_time, 1.0) * 1e6:.2f} us/sim_s)",
    )

    sim.move_to_com()
    settled_diag = body_diagnostics(sim, "settled")
    actual_mass = float(sum(particle.m for particle in sim.particles))
    v_esc_core = float(np.sqrt(2.0 * G * actual_mass / info["ball_radius"]))
    if (
        settled_diag["bound_mass_fraction"] < MIN_SETTLED_BOUND_FRACTION
        or settled_diag["extent"] > MAX_SETTLED_EXTENT_FACTOR * info["initial_extent"]
        or settled_diag["v_rms"] > v_esc_core
    ):
        raise RuntimeError(
            "settled body is not stable: "
            f"bound(mass)={settled_diag['bound_mass_fraction']:.3f} "
            f"(need >= {MIN_SETTLED_BOUND_FRACTION}), "
            f"extent={settled_diag['extent']:.3e} m "
            f"(limit {MAX_SETTLED_EXTENT_FACTOR} x "
            f"{info['initial_extent']:.3e} m), "
            f"v_rms={settled_diag['v_rms']:.3f} m/s "
            f"(v_esc_core={v_esc_core:.3f} m/s).  Check --dt-settle, "
            "--settle-time, packing, or N"
        )

    state = np.zeros((sim.N, 7))
    for index, particle in enumerate(sim.particles):
        state[index] = [
            particle.x,
            particle.y,
            particle.z,
            particle.vx,
            particle.vy,
            particle.vz,
            particle.r,
        ]
    rel_mass_err = actual_mass / target_mass - 1.0
    if abs(rel_mass_err) > 1.0e-6:
        raise RuntimeError(
            f"settled mass is off by {rel_mass_err:+.3e} (should be exact)"
        )

    meta: dict[str, Any] = {
        **collect_metadata(),
        "cache_version": VERSION,
        "target_mass": target_mass,
        "actual_mass": actual_mass,
        "rel_mass_err": rel_mass_err,
        "target_n": target_n,
        "placed": info["placed"],
        "seed": seed,
        "slope": slope,
        "density": density,
        "radius_ratio": radius_ratio,
        "r_lo": info["r_lo"],
        "r_hi": info["r_hi"],
        "r_lo_base": info["r_lo_base"],
        "r_hi_base": info["r_hi_base"],
        "ball_radius": info["ball_radius"],
        "packing_actual": info["packing_actual"],
        "packing_fraction": PACKING_FRACTION,
        "softening": softening,
        "buffer": buffer,
        "settle_time": settle_time,
        "dt_requested": dt,
        "dt_min_used": dt_min_used,
        "initial_extent": info["initial_extent"],
        "settled_bound_mass_fraction": settled_diag["bound_mass_fraction"],
        "settled_extent": settled_diag["extent"],
        "settled_v_rms": settled_diag["v_rms"],
        "dt": dt,
        "integrator": integrator,
        "root_size": root_size,
        "size_scaling": info["size_scaling"],
        "settle_wall_time": wall_time,
    }
    atomic_savez(
        settled_file,
        state=state,
        density=np.array(density),
        target_mass=np.array(target_mass),
        actual_mass=np.array(actual_mass),
        rel_mass_err=np.array(rel_mass_err),
        target_n=np.array(target_n),
        r_lo=np.array(info["r_lo"]),
        r_hi=np.array(info["r_hi"]),
        slope=np.array(slope),
        seed=np.array(seed),
        meta_json=np.array(json.dumps(meta, sort_keys=True)),
    )
    log_info(f"         Recorded state -> {settled_file}")
    return {
        "settled_file": str(settled_file),
        "target_mass": target_mass,
        "target_n": target_n,
        "actual_mass": actual_mass,
        "rel_mass_err": rel_mass_err,
        "meta": meta,
    }


def settle_bodies(
    jobs: list[dict[str, Any]],
    *,
    parallel: bool = True,
    log_path: str | Path | None = None,
    show_progress: bool = True,
) -> None:
    """Run `process_asteroid` serially or in a process pool"""
    if not jobs:
        return

    # De-duplicate by cache path: identical bodies must not be written
    # concurrently to the same file.
    unique: dict[Path, dict[str, Any]] = {}
    for job in jobs:
        path = Path(job["settled_file"])
        if path in unique:
            log_info(
                f"         [settle] duplicate cache path {path.name}; settling it once",
            )
            continue
        unique[path] = job
    jobs = list(unique.values())

    if not parallel or len(jobs) == 1:
        log_info("         [settle] serial execution")
        for job in jobs:
            process_asteroid(job)
        return

    workers = min(len(jobs), max(1, os.cpu_count() or 1))
    log_info(
        f"         [settle] parallel execution: {len(jobs)} bodies across "
        f"{workers} worker process(es)",
    )
    t0 = time.time()
    initializer = (
        functools.partial(configure_worker_logging, str(log_path))
        if log_path is not None
        else None
    )
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=initializer,
        initargs=(),
    ) as pool:
        futures = {pool.submit(process_asteroid, job): job for job in jobs}
        for future in tqdm(
            as_completed(futures),
            total=len(jobs),
            desc="settling",
            ncols=88,
            disable=not show_progress,
        ):
            job = futures[future]
            name = Path(job["settled_file"]).name
            try:
                future.result()
                log_debug(
                    f"         [settle] finished {name} "
                    f"(elapsed {time.time() - t0:.1f} s)",
                )
            except Exception as exc:
                log_warning(f"         [settle] FAILED {name}: {exc!r}")
                raise
    log_info(
        f"         [settle] all bodies settled in {time.time() - t0:.1f} s",
    )


def list_settled_caches(directory: Path = DATA_DIR) -> None:
    """Print the settled-body library, including its build recipe"""
    files = sorted(
        directory.glob(f"{SETTLED_STEM}_M*.npz"),
        key=lambda path: path.stat().st_mtime,
    )
    if not files:
        log_info(f"no settled caches in {directory}")
        return
    log_info(
        f"{'mass [kg]':>11s} {'N':>6s} {'seed':>5s} {'rho':>7s} "
        f"{'slope':>6s} {'soft [m]':>9s} {'settle [s]':>11s} "
        f"{'dt':>5s} {'integrator':>10s} {'file'}"
    )
    for path in files:
        try:
            with np.load(path, allow_pickle=False) as data:
                if "meta_json" not in data:
                    log_info(
                        f"{'(legacy)':>11s} {'':>6s} {'':>5s} {'':>7s} "
                        f"{'':>6s} {'':>9s} {'':>11s} {'':>5s} {'':>10s} "
                        f"{path.name}"
                    )
                    continue
                meta = json.loads(data["meta_json"].item())
                log_info(
                    f"{meta.get('target_mass', float('nan')):11.3e} "
                    f"{meta.get('target_n', -1):6d} "
                    f"{meta.get('seed', -1):5d} "
                    f"{meta.get('density', float('nan')):7.1f} "
                    f"{meta.get('slope', float('nan')):6.2f} "
                    f"{meta.get('softening', float('nan')):9.1f} "
                    f"{meta.get('settle_time', float('nan')):11.0f} "
                    f"{meta.get('dt_requested', meta.get('dt', float('nan'))):5.1f} "
                    f"{meta.get('integrator', '?'):>10s} {path.name}"
                )
        except Exception as exc:
            log_info(f"{'(unreadable)':>11s} {path.name}: {exc!r}")


def load_settled_asteroid(
    path: str | Path,
    *,
    expected: dict[str, Any] | None = None,
    density: float | None = None,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """
    Load and validate a cached `(N, 7)` state

    Raises `ValueError` if the cache
        has no metadata
        is malformed
        does not match `expected` (mass, N, slope, density, seed)
    This prevents silently reusing a state built with different physics
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        if "state" not in data:
            raise ValueError(f"{path} has no 'state' array")
        state = np.asarray(data["state"], float)
        target_mass = float(data["target_mass"])
        target_n = int(data["target_n"])
        stored_density = float(data["density"])
        slope = float(data["slope"])
        seed = int(data["seed"])
        if "meta_json" not in data:
            raise ValueError(f"{path} has no metadata; re-settle with main.py")
        meta = json.loads(data["meta_json"].item())

    if state.ndim != 2 or state.shape[1] != 7:
        raise ValueError(f"{path}: state must have shape (N, 7), got {state.shape}")
    if state.shape[0] != target_n:
        raise ValueError(
            f"{path}: state has {state.shape[0]} particles, metadata says {target_n}"
        )
    if not np.all(np.isfinite(state)):
        raise ValueError(f"{path}: state contains non-finite values")
    if np.any(state[:, 6] <= 0):
        raise ValueError(f"{path}: non-positive particle radius")

    actual_mass = float(np.asarray(mass_from_radius(state[:, 6], stored_density)).sum())
    if abs(actual_mass / target_mass - 1.0) > 1.0e-6:
        raise ValueError(
            f"{path}: radius-derived mass {actual_mass:.6e} does not match "
            f"target {target_mass:.6e}"
        )

    if expected is not None:
        checks = {
            "target_mass": target_mass,
            "target_n": target_n,
            "slope": slope,
            "density": stored_density,
            "seed": seed,
        }
        for key, value in checks.items():
            if key not in expected:
                continue
            want = expected[key]
            if isinstance(want, float):
                same = np.isclose(float(value), float(want), rtol=1e-12, atol=0.0)
            else:
                same = value == want
            if not same:
                raise ValueError(
                    f"{path}: cached {key}={value!r} != requested {want!r}"
                )

    if density is not None and not np.isclose(density, stored_density):
        raise ValueError(
            f"{path}: cached density {stored_density} != requested {density}"
        )

    extent = state_extent(state[:, :3])
    return state, extent, meta


# ==================================================================================== #
# Scene construction
# ==================================================================================== #


def get_scene_params(
    a_extent1: float,
    a_extent2: float,
    a_extent3: float,
    *,
    separation_factor: float = SEPARATION_FACTOR,
    box_size_factor: float = BOX_SIZE_FACTOR,
) -> tuple[float, float, float]:
    """Return `mean_extent, separation, box_size` for the fixed scene"""
    a_extent = (a_extent1 + a_extent2 + a_extent3) / 3.0
    separation = separation_factor * a_extent
    return a_extent, separation, box_size_factor * separation


def add_settled_at_point_state(
    sim: rebound.Simulation,
    state: np.ndarray,
    point_state: np.ndarray,
    *,
    target_mass: float,
    density: float = PARTICLE_DENSITY,
    mass_mode: str = "scale",
) -> None:
    """
    Place one rubble pile so its COM sits at the point-mass state
    """
    state = np.asarray(state, float)
    point_state = np.asarray(point_state, float)
    if state.ndim != 2 or state.shape[1] != 7:
        raise ValueError("state must have shape (N, 7)")
    if point_state.shape != (6,):
        raise ValueError("point_state must have shape (6,)")

    radii = state[:, 6]
    masses = np.asarray(mass_from_radius(radii, density), float)
    com_pos, com_vel, total = _com_state(masses, state[:, :3], state[:, 3:6])
    if mass_mode == "scale":
        if target_mass <= 0:
            raise ValueError("mass_mode='scale' requires target_mass > 0")
        masses *= target_mass / total
        log_info(
            f"         [mass] target={target_mass:.4e}  "
            f"rubble={total:.4e}  scale={target_mass / total:.6f}",
        )
    elif mass_mode != "preserve":
        raise ValueError("mass_mode must be 'preserve' or 'scale'")

    positions = state[:, :3] - com_pos + point_state[:3]
    velocities = state[:, 3:6] - com_vel + point_state[3:6]
    for index in range(state.shape[0]):
        sim.add(
            m=float(masses[index]),
            r=float(radii[index]),
            x=float(positions[index, 0]),
            y=float(positions[index, 1]),
            z=float(positions[index, 2]),
            vx=float(velocities[index, 0]),
            vy=float(velocities[index, 1]),
            vz=float(velocities[index, 2]),
        )


def add_settled_fixed(
    sim: rebound.Simulation,
    state: np.ndarray,
    offset: np.ndarray,
    bulk_velocity: np.ndarray,
    *,
    density: float = PARTICLE_DENSITY,
) -> None:
    """Place a rubble pile at a fixed offset with a bulk velocity"""
    state = np.asarray(state, float)
    radii = state[:, 6]
    masses = np.asarray(mass_from_radius(radii, density), float)
    com_pos, com_vel, _ = _com_state(masses, state[:, :3], state[:, 3:6])
    positions = state[:, :3] - com_pos + np.asarray(offset, float)
    velocities = state[:, 3:6] - com_vel + np.asarray(bulk_velocity, float)
    for index in range(state.shape[0]):
        sim.add(
            m=float(masses[index]),
            r=float(radii[index]),
            x=float(positions[index, 0]),
            y=float(positions[index, 1]),
            z=float(positions[index, 2]),
            vx=float(velocities[index, 0]),
            vy=float(velocities[index, 1]),
            vz=float(velocities[index, 2]),
        )


def build_scene_fixed(
    state1: np.ndarray,
    state2: np.ndarray,
    state3: np.ndarray,
    *,
    density: float,
    dt: float,
    integrator: str,
    softening: float,
    bullet_speed: float = BULLET_SPEED,
) -> tuple[rebound.Simulation, float, float, float, int, float, float]:
    """
    Build the fixed-geometry scene

    Returns `sim, a_bin, t_bin, t_impact, n_host, separation, a_extent`
    """
    extent1 = state_extent(state1[:, :3])  # host 1
    extent2 = state_extent(state2[:, :3])  # host 2
    extent3 = state_extent(state3[:, :3])  # bullet
    a_extent, separation, box_size = get_scene_params(extent1, extent2, extent3)

    sim = make_fresh_simulation(
        dt=dt,
        softening=softening,
        root_size=box_size,
        integrator=integrator,
    )
    add_settled_fixed(
        sim,
        state1,
        OFFSET_1 * separation,
        BLKVEL_1 * bullet_speed,
        density=density,
    )
    add_settled_fixed(
        sim,
        state2,
        OFFSET_2 * separation,
        BLKVEL_2 * bullet_speed,
        density=density,
    )
    add_settled_fixed(
        sim,
        state3,
        OFFSET_3 * separation,
        BLKVEL_3 * bullet_speed,
        density=density,
    )
    align_root_size(sim)
    a_bin = float(separation)
    m_host = float(
        np.asarray(mass_from_radius(state1[:, 6], density)).sum()
        + np.asarray(mass_from_radius(state2[:, 6], density)).sum()
    )
    t_bin = float(2.0 * np.pi * np.sqrt(a_bin**3 / (G * m_host)))
    t_impact = float((separation - a_extent) / bullet_speed)
    return sim, a_bin, t_bin, t_impact, state1.shape[0], separation, a_extent


def build_scene_realisation(
    realisation: dict[str, Any],
    states: tuple[np.ndarray, np.ndarray, np.ndarray],
    point_states: np.ndarray,
    target_masses: tuple[float, float, float],
    *,
    density: float,
    dt: float,
    integrator: str,
    softening: float,
) -> tuple[rebound.Simulation, float, float, int]:
    """
    Build the scene from a recorded point-mass state

    Returns `sim, a_bin, t_bin, n_host`
    """
    if point_states.shape != (3, 6):
        raise ValueError(
            f"point_states must have shape (3, 6), got {point_states.shape}"
        )
    a_bin = float(realisation["a_bin"])
    t_bin = float(realisation["T_bin"])

    sim = make_fresh_simulation(
        dt=dt,
        softening=softening,
        root_size=HIT_ROOT_SIZE,
        integrator=integrator,
    )
    for state, point_state, target_mass in zip(
        states, point_states, target_masses, strict=True
    ):
        add_settled_at_point_state(
            sim,
            state,
            point_state,
            target_mass=float(target_mass),
            density=density,
            mass_mode="scale",
        )
    align_root_size(sim)
    return sim, a_bin, t_bin, states[0].shape[0]


# ==================================================================================== #
# Bound-debris checker
# ==================================================================================== #


def host_state(
    sim: rebound.Simulation,
    n_host: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Return the COM position, COM velocity, and mass of the larger host

    The host is the leading `n_host` particles
    If particles have been removed, fall back to the single most massive particle
    """
    if sim.N == 0:
        return np.zeros(3), np.zeros(3), 0.0
    if n_host <= sim.N and n_host > 0:
        indices = list(range(n_host))
    else:
        indices = [max(range(sim.N), key=lambda i: sim.particles[i].m)]  # type: ignore
    masses = np.array([sim.particles[i].m for i in indices])  # type: ignore
    positions = np.array(
        [[sim.particles[i].x, sim.particles[i].y, sim.particles[i].z] for i in indices]  # type: ignore
    )
    velocities = np.array(
        [
            [sim.particles[i].vx, sim.particles[i].vy, sim.particles[i].vz]  # type: ignore
            for i in indices
        ]
    )
    com_pos, com_vel, total = _com_state(masses, positions, velocities)
    return com_pos, com_vel, total


def bound_debris_stats(
    sim: rebound.Simulation,
    *,
    n_host: int,
    radius: float,
    label: str,
) -> dict[str, float]:
    """
    Crude bound-debris check

    Radially bound if it lies within `radius` of the host COM.
    Energetically bound if `0.5 v^2 - G M_host / r < 0` in the host rest frame.
    Both mass fractions are reported, with the energy criterion the more physical one.
    """
    host_pos, host_vel, host_mass = host_state(sim, n_host)
    if n_host >= sim.N or host_mass <= 0:
        stats = {
            "n_host": float(n_host),
            "n_debris": 0.0,
            "mass_debris": 0.0,
            "n_within": 0.0,
            "mass_within": 0.0,
            "f_bound_radius": 0.0,
            "f_bound_energy": 0.0,
            "f_bound_both": 0.0,
            "median_r": 0.0,
        }
        log_debug(f"         [bound:{label}] no debris particles")
        return stats

    debris_mass = 0.0
    within_mass = 0.0
    bound_mass = 0.0
    both_mass = 0.0
    radii: list[float] = []
    n_within = 0
    for index in range(n_host, sim.N):
        particle = sim.particles[index]
        rel_r = np.array([particle.x, particle.y, particle.z]) - host_pos  # type: ignore
        rel_v = np.array([particle.vx, particle.vy, particle.vz]) - host_vel  # type: ignore
        r = float(np.linalg.norm(rel_r))
        v2 = float(np.dot(rel_v, rel_v))
        energy = 0.5 * v2 - G * host_mass / max(r, 1e-9)
        mass = float(particle.m)  # type: ignore
        debris_mass += mass
        radii.append(r)
        is_within = r <= radius
        is_bound = energy < 0.0
        if is_within:
            within_mass += mass
            n_within += 1
        if is_bound:
            bound_mass += mass
        if is_within and is_bound:
            both_mass += mass

    stats = {
        "n_host": float(n_host),
        "n_debris": float(sim.N - n_host),
        "mass_debris": debris_mass,
        "n_within": float(n_within),
        "mass_within": within_mass,
        "f_bound_radius": within_mass / debris_mass if debris_mass else 0.0,
        "f_bound_energy": bound_mass / debris_mass if debris_mass else 0.0,
        "f_bound_both": both_mass / debris_mass if debris_mass else 0.0,
        "median_r": float(np.median(radii)) if radii else 0.0,
    }
    log_debug(
        f"         [bound:{label}] host M={host_mass:.3e} kg  "
        f"debris N={int(stats['n_debris'])}  M={debris_mass:.3e} kg  "
        f"within<{radius / 1000:.1f} km={stats['f_bound_radius']:.3f}  "
        f"E<0={stats['f_bound_energy']:.3f}  both={stats['f_bound_both']:.3f}  "
        f"median r={stats['median_r'] / 1000:.1f} km",
    )
    return stats


# ==================================================================================== #
# Rendering
# ==================================================================================== #


def compute_viewport(
    sim: rebound.Simulation,
    *,
    pad_factor: float = 1.10,
    min_half: float = 1.0,
    max_half: float | None = None,
    percentile: float = VIEW_PERCENTILE,
) -> tuple[float, float, float]:
    """
    Return `half_extent, cx, cy` for the xy viewport

    Centre is the mass-weighted COM
    Half-size is the requested percentile of particle distances from the COM,
        times `pad_factor`, then clamped to `[min_half, max_half]`
    """
    if sim.N == 0:
        return min_half, 0.0, 0.0
    masses: np.ndarray = np.fromiter(
        (p.m for p in sim.particles), dtype=float, count=sim.N
    )
    xs: np.ndarray = np.fromiter((p.x for p in sim.particles), dtype=float, count=sim.N)
    ys: np.ndarray = np.fromiter((p.y for p in sim.particles), dtype=float, count=sim.N)
    total = float(masses.sum())
    if total <= 0:
        return min_half, 0.0, 0.0
    cx = float((masses * xs).sum() / total)
    cy = float((masses * ys).sum() / total)
    distance = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    half = float(np.percentile(distance, percentile)) * pad_factor
    half = max(half, min_half)
    if max_half is not None:
        half = min(half, max_half)
    return half, cx, cy


def make_canvas(
    sim: rebound.Simulation,
    *,
    fig_size_in: float = FIG_SIZE_IN,
    dpi: int = RENDER_DPI,
) -> tuple[Figure, Axes, FigureCanvasAgg, list[patches.Circle]]:
    """Build one fig, ax, and one Circle artist per particle"""
    fig, ax = plt.subplots(figsize=(fig_size_in, fig_size_in), dpi=dpi)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    circles = [
        patches.Circle(
            (particle.x, particle.y),
            radius=max(particle.r, 1.0),
            facecolor="darkgray",
            edgecolor="black",
            linewidth=0.4,
            alpha=0.85,
        )
        for particle in sim.particles
    ]
    for circle in circles:
        ax.add_patch(circle)
    canvas = FigureCanvasAgg(fig)
    return fig, ax, canvas, circles


def render_frame_rgb(
    canvas: FigureCanvasAgg,
    ax: Axes,
    circles: list[patches.Circle],
    sim: rebound.Simulation,
    t: float,
    index: int,
    *,
    min_half: float,
    max_half: float | None,
    view_percentile: float,
    grayscale: bool = True,
) -> np.ndarray:
    """Update the artists and viewport and return an RGB or grayscale frame"""
    for circle, particle in zip(circles, sim.particles, strict=False):
        circle.center = (particle.x, particle.y)
    half, cx, cy = compute_viewport(
        sim,
        min_half=min_half,
        max_half=max_half,
        percentile=view_percentile,
    )
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_title(
        f"t = {t:7.1f} s ({t / 60:6.2f} min) | "
        f"+/-{half / 1000:6.1f} km | frame {index:04d}"
    )
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    if grayscale:
        return rgba[..., 0].copy()
    return rgba[..., :3].copy()


def save_frames_npz(frames: list[np.ndarray], path: str | Path) -> tuple[int, ...]:
    """Persist `N, H, W` grayscale or `N, H, W, 3` RGB frames"""
    if not frames:
        raise ValueError("no frames to save")
    array = np.stack(frames, axis=0)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.savez(handle, frames=array)
    tmp.replace(path)
    return array.shape


def load_frames_npz(path: str | Path) -> np.ndarray:
    """Read grayscale or RGB frames written by `save_frames_npz`"""
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["frames"])


def chunk_sort_key(path: Path) -> int:
    """Sort by numeric chunk order"""
    try:
        return int(Path(path).stem.split("_")[-1])
    except (ValueError, IndexError):
        return 10**18


def build_gif_from_chunks(
    chunk_paths: list[Path],
    out_path: str | Path,
    *,
    duration_ms: int = DURATION_MS,
    quantize_colors: int = GIF_QUANTIZE_COLORS,
    show_progress: bool = True,
) -> None:
    """Read chunk files one at a time and encode a GIF atomically"""
    if not chunk_paths:
        log_info("\n          No chunks to build gif.\n")
        return
    if duration_ms <= 0:
        raise ValueError("duration_ms must be positive")
    if not 2 <= quantize_colors <= 256:
        raise ValueError("quantize_colors must be between 2 and 256")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil_frames: list[Image.Image] = []
    total = 0
    for chunk_index, path in enumerate(
        tqdm(
            chunk_paths,
            desc="encoding",
            ncols=88,
            disable=not show_progress,
        )
    ):
        frames = load_frames_npz(path)
        for frame in frames:
            pil_frames.append(Image.fromarray(frame).quantize(colors=quantize_colors))
            total += 1
        log_debug(
            f"         GIF assembly: chunk {chunk_index + 1}/"
            f"{len(chunk_paths)} {total} frames",
        )
    if not pil_frames:
        raise RuntimeError("chunk files contained no frames")

    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    pil_frames[0].save(
        str(tmp),
        format="GIF",
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    tmp.replace(out_path)
    log_info(f"\n         Saved GIF: {out_path}  ({total} frames)")


class FFmpegVideoWriter:
    """Stream grayscale or RGB frames to H.264"""

    def __init__(
        self,
        out_path: str | Path,
        width: int,
        height: int,
        *,
        fps: int = VIDEO_FPS,
        crf: int = VIDEO_CRF,
        grayscale: bool = True,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("frame dimensions must be positive")
        if fps <= 0:
            raise ValueError("fps must be positive")
        if not 0 <= crf <= 51:
            raise ValueError("crf must be between 0 and 51")
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError(
                "ffmpeg not found on PATH; install ffmpeg to encode video"
            )

        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.tmp_path = self.out_path.with_name(self.out_path.stem + ".part.mp4")
        self.tmp_path.unlink(missing_ok=True)
        self.width = width
        self.height = height
        self.grayscale = grayscale
        self.total = 0
        self.closed = False

        command = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "gray" if grayscale else "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-vcodec",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-f",
            "mp4",
            str(self.tmp_path),
        ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg stdin pipe was not created")

    def write(self, frame: np.ndarray) -> None:
        """Write one `H, W` grayscale or `H, W, 3` RGB frame"""
        expected = (
            (self.height, self.width)
            if self.grayscale
            else (self.height, self.width, 3)
        )
        if frame.shape != expected:
            raise ValueError(f"frame shape {frame.shape} != {expected}")
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg stdin pipe is closed")
        self.process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        self.total += 1

    def close(self) -> None:
        """Close stream, finalise the MP4, and log the result"""
        if self.closed:
            return
        self.closed = True
        if self.process.stdin is not None:
            self.process.stdin.close()
        stderr = (
            self.process.stderr.read().decode("utf-8", errors="replace")
            if self.process.stderr is not None
            else ""
        )
        return_code = self.process.wait()
        if return_code != 0:
            self.tmp_path.unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg failed ({return_code}): {stderr.strip()}")
        self.tmp_path.replace(self.out_path)
        log_info(f"\n         Saved video: {self.out_path}  ({self.total} frames)")

    def abort(self) -> None:
        """Kill ffmpeg and discard the partial file after an error"""
        if self.closed:
            return
        self.closed = True
        if self.process.poll() is None:
            self.process.kill()
        self.tmp_path.unlink(missing_ok=True)


def build_video_from_chunks(
    chunk_paths: list[Path],
    out_path: str | Path,
    *,
    fps: int = VIDEO_FPS,
    crf: int = VIDEO_CRF,
    show_progress: bool = True,
) -> None:
    """Stream grayscale or RGB frame chunks to H.264 with ffmpeg"""
    if not chunk_paths:
        log_info("\n          No chunks to build video.\n")
        return
    first = load_frames_npz(chunk_paths[0])
    if first.ndim == 3:
        grayscale = True
        height, width = int(first.shape[1]), int(first.shape[2])
    elif first.ndim == 4 and first.shape[-1] == 3:
        grayscale = False
        height, width = int(first.shape[1]), int(first.shape[2])
    else:
        raise ValueError(
            f"expected frames (n, H, W) or (n, H, W, 3), got {first.shape}"
        )
    writer = FFmpegVideoWriter(
        out_path,
        width,
        height,
        fps=fps,
        crf=crf,
        grayscale=grayscale,
    )
    try:
        for path in tqdm(
            chunk_paths,
            desc="video",
            ncols=88,
            disable=not show_progress,
        ):
            frames = load_frames_npz(path)
            for frame in frames:
                writer.write(frame)
    except Exception:
        writer.abort()
        raise
    writer.close()


def rebuild_gif_from_dir(
    frames_dir: str | Path,
    out_path: str | Path,
    *,
    duration_ms: int,
    quantize_colors: int,
    show_progress: bool = True,
) -> None:
    """Rebuild a GIF from a preserved `collision_frames_xxx` directory"""
    frames_dir = Path(frames_dir)
    if not frames_dir.is_dir():
        raise SystemExit(f"--rebuild-gif: not a directory: {frames_dir}")
    chunk_paths = sorted(frames_dir.glob("chunk_*.npz"), key=chunk_sort_key)
    if not chunk_paths:
        raise SystemExit(f"--rebuild-gif: no chunk_*.npz in {frames_dir}")
    log_info(
        f"         rebuilding GIF from {len(chunk_paths)} chunk(s) in {frames_dir}",
    )
    build_gif_from_chunks(
        chunk_paths,
        out_path,
        duration_ms=duration_ms,
        quantize_colors=quantize_colors,
        show_progress=show_progress,
    )


def rebuild_video_from_dir(
    frames_dir: str | Path,
    out_path: str | Path,
    *,
    fps: int,
    crf: int,
    show_progress: bool = True,
) -> None:
    """Rebuild an MP4 from a preserved `collision_frames_xxx` directory"""
    frames_dir = Path(frames_dir)
    if not frames_dir.is_dir():
        raise SystemExit(f"--rebuild-video: not a directory: {frames_dir}")
    chunk_paths = sorted(frames_dir.glob("chunk_*.npz"), key=chunk_sort_key)
    if not chunk_paths:
        raise SystemExit(f"--rebuild-video: no chunk_*.npz in {frames_dir}")
    log_info(
        f"         rebuilding video from {len(chunk_paths)} chunk(s) in {frames_dir}",
    )
    build_video_from_chunks(
        chunk_paths,
        out_path,
        fps=fps,
        crf=crf,
        show_progress=show_progress,
    )


# ==================================================================================== #
# CLI
# ==================================================================================== #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="REBOUND rubble-pile binary/bullet collision pipeline"
    )
    parser.add_argument("--overwrite-settled", action="store_true")
    parser.add_argument(
        "--log-dir",
        type=str,
        default=str(_cfg("run", "log_dir", "logs")),
        help="directory for the run log (default: logs)",
    )
    parser.add_argument(
        "--list-settled",
        action="store_true",
        help="list the settled-body cache library and exit",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress INFO console output and progress bars (DEBUG file log remains)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable progress bars only",
    )
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=FRAMES_CHUNK_SIZE)
    parser.add_argument(
        "--seed",
        type=int,
        default=RUN_SEED,
        help="run-level seed (random realisation choice); does not change "
        "the settled asteroid caches",
    )
    parser.add_argument(
        "--placement-seed",
        type=int,
        default=PLACEMENT_SEED,
        help="seed for asteroid construction; changing it creates a new "
        "settled body, changing --seed does not",
    )
    parser.add_argument("--no-parallel-settle", action="store_true")

    parser.add_argument("--param-path", type=str, default=None)
    parser.add_argument("--param-index", type=int, default=None)
    parser.add_argument("--look-back", type=float, default=100.0)
    parser.add_argument(
        "--start-at",
        choices=("contact", "lookback"),
        default="contact",
        help="contact = state_0 at t=0; lookback = selected -look_back snapshot",
    )

    parser.add_argument("--particles-per-body", type=int, default=None)
    parser.add_argument("--n1", type=int, default=N1)
    parser.add_argument("--n2", type=int, default=N2)
    parser.add_argument("--n3", type=int, default=N3)
    parser.add_argument("--ma", type=float, default=M1)
    parser.add_argument("--mb", type=float, default=M2)
    parser.add_argument("--mc", type=float, default=M3)
    parser.add_argument("--density", type=float, default=PARTICLE_DENSITY)
    parser.add_argument("--radius-ratio", type=float, default=None)

    parser.add_argument("--settle-time", type=float, default=SETTLE_TIME)
    parser.add_argument("--dt-settle", type=float, default=DT_SETTLE)
    parser.add_argument(
        "--settle-integrator",
        choices=("leapfrog", "ias15"),
        default=SETTLE_INTEGRATOR,
    )
    parser.add_argument("--dt-hit", type=float, default=DT_HIT)
    parser.add_argument(
        "--hit-integrator",
        choices=("leapfrog", "ias15"),
        default=HIT_INTEGRATOR,
    )
    parser.add_argument("--softening", type=float, default=SOFTENING_DEFAULT)
    parser.add_argument(
        "--bullet-speed",
        type=float,
        default=BULLET_SPEED,
        help="fixed-geometry bullet speed in m/s (ignored with --param-path)",
    )
    parser.add_argument(
        "--post-impact-tail",
        type=float,
        default=POST_IMPACT_TAIL,
        help="simulation time after impact in seconds",
    )
    parser.add_argument(
        "--steps-per-frame",
        type=int,
        default=STEPS_PER_FRAME_HIT,
        help="simulation steps between rendered frames",
    )
    parser.add_argument("--fig-size-in", type=float, default=FIG_SIZE_IN)
    parser.add_argument("--render-dpi", type=int, default=RENDER_DPI)

    parser.add_argument("--view-percentile", type=float, default=VIEW_PERCENTILE)
    parser.add_argument("--min-half-extent-km", type=float, default=MIN_FRAME_HALF_KM)
    parser.add_argument(
        "--max-half-extent-a-bin",
        type=float,
        default=MAX_HALF_EXTENT_A_BIN,
        help="half-extent ceiling in units of a_bin (full frame = 2x)",
    )
    parser.add_argument("--max-half-extent-km", type=float, default=None)

    parser.add_argument("--bound-radius-a-bin", type=float, default=BOUND_RADIUS_A_BIN)
    parser.add_argument("--bound-check-every", type=int, default=BOUND_CHECK_EVERY)
    parser.add_argument(
        "--stop-if-unbound",
        type=float,
        default=None,
        help="stop after impact if the bound energy mass fraction falls below this",
    )

    parser.add_argument(
        "--media-format",
        choices=("gif", "mp4", "both"),
        default=MEDIA_FORMAT,
        help="write a GIF, an MP4, or both",
    )
    parser.add_argument(
        "--color-mode",
        choices=("gray", "rgb"),
        default=COLOR_MODE,
        help="frame channels (default: gray)",
    )
    parser.add_argument("--video-fps", type=int, default=VIDEO_FPS)
    parser.add_argument("--video-crf", type=int, default=VIDEO_CRF)
    parser.add_argument("--gif-duration-ms", type=int, default=DURATION_MS)
    parser.add_argument("--gif-quantize-colors", type=int, default=GIF_QUANTIZE_COLORS)
    parser.add_argument("--rebuild-gif", type=str, default=None)
    parser.add_argument("--rebuild-gif-out", type=str, default=None)
    parser.add_argument("--rebuild-video", type=str, default=None)
    parser.add_argument("--rebuild-video-out", type=str, default=None)
    return parser.parse_args(argv)


# ==================================================================================== #
# Main
# ==================================================================================== #


def main(argv: list[str] | None = None) -> None:
    """Main cli entry point: settle, build, integrate, render, and encode the video"""
    args = parse_args(argv)
    _, log_path = setup_logging(
        "ringforge.main",
        log_dir=args.log_dir,
        console_level=logging.WARNING if args.quiet else logging.INFO,
    )
    show_progress = not (args.quiet or args.no_progress)
    log_debug(f"argv: {sys.argv}")
    DATA_DIR.mkdir(exist_ok=True)
    MEDIA_DIR.mkdir(exist_ok=True)

    if args.list_settled:
        list_settled_caches(DATA_DIR)
        return

    run_id = make_run_id(args.param_index)
    log_info(f"run_id={run_id}")
    if args.rebuild_gif is not None or args.rebuild_video is not None:
        if args.rebuild_gif is not None:
            gif_path = (
                Path(args.rebuild_gif_out)
                if args.rebuild_gif_out is not None
                else GIF_DIR / f"{GIF_STEM}_rebuild_{run_id}.gif"
            )
            rebuild_gif_from_dir(
                args.rebuild_gif,
                gif_path,
                duration_ms=args.gif_duration_ms,
                quantize_colors=args.gif_quantize_colors,
                show_progress=show_progress,
            )
        if args.rebuild_video is not None:
            video_path = (
                Path(args.rebuild_video_out)
                if args.rebuild_video_out is not None
                else VID_DIR / f"{VID_STEM}_rebuild_{run_id}.mp4"
            )
            rebuild_video_from_dir(
                args.rebuild_video,
                video_path,
                fps=args.video_fps,
                crf=args.video_crf,
                show_progress=show_progress,
            )
        return

    if not 0.0 < args.view_percentile <= 100.0:
        raise SystemExit("--view-percentile must be in (0, 100]")
    if args.chunk_size <= 0:
        raise SystemExit("--chunk-size must be positive")
    if args.dt_hit <= 0 or args.dt_settle <= 0 or args.settle_time <= 0:
        raise SystemExit("dt-hit, dt-settle, and settle-time must be positive")
    if args.bound_check_every <= 0:
        raise SystemExit("--bound-check-every must be positive")
    if args.min_half_extent_km < 0:
        raise SystemExit("--min-half-extent-km must be non-negative")
    if args.max_half_extent_a_bin <= 0:
        raise SystemExit("--max-half-extent-a-bin must be positive")
    if args.max_half_extent_km is not None and args.max_half_extent_km <= 0:
        raise SystemExit("--max-half-extent-km must be positive when set")
    if args.bound_radius_a_bin <= 0:
        raise SystemExit("--bound-radius-a-bin must be positive")
    if args.stop_if_unbound is not None and not (0.0 <= args.stop_if_unbound <= 1.0):
        raise SystemExit("--stop-if-unbound must be in [0, 1]")
    if args.softening < 0:
        raise SystemExit("--softening must be non-negative")
    if args.bullet_speed <= 0:
        raise SystemExit("--bullet-speed must be positive")
    if args.video_fps <= 0:
        raise SystemExit("--video-fps must be positive")
    if args.steps_per_frame <= 0:
        raise SystemExit("--steps-per-frame must be positive")
    if args.color_mode not in ("gray", "rgb"):
        raise SystemExit("--color-mode must be gray or rgb")
    if args.fig_size_in <= 0 or args.render_dpi <= 0:
        raise SystemExit("--fig-size-in and --render-dpi must be positive")
    if not 0 <= args.video_crf <= 51:
        raise SystemExit("--video-crf must be in [0, 51]")

    # resolve masses
    realisation: dict[str, Any] | None = None
    chosen_index: int | None = None
    if args.param_path is not None:
        realisation, chosen_index = param_utils.pick_realisation(
            args.param_path,
            seed=args.seed,
            index=args.param_index,
        )
        target_masses = (
            float(cast(float, realisation["ma"])),
            float(cast(float, realisation["mb"])),
            float(cast(float, realisation["mc"])),
        )
        log_info(
            f"Loaded realisation index {chosen_index} from {args.param_path}",
        )
        if not np.isclose(args.bullet_speed, BULLET_SPEED):
            log_warning(
                "         --bullet-speed is ignored with --param-path; "
                "the recorded scan v_rel is used",
            )
    else:
        target_masses = (float(args.ma), float(args.mb), float(args.mc))

    n_particles = (
        (int(args.particles_per_body),) * 3
        if args.particles_per_body is not None
        else (int(args.n1), int(args.n2), int(args.n3))
    )
    if any(n <= 0 for n in n_particles):
        raise SystemExit("particle counts must be positive")
    if any(mass <= 0 for mass in target_masses):
        raise SystemExit("masses must be positive")
    if args.density <= 0:
        raise SystemExit("--density must be positive")

    labels = ("host 1", "host 2", "bullet")
    seeds = (
        args.placement_seed,
        args.placement_seed + 1,
        args.placement_seed + 2,
    )

    log_info(
        f"\nstart_at={args.start_at}  run_seed={args.seed}  "
        f"placement_seed={args.placement_seed}  density={args.density}  "
        f"settle_integrator={args.settle_integrator}  "
        f"softening={args.softening:g} m  bullet_speed={args.bullet_speed:g} m/s  "
        f"media={args.media_format}",
    )
    log_info("Target bodies:")
    radius_ratios: list[float] = []
    for label, mass, n in zip(labels, target_masses, n_particles, strict=True):
        if args.radius_ratio is not None:
            ratio = float(args.radius_ratio)
        else:
            ratio = auto_radius_ratio(n)
        radius_ratios.append(ratio)
        describe_body(label, mass, n, ratio, density=args.density)

    # settle if none are saved already
    log_info("\nStage 1: build/settle/reuse three rubble piles")
    settled_paths: list[Path] = []
    jobs: list[dict[str, Any]] = []
    for label, mass, n, seed, ratio in zip(
        labels, target_masses, n_particles, seeds, radius_ratios, strict=True
    ):
        path = settled_path(
            mass,
            n,
            seed=seed,
            radius_ratio=ratio,
            density=args.density,
            settle_time=args.settle_time,
            dt=args.dt_settle,
            integrator=args.settle_integrator,
            softening=args.softening,
            root_size=SETTLE_ROOT_SIZE,
        )
        settled_paths.append(path)
        log_info(f"         {label}: M={mass:.6e} kg -> {path.name}")
        if path.exists() and not args.overwrite_settled:
            log_info(f"         [cache] found {path.name}; validating")
            expected = {
                "target_mass": mass,
                "target_n": n,
                "slope": SIZE_SLOPE,
                "density": args.density,
                "seed": seed,
            }
            try:
                load_settled_asteroid(path, expected=expected, density=args.density)
            except Exception as exc:
                log_info(
                    f"         [cache] {path.name} is not usable ({exc}); re-settling",
                )
            else:
                log_info(f"         [cache] reusing {path.name}")
                continue
        if args.overwrite_settled:
            log_info("         [cache] --overwrite-settled requested")
        log_info(
            f"         [cache] no reusable cache for this recipe; "
            f"will settle -> {path.name}",
        )
        jobs.append(
            {
                "target_mass": mass,
                "target_n": n,
                "settled_file": path,
                "seed": seed,
                "radius_ratio": ratio,
                "slope": SIZE_SLOPE,
                "density": args.density,
                "settle_time": args.settle_time,
                "dt": args.dt_settle,
                "integrator": args.settle_integrator,
                "softening": args.softening,
                "buffer": PARTICLE_BUFFER,
                "root_size": SETTLE_ROOT_SIZE,
            }
        )

    settle_bodies(
        jobs,
        parallel=not args.no_parallel_settle,
        log_path=log_path,
        show_progress=show_progress,
    )
    log_info("\n         settlement stage complete")

    states: list[np.ndarray] = []
    for label, mass, n, path in zip(
        labels, target_masses, n_particles, settled_paths, strict=True
    ):
        state, _extent, meta = load_settled_asteroid(
            path,
            expected={
                "target_mass": mass,
                "target_n": n,
                "slope": SIZE_SLOPE,
                "density": args.density,
            },
            density=args.density,
        )
        states.append(state)
        log_info(f"\n         Loaded {label}: {path.name}")
        r_lo = meta.get("r_lo")
        r_hi = meta.get("r_hi")
        check_powerlaw(
            state[:, 6],
            SIZE_SLOPE,
            float(r_lo) if r_lo is not None else float(state[:, 6].min()),
            float(r_hi) if r_hi is not None else float(state[:, 6].max()),
            label=label,
        )
        state_diagnostics(state, density=args.density, label=label)

    state1, state2, state3 = states

    # build collision case
    log_info("\nStage 2: build collision scene")
    t_impact: float
    approach_time: float
    if realisation is not None:
        if args.start_at == "contact":
            point_states = np.asarray(realisation["state_0"], float)
            approach_time = 0.0
            snapshot_index = -1
            actual_lookback = 0.0
            log_info("         start_at=contact (state_0 at t=0)")
        else:
            point_states, snapshot_index, actual_lookback = param_utils.select_state(
                realisation, args.look_back
            )
            approach_time = float(actual_lookback * float(realisation["T_bin"]))
            log_info(
                f"         start_at=lookback snapshot {snapshot_index} "
                f"(-{actual_lookback:g} T_bin), approach={approach_time:.3e} s",
            )
        sim, a_bin, t_bin, n_host = build_scene_realisation(
            realisation,
            (state1, state2, state3),
            point_states,
            target_masses,
            density=args.density,
            dt=args.dt_hit,
            integrator=args.hit_integrator,
            softening=args.softening,
        )
        t_impact = approach_time
        t_max = approach_time + float(args.post_impact_tail)
        max_half_m: float | None = args.max_half_extent_a_bin * a_bin
    else:
        sim, a_bin, t_bin, t_impact, n_host, separation, _a_extent = build_scene_fixed(
            state1,
            state2,
            state3,
            density=args.density,
            dt=args.dt_hit,
            integrator=args.hit_integrator,
            softening=args.softening,
            bullet_speed=args.bullet_speed,
        )
        approach_time = t_impact
        t_max = 2.0 * approach_time + float(args.post_impact_tail)
        max_half_m = (
            args.max_half_extent_km * 1000.0
            if args.max_half_extent_km is not None
            else separation
        )
        snapshot_index = -1
        actual_lookback = 0.0

    if args.max_half_extent_km is not None:
        max_half_m = args.max_half_extent_km * 1000.0
    min_half_m = args.min_half_extent_km * 1000.0
    if max_half_m is not None and min_half_m > max_half_m:
        log_info(
            f"         WARNING: min half-extent {min_half_m / 1000:.1f} km > "
            f"max {max_half_m / 1000:.1f} km; clamping the floor",
        )
        min_half_m = 0.5 * max_half_m

    log_info(f"         Particles total : {sim.N}")
    log_info(
        f"         Total mass      : {sum(p.m for p in sim.particles):.3e} kg",
    )
    log_info(f"         a_bin           : {a_bin:.6e} m")
    log_info(
        f"         T_bin           : {t_bin:.6e} s "
        f"({t_bin / 86400:.3f} d, {t_bin / 3600:.3f} h)",
    )
    log_info(f"         dt              : {sim.dt} s")
    log_info(f"         integrator      : {sim.integrator}")
    log_info(f"         softening       : {sim.softening} m")
    log_info(
        f"         Frame half-extent: floor={min_half_m / 1000:.1f} km  "
        f"ceiling="
        f"{max_half_m / 1000 if max_half_m is not None else float('inf'):.1f} km",
    )
    if max_half_m is not None:
        log_info(
            f"         Frame full extent: "
            f"{2.0 * min_half_m / 1000:.1f} - {2.0 * max_half_m / 1000:.1f} km",
        )
    log_info(
        f"         Top-level bound check radius: "
        f"{args.bound_radius_a_bin * a_bin / 1000:.1f} km",
    )

    if t_max <= 0:
        raise SystemExit("t_max is not positive; check the impact geometry")
    expected_frames = int(np.ceil(t_max / (args.steps_per_frame * args.dt_hit))) + 1
    log_info(
        f"         Approach time   : {approach_time:.3e} s "
        f"({approach_time / t_bin:.3f} T_bin)",
    )
    log_info(
        f"         Total simulated : {t_max:.3e} s "
        f"({t_max / t_bin:.3f} T_bin, {t_max / 86400:.2f} d)",
    )
    log_info(f"         Expected frames : ~{expected_frames}")
    if expected_frames > 2_000_000:
        log_info(
            "         WARNING: this is a very long render; consider "
            "--start-at contact or a larger --dt-hit",
        )

    run_meta = {
        **collect_metadata(),
        "run_id": run_id,
        "log_file": str(log_path),
        "args": vars(args),
        "target_masses": list(target_masses),
        "n_particles": list(n_particles),
        "placement_seeds": list(seeds),
        "run_seed": int(args.seed),
        "placement_seed": int(args.placement_seed),
        "radius_ratios": list(radius_ratios),
        "settled_paths": [str(path) for path in settled_paths],
        "a_bin": a_bin,
        "t_bin": t_bin,
        "t_impact": t_impact,
        "t_max": t_max,
        "n_host": n_host,
        "realisation_index": chosen_index,
        "snapshot_index": snapshot_index,
        "lookback": actual_lookback,
    }
    manifest_path = DATA_DIR / f"run_{run_id}.json"
    atomic_write_json(manifest_path, run_meta)

    # integrate and render
    log_info("\nStage 3: run sim and cache frames")
    gif_path = GIF_DIR / f"{GIF_STEM}_{run_id}.gif"
    video_path = VID_DIR / f"{VID_STEM}_{run_id}.mp4"
    save_gif = args.media_format in ("gif", "both")
    save_video = args.media_format in ("mp4", "both")
    grayscale = args.color_mode == "gray"
    stream_video = save_video and not save_gif and not args.keep_frames
    if stream_video:
        log_info(
            "         media mode: streaming MP4; no frame chunks are written",
        )
    chunks_dir: Path | None = None
    if not stream_video:
        chunks_dir = DATA_DIR / f"collision_frames_{run_id}"
        chunks_dir.mkdir(parents=True, exist_ok=True)

    fig, ax, canvas, circles = make_canvas(
        sim,
        fig_size_in=args.fig_size_in,
        dpi=args.render_dpi,
    )
    frames_buf: list[np.ndarray] = []
    chunk_paths: list[Path] = []
    chunk_idx = 0
    frame_idx = 0
    t0_wall = time.time()

    def flush_chunk() -> None:
        nonlocal frames_buf, chunk_idx
        if not frames_buf or chunks_dir is None:
            return
        path = chunks_dir / f"chunk_{chunk_idx:04d}.npz"
        save_frames_npz(frames_buf, path)
        chunk_paths.append(path)
        log_debug(
            f"         Chunk {chunk_idx:4d} "
            f"{path.stat().st_size / 1e6:6.2f} MB -> {path.name}",
        )
        frames_buf = []
        chunk_idx += 1

    # Frame 000 explicitly, with a blank-frame check.
    frame0 = render_frame_rgb(
        canvas,
        ax,
        circles,
        sim,
        sim.t,
        0,
        min_half=min_half_m,
        max_half=max_half_m,
        view_percentile=args.view_percentile,
        grayscale=grayscale,
    )
    non_white = int((frame0.mean(axis=-1) < 200).sum())
    frac = non_white / float(frame0.shape[0] * frame0.shape[1])
    log_debug(
        f"         frame 000: shape={frame0.shape} mean={frame0.mean():.1f} "
        f"non_white={frac:.4f}",
    )
    if frac == 0.0:
        log_warning("         frame 000 is blank")
    video_writer: FFmpegVideoWriter | None = None
    if stream_video:
        video_writer = FFmpegVideoWriter(
            video_path,
            int(frame0.shape[1]),
            int(frame0.shape[0]),
            fps=args.video_fps,
            crf=args.video_crf,
            grayscale=grayscale,
        )
        video_writer.write(frame0)
    else:
        frames_buf.append(frame0)
    frame_idx = 1
    if not stream_video and len(frames_buf) >= args.chunk_size:
        flush_chunk()

    rate: float | None = None
    eta_wall: float | None = None
    bound_history: list[dict[str, float]] = []
    stopped_early = False
    progress = tqdm(
        total=int(np.ceil(float(t_max))),
        unit="s",
        desc="simulating",
        ncols=88,
        disable=not show_progress,
    )
    last_t = float(sim.t)

    try:
        while sim.t < t_max - 1e-9:
            t_next = min(
                sim.t + args.steps_per_frame * args.dt_hit,
                t_max,
            )
            sim.integrate(t_next)
            align_root_size(sim)
            delta = max(0.0, float(sim.t) - last_t)
            progress.update(min(delta, float(progress.total) - progress.n))
            last_t = float(sim.t)

            if len(circles) != sim.N:
                plt.close(fig)
                fig, ax, canvas, circles = make_canvas(
                    sim,
                    fig_size_in=args.fig_size_in,
                    dpi=args.render_dpi,
                )

            rendered = render_frame_rgb(
                canvas,
                ax,
                circles,
                sim,
                sim.t,
                frame_idx,
                min_half=min_half_m,
                max_half=max_half_m,
                view_percentile=args.view_percentile,
                grayscale=grayscale,
            )
            if video_writer is not None:
                video_writer.write(rendered)
            else:
                frames_buf.append(rendered)
            frame_idx += 1
            if video_writer is None and len(frames_buf) >= args.chunk_size:
                flush_chunk()

            if sim.t >= t_impact and frame_idx % args.bound_check_every == 0:
                stats = bound_debris_stats(
                    sim,
                    n_host=n_host,
                    radius=args.bound_radius_a_bin * a_bin,
                    label=f"t={sim.t / t_bin:.2f} T_bin",
                )
                bound_history.append(
                    {
                        "t": float(sim.t),
                        "f_bound_radius": stats["f_bound_radius"],
                        "f_bound_energy": stats["f_bound_energy"],
                    }
                )
                if (
                    args.stop_if_unbound is not None
                    and sim.t > t_impact + 0.25 * (t_max - t_impact)
                    and stats["f_bound_energy"] < args.stop_if_unbound
                ):
                    log_warning(
                        f"         stopping early: bound energy fraction "
                        f"{stats['f_bound_energy']:.3f} < {args.stop_if_unbound:.3f}",
                    )
                    stopped_early = True
                    break

            if rate is None and frame_idx >= 100:
                elapsed = time.time() - t0_wall
                rate = elapsed / frame_idx
                remaining = rate * (expected_frames - frame_idx)
                eta_wall = time.time() + remaining
                log_debug(
                    f"         rate={rate:.3f} s/frame  elapsed={format_hms(elapsed)}  "
                    f"ETA {format_hms(remaining)} (done {format_clock(eta_wall)})",
                )
            if frame_idx % 100 == 0:
                eta_text = (
                    format_hms(max(eta_wall - time.time(), 0.0))
                    if eta_wall is not None
                    else "-"
                )
                log_debug(
                    f"         frame {frame_idx:5d}  t={format_hms(sim.t)}  "
                    f"wall={format_hms(time.time() - t0_wall)}  eta={eta_text}",
                )

    except BaseException:
        if video_writer is not None:
            video_writer.abort()
        plt.close(fig)
        raise

    if progress.n < progress.total:
        progress.update(float(progress.total) - progress.n)
    progress.close()
    flush_chunk()
    if video_writer is not None:
        video_writer.close()
    plt.close(fig)
    total_frames = frame_idx
    log_info(
        f"\n         Collected {total_frames} frames"
        + ("" if stream_video else f" over {len(chunk_paths)} chunk(s)")
    )
    body_diagnostics(sim, "t_max")
    final_bound = bound_debris_stats(
        sim,
        n_host=n_host,
        radius=args.bound_radius_a_bin * a_bin,
        label="final",
    )
    if final_bound["f_bound_energy"] < 0.05:
        log_warning(
            "         less than 5% of the debris mass is energetically "
            "bound to host 1; most of it escapes",
        )
    log_info(
        f"         bound debris: radial={final_bound['f_bound_radius']:.3f}  "
        f"energy={final_bound['f_bound_energy']:.3f}  "
        f"both={final_bound['f_bound_both']:.3f}",
    )

    if save_gif:
        if total_frames > 2000:
            log_warning(
                f"         GIF with {total_frames} frames is memory-heavy; "
                "consider --media-format mp4",
            )
        build_gif_from_chunks(
            chunk_paths,
            gif_path,
            duration_ms=args.gif_duration_ms,
            quantize_colors=args.gif_quantize_colors,
            show_progress=show_progress,
        )
    if save_video and not stream_video:
        build_video_from_chunks(
            chunk_paths,
            video_path,
            fps=args.video_fps,
            crf=args.video_crf,
            show_progress=show_progress,
        )

    run_meta.update(
        {
            "frames": total_frames,
            "chunks": [str(path) for path in chunk_paths],
            "media_format": args.media_format,
            "gif": str(gif_path) if save_gif else None,
            "video": str(video_path) if save_video else None,
            "streamed_video": stream_video,
            "stopped_early": stopped_early,
            "bound_final": final_bound,
            "bound_history": bound_history,
            "wall_time_s": time.time() - t0_wall,
        }
    )
    atomic_write_json(manifest_path, run_meta)

    if chunks_dir is not None and not args.keep_frames:
        log_info(
            f"         Removing chunk dir {chunks_dir.name} "
            f"(use --keep-frames to keep it)",
        )
        shutil.rmtree(chunks_dir, ignore_errors=True)
    elif chunks_dir is not None:
        log_info(
            f"         Kept chunk dir {chunks_dir.name}; rebuild with "
            f"--rebuild-gif or --rebuild-video {chunks_dir}",
        )


if __name__ == "__main__":
    main()

"""
Rubble pile asteroid ring forge playground. Work in progress.

Pipeline
    Stage 1 - init and settle asteroids -> cache to `npz` file
    Stage 2 - make 3 body collision scene from cached asteroids
    Stage 3 - run sim via nbody integration, save states in chunks to `npz`
    Stage 4 - make gif from cached `npz` files

Example CLI
    python test.py --overwrite-settled --keep-frames

"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import rebound

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image

# ==================================================================================== #
# Physical constants
# ==================================================================================== #

G: float = 6.67428e-11  # N m^2/kg^2
PARTICLE_DENSITY: float = 3000.0  # kg/m^3

# ==================================================================================== #
# Asteroid parameters
# ==================================================================================== #

N1: int = 2100
N2: int = 900
N3: int = 300

ROOT_SIZE_BASE: float = 1_000_000.0
R_MIN: float = 50.0
R_MAX: float = 1000.0
SIZE_SLOPE: float = -3.0
PARTICLE_BUFFER: float = 2.0

# ==================================================================================== #
# Spiral placement geometry
# ==================================================================================== #

SPIRAL_RADIUS: float = 6000.0
SPIRAL_Z_SPAN: float = 6000.0
SPIRAL_TURNS: float = 4.0
SPIRAL_JITTER: float = 50.0
SPIRAL_SEED: int = 42
SPIRAL_MAX_TRIES: int = 3000

# ==================================================================================== #
# Ball placement geometry
# ==================================================================================== #

BALL_RADIUS_REF: float = 3600.0  # place N_REF particles inside this radius
N_REF: int = 2100
PLACEMENT_METHOD: str = "ball"  # "ball" | "spiral"

# ==================================================================================== #
# Time control
# ==================================================================================== #

SETTLE_TIME: float = 108000.0  # s
DT_SETTLE: float = 10.0  # s
DT_HIT: float = 1.0  # s

STEPS_PER_FRAME: int = 10
DURATION_MS: int = 50  # ms per GIF frame

# ==================================================================================== #
# Impact geometry
# ==================================================================================== #

SEPARATION_FACTOR: float = 6.0
BULLET_SPEED: float = 100.0  # m/s
BOX_SIZE_FACTOR: float = 32.0
POST_IMPACT_TAIL: float = 6000.0

# Offsets are in units of separation
# bulk velocities are in units of BULLET_SPEED
# host 1
OFFSET_1: np.ndarray = np.asarray([+0.00, 0.25, 0.00])
BLKVEL_1: np.ndarray = np.asarray([-0.01, 0.00, 0.00])
# host 2
OFFSET_2: np.ndarray = np.asarray([+0.00, 0.00, 0.00])
BLKVEL_2: np.ndarray = np.asarray([+0.01, 0.00, 0.00])
# bullet
OFFSET_3: np.ndarray = np.asarray([-1.00, 0.00, 0.00])
BLKVEL_3: np.ndarray = np.asarray([+0.20, 0.00, 0.00])

# ==================================================================================== #
# I/O
# ==================================================================================== #

DATA_DIR: Path = Path("./data")
MEDIA_DIR: Path = Path("./media")
SETTLED_STEM: str = "settled"
GIF_STEM: str = "gif"

# ==================================================================================== #
# Rendering
# ==================================================================================== #

SHOW_FRAMES: bool = False
FRAMES_CHUNK_SIZE: int = 100
VIS_RADIUS_SCALE: float = 1.0
FIG_SIZE_IN: float = 8.0
DPI: int = 100
RENDER_DPI: int = 100
STEPS_PER_FRAME_HIT: int = 5

# ==================================================================================== #
# Helpers
# ==================================================================================== #


def format_hms(seconds: float) -> str:
    """
    Format a duration as ``HH:MM:SS``.
    """
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_clock(epoch_seconds: float) -> str:
    """
    Format an absolute epoch time as ``HH:MM:SS`` (local time).
    """
    return time.strftime("%H:%M:%S", time.localtime(epoch_seconds))


def powerlaw_size(
    rng: np.random.Generator,
    slope: float,
    r_min: float,
    r_max: float,
) -> float:
    """
    Sample a radius from :math:`p(r) \\propto r^{slope}` on [r_min, r_max].
    """
    u = rng.uniform()
    p_max = r_max ** (slope + 1.0)
    p_min = r_min ** (slope + 1.0)
    return float(((p_max - p_min) * u + p_min) ** (1.0 / (slope + 1.0)))


def mass_from_radius(r: float, density: float = PARTICLE_DENSITY) -> float:
    """
    Mass of a solid sphere of radius ``r`` at uniform ``density``.
    """
    return density * (4.0 / 3.0) * np.pi * r**3


def _fits(
    sim: rebound.Simulation,
    x: float,
    y: float,
    z: float,
    r_new: float,
    buffer: float = 0.0,
) -> bool:
    """
    True if ``(x, y, z)`` clears every existing particle by ``r_i + r_new + buffer``.
    """
    for p in sim.particles:
        dx = p.x - x
        dy = p.y - y
        dz = p.z - z
        if dx * dx + dy * dy + dz * dz < (p.r + r_new + buffer) ** 2:
            return False
    return True


def ball_radius_for(n: int) -> float:
    """Ball radius that keeps a fixed packing fraction across particle counts."""
    return BALL_RADIUS_REF * (n / N_REF) ** (1.0 / 3.0)


def check_powerlaw(
    radii: np.ndarray,
    slope_input: float,
    r_min: float,
    r_max: float,
    label: str = "",
) -> dict[str, Any] | None:
    """
    Compare an empirical radius sample to :math:`p(r) \\propto r^{slope}`.

    Fits both the PDF (expects ``slope_input``) and the survival function
    (expects ``slope_input + 1``).

    Returns ``None`` if the sample is too small to fit, otherwise a dict of diagnostics.
    """
    r = np.asarray(radii, dtype=float)
    n = r.size
    print()
    print(f"Stage 1: power-law check: {label}")
    print(f"         N sampled       : {n}")
    print(f"         r range (input) : [{r_min:7.1f}, {r_max:7.1f}] m")
    print(f"         r range (actual): [{r.min():7.1f}, {r.max():7.1f}] m")
    print(f"         input slope     : {slope_input:+.3f}")

    if n < 20:
        print("         WARNING: too few samples for a fit")
        return None

    if r.max() / r.min() < 3.0:
        print(f"         WARNING: dynamic range at {r.max() / r.min():.2f}x -> no fit")

    nbins = 15
    edges = np.logspace(np.log10(r.min() * 0.99), np.log10(r.max() * 1.01), nbins + 1)
    counts, _ = np.histogram(r, bins=edges)
    widths = np.diff(edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    pdf = counts / widths / n

    ok = counts > 0
    slope_pdf = np.nan
    if ok.sum() >= 2:
        log_r = np.log10(centers[ok])
        log_p = np.log10(pdf[ok])
        slope_pdf = np.polyfit(log_r, log_p, 1)[0]

    r_sorted = np.sort(r)
    survival = (n - np.arange(n)) / n
    lo, hi = int(0.1 * n), int(0.9 * n)
    log_r_s = np.log10(r_sorted[lo:hi])
    log_s = np.log10(survival[lo:hi])
    slope_sf = np.polyfit(log_r_s, log_s, 1)[0]

    print("         Power-law check results:")
    print(f"         fit  pdf slope: {slope_pdf:+.3f}(expected {slope_input:+.3f})")
    print(f"         fit  SF  slope: {slope_sf:+.3f} (expected {slope_input + 1:+.3f})")

    return {
        "label": label,
        "n": n,
        "slope_input": slope_input,
        "slope_pdf": slope_pdf,
        "slope_sf": slope_sf,
        "r_lo": r.min(),
        "r_hi": r.max(),
    }


# ==================================================================================== #
# Deterministic cache filenames
# ==================================================================================== #


def _fingerprint(*parts: object) -> str:
    """
    Short stable hash of a sequence of primitives.
    """
    key = "|".join(str(p) for p in parts)
    return hashlib.sha1(key.encode()).hexdigest()[:10]


def settled_path(
    n: int,
    *,
    seed: int = SPIRAL_SEED,
    slope: float = SIZE_SLOPE,
    r_min: float = R_MIN,
    r_max: float = R_MAX,
    settle_time: float = SETTLE_TIME,
    dt: float = DT_SETTLE,
    integrator: str = "leapfrog",
    density: float = PARTICLE_DENSITY,
    placement: str = PLACEMENT_METHOD,
    buffer: float = PARTICLE_BUFFER,
) -> Path:
    """
    Deterministic path for a settled asteroid with these exact params.
    """
    br = ball_radius_for(n) if placement == "ball" else SPIRAL_RADIUS
    h = _fingerprint(
        placement,
        seed,
        slope,
        r_min,
        r_max,
        settle_time,
        dt,
        integrator,
        density,
        br,
        buffer,
    )
    return DATA_DIR / f"{SETTLED_STEM}_N{n}_{h}.npz"


# ==================================================================================== #
# CLI
# ==================================================================================== #


def parse_args() -> argparse.Namespace:
    """
    Parse cli flags.
    """
    p = argparse.ArgumentParser()
    p.add_argument(
        "--overwrite-settled",
        action="store_true",
        help="Re-settle asteroids even if a matching saved state exists.",
    )
    p.add_argument(
        "--keep-frames",
        action="store_true",
        help="Keep the per-run chunk directory after the GIF is written.",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=FRAMES_CHUNK_SIZE,
        help=f"Frames per npz chunk (default {FRAMES_CHUNK_SIZE}).",
    )
    return p.parse_args()


# ==================================================================================== #
# Placement: ball
# ==================================================================================== #


def add_particles_ball_powerlaw(
    sim: rebound.Simulation,
    n: int,
    ball_radius: float | None = None,
    r_min: float = R_MIN,
    r_max: float = R_MAX,
    slope: float = SIZE_SLOPE,
    buffer: float = PARTICLE_BUFFER,
    max_tries: int = SPIRAL_MAX_TRIES,
    seed: int = SPIRAL_SEED,
) -> int:
    """
    Uniform-in-ball placement with power-law radii and overlap rejection.

    Returns the number of particles successfully placed
    (may be less than ``n`` if the volume is too crowded).
    """
    if ball_radius is None:
        ball_radius = ball_radius_for(n)

    rng = np.random.default_rng(seed)
    placed = 0
    for _ in range(n):
        radius = powerlaw_size(rng, slope, r_min, r_max)
        mass = mass_from_radius(radius)
        for _ in range(max_tries):
            u = rng.uniform()
            r = ball_radius * u ** (1.0 / 3.0)
            ct = rng.uniform(-1.0, 1.0)
            st = np.sqrt(1.0 - ct * ct)
            phi = rng.uniform(0.0, 2.0 * np.pi)
            x = r * st * np.cos(phi)
            y = r * st * np.sin(phi)
            z = r * ct
            if _fits(sim, x, y, z, radius, buffer):
                sim.add(m=mass, r=radius, x=x, y=y, z=z, vx=0.0, vy=0.0, vz=0.0)
                placed += 1
                break
    print(f"         ball: placed {placed}/{n} particles (R={ball_radius:.0f} m)")
    return placed


# ==================================================================================== #
# Placement: spiral
# ==================================================================================== #


def add_particles_spiral_powerlaw(
    sim: rebound.Simulation,
    n: int,
    r_spiral: float = SPIRAL_RADIUS,
    z_span: float = SPIRAL_Z_SPAN,
    turns: float = SPIRAL_TURNS,
    jitter: float = SPIRAL_JITTER,
    r_min: float = R_MIN,
    r_max: float = R_MAX,
    slope: float = SIZE_SLOPE,
    buffer: float = PARTICLE_BUFFER,
    max_tries: int = SPIRAL_MAX_TRIES,
    seed: int = SPIRAL_SEED,
) -> int:
    """
    Place particles on a 3D helix with power-law radii and rejection.

    Returns the number of particles successfully placed.
    """
    rng = np.random.default_rng(seed)
    placed = 0
    ts = np.linspace(0.0, 1.0, n)

    for t in ts:
        theta = 2.0 * np.pi * turns * t
        r_curve = r_spiral * t
        z_curve = z_span * (t - 0.5)
        radius = powerlaw_size(rng, slope, r_min, r_max)
        mass = mass_from_radius(radius)

        for _ in range(max_tries):
            x = r_curve * np.cos(theta) + rng.normal(0.0, jitter)
            y = r_curve * np.sin(theta) + rng.normal(0.0, jitter)
            z = z_curve + rng.normal(0.0, jitter)
            if _fits(sim, x, y, z, radius, buffer):
                sim.add(m=mass, r=radius, x=x, y=y, z=z, vx=0.0, vy=0.0, vz=0.0)
                placed += 1
                break

    print(f"         spiral: placed {placed}/{n} particles")
    return placed


def add_particles(
    sim: rebound.Simulation,
    n: int,
    method: str = PLACEMENT_METHOD,
) -> int:
    """
    Dispatch to the requested placement method; return placed count.
    """
    if method == "ball":
        return add_particles_ball_powerlaw(sim, n, ball_radius=ball_radius_for(n))
    return add_particles_spiral_powerlaw(sim, n)


# ==================================================================================== #
# Tree sim box management
# ==================================================================================== #


def adapt_root_size(
    sim: rebound.Simulation,
    safety: float = 1.5,
    minimum: float = 1e6,
) -> None:
    """
    Grow ``sim.root_size`` so the tree box encloses every particle.

    Called between ``integrate`` calls.  Cheap enough to call every step.
    """
    max_r = max(max(abs(p.x), abs(p.y), abs(p.z)) for p in sim.particles)
    v_max = max(np.sqrt(p.vx**2 + p.vy**2 + p.vz**2) for p in sim.particles)
    step_margin = v_max * sim.dt
    needed = 2.0 * (max_r + step_margin) * safety
    if sim.root_size < needed:
        sim.root_size = max(needed, minimum)
        print(
            f"         [tree] root_size -> {sim.root_size:,.0f} m  "
            f"(max_r = {max_r:,.0f} m, v_max = {v_max:.0f} m/s)"
        )


# ==================================================================================== #
# Simulation factory
# ==================================================================================== #


def coefficient_of_restitution_constant(r: float, v: float) -> float:
    """
    Constant coefficient of restitution for hard-sphere collisions.
    """
    return 0.5


def make_fresh_simulation(
    dt: float = 60.0,
    softening: float = 0.5,
    root_size: float = ROOT_SIZE_BASE,
) -> rebound.Simulation:
    """
    Create a `REBOUND` sim preconfigured for tree gravity + collisions.
    """
    sim = rebound.Simulation()
    sim.G = G
    sim.softening = softening
    sim.dt = dt
    sim.integrator = "ias15"

    sim.gravity = "tree"
    sim.root_size = root_size
    sim.opening_angle2 = 0.5

    sim.collision = "tree"
    sim.coefficient_of_restitution = coefficient_of_restitution_constant
    sim.collision_resolve = "hardsphere"

    return sim


# ==================================================================================== #
# Stage 1: settle one asteroid
# ==================================================================================== #


def process_asteroid(
    n: int,
    settled_file: Path | str,
    root_size: float = ROOT_SIZE_BASE,
    dt: float = DT_SETTLE,
    integrator: str = "ias15",
    settle_time: float = SETTLE_TIME,
) -> None:
    """
    Place, settle, and record a single asteroid state to ``settled_file``.
    """
    settled_file = Path(settled_file)
    print(f"\n          Process Asteroid - {settled_file.name}")

    sim_settle = make_fresh_simulation(dt, root_size)
    sim_settle.integrator = integrator

    print("         Adding particles to settle")
    add_particles(sim_settle, n)
    sim_settle.move_to_com()

    xs = np.array([p.x for p in sim_settle.particles])
    ys = np.array([p.y for p in sim_settle.particles])
    zs = np.array([p.z for p in sim_settle.particles])
    rs = np.array([p.r for p in sim_settle.particles])
    ms = np.array([p.m for p in sim_settle.particles])

    print(
        f"         Initial extent (x,y,z) : "
        f"{np.ptp(xs):.0f} * {np.ptp(ys):.0f} * {np.ptp(zs):.0f}  m"
    )
    print(f"         Radii range            : [{rs.min():.0f}, {rs.max():.0f}]  m")
    print(f"         Total mass             : {ms.sum():.3e}  kg")

    print(f"\n          Integrating for {settle_time:.0f} s ...")
    sim_settle.integrate(settle_time)

    v_rms_pre = float(
        np.sqrt(np.mean([p.vx**2 + p.vy**2 + p.vz**2 for p in sim_settle.particles]))
    )
    print(f"         [diag] t={sim_settle.t:.0f} s")
    print(f"         integrator={sim_settle.integrator}")
    print(f"         gravity={sim_settle.gravity}")
    print(f"         root_size={sim_settle.root_size:.0f}")
    print(f"         pre-COM v_rms={v_rms_pre:.4f} m/s")

    sim_settle.move_to_com()

    xs = np.array([p.x for p in sim_settle.particles])
    ys = np.array([p.y for p in sim_settle.particles])
    zs = np.array([p.z for p in sim_settle.particles])
    vxs = np.array([p.vx for p in sim_settle.particles])
    vys = np.array([p.vy for p in sim_settle.particles])
    vzs = np.array([p.vz for p in sim_settle.particles])

    extent = max(np.ptp(xs), np.ptp(ys), np.ptp(zs))
    v_rms = float(np.sqrt(np.mean(vxs**2 + vys**2 + vzs**2)))
    m_tot = ms.sum()
    v_esc = np.sqrt(2.0 * G * m_tot / (0.5 * extent))

    print("\n          After settling:")
    print(
        f"         Settled extent         : {extent:.0f} m  (~{extent / 1000:.1f} km)"
    )
    print(f"         RMS internal velocity  : {v_rms:.3f} m/s")
    print(f"         Escape velocity        : {v_esc:.3f} m/s")
    print(f"         Bound?                 : {'YES' if v_rms < v_esc else 'NO'}")

    state = np.zeros((sim_settle.N, 7))
    for i, p in enumerate(sim_settle.particles):
        state[i] = [p.x, p.y, p.z, p.vx, p.vy, p.vz, p.r]

    np.savez(settled_file, state=state, density=PARTICLE_DENSITY)
    print(f"\n          Recorded state -> {settled_file}")


# ==================================================================================== #
# Stage 2: build the collision scene
# ==================================================================================== #


def load_settled_asteroid(settled_file: Path | str) -> tuple[np.ndarray, float]:
    """
    Load a recorded state and return ``(state, max_extent)``.
    """
    data = np.load(settled_file)
    state = data["state"]
    a_extent = max(np.ptp(state[:, 0]), np.ptp(state[:, 1]), np.ptp(state[:, 2]))
    return state, a_extent


def get_scene_params(
    a_extent1: float,
    a_extent2: float,
    a_extent3: float,
    separation_factor: float = SEPARATION_FACTOR,
    box_size_factor: float = BOX_SIZE_FACTOR,
) -> tuple[float, float, float]:
    """
    Return ``(mean_extent, separation, box_size)`` for the collision scene.
    """
    a_extent = (a_extent1 + a_extent2 + a_extent3) / 3
    separation = separation_factor * a_extent
    box_size_hit = box_size_factor * separation
    return a_extent, separation, box_size_hit


def add_settled_asteroid(
    sim: rebound.Simulation,
    state: np.ndarray,
    offset: np.ndarray,
    bulk_vel: np.ndarray,
) -> None:
    """
    Place a recorded asteroid with the given offset and bulk velocity.

    ``state`` is an ``(N, 7)`` array of ``[x, y, z, vx, vy, vz, r]``.
    """
    offset = np.asarray(offset, dtype=float)
    bulk_vel = np.asarray(bulk_vel, dtype=float)

    pos = state[:, 0:3] + offset  # (N, 3)
    vel = state[:, 3:6] + bulk_vel  # (N, 3)
    rad = state[:, 6]  # (N,)
    mass = PARTICLE_DENSITY * (4.0 / 3.0) * np.pi * rad**3

    for i in range(state.shape[0]):
        sim.add(
            m=mass[i],
            r=rad[i],
            x=pos[i, 0],
            y=pos[i, 1],
            z=pos[i, 2],
            vx=vel[i, 0],
            vy=vel[i, 1],
            vz=vel[i, 2],
        )


def build_collision_case(
    state1: np.ndarray,
    state2: np.ndarray,
    state3: np.ndarray,
    separation: float,
    box_size_hit: float,
    bullet_speed: float = BULLET_SPEED,
    dt: float = DT_HIT,
    integrator: str = "ias15",
) -> rebound.Simulation:
    """
    Return a fresh simulation containing the three-body collision scene.
    """
    sim_hit = make_fresh_simulation(dt, box_size_hit)
    sim_hit.integrator = integrator

    # S3 N3 - bullet on the left, moving to the right
    add_settled_asteroid(
        sim_hit,
        state3,
        offset=OFFSET_3 * separation,
        bulk_vel=BLKVEL_3 * bullet_speed,
    )
    # S1 N1 - target, offset upward
    add_settled_asteroid(
        sim_hit,
        state1,
        offset=OFFSET_1 * separation,
        bulk_vel=BLKVEL_1 * bullet_speed,
    )
    # S2 N2 - second target, near centre
    add_settled_asteroid(
        sim_hit,
        state2,
        offset=OFFSET_2 * separation,
        bulk_vel=BLKVEL_2 * bullet_speed,
    )
    return sim_hit


# ==================================================================================== #
# Stage 3: rendering
# ==================================================================================== #


def compute_half_extent(
    sim: rebound.Simulation,
    pad_factor: float = 1.10,
    min_half: float = 1.0,
) -> float:
    """
    Adaptive half-extent of the xy viewport, from current particle positions.
    """
    if sim.N == 0:
        return min_half
    xs = np.fromiter((p.x for p in sim.particles), dtype=float, count=sim.N)
    ys = np.fromiter((p.y for p in sim.particles), dtype=float, count=sim.N)
    r_max = max(np.abs(xs).max(), np.abs(ys).max())
    return float(max(r_max * pad_factor, min_half))


def make_canvas(
    sim: rebound.Simulation,
    vis_radius_scale: float = VIS_RADIUS_SCALE,
    dpi: int = RENDER_DPI,
) -> tuple[Figure, Axes, FigureCanvasAgg, list[patches.Circle]]:
    """
    Build a figure, axes, and one ``Circle`` per particle -- once.

    Returns ``(fig, ax, canvas, circles)``.  Subsequent frames should reuse
    these via :func:`render_frame_rgb` rather than rebuilding.
    """
    fig, ax = plt.subplots(figsize=(FIG_SIZE_IN, FIG_SIZE_IN), dpi=dpi)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    # placeholder; the first render_frame_rgb call fixes these
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)

    circles = [
        patches.Circle(
            (p.x, p.y),
            radius=p.r * vis_radius_scale,
            facecolor="darkgray",
            edgecolor="black",
            linewidth=0.4,
            alpha=0.85,
        )
        for p in sim.particles
    ]
    for c in circles:
        ax.add_patch(c)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    return fig, ax, canvas, circles


def render_frame_rgb(
    canvas: FigureCanvasAgg,
    ax: Axes,
    circles: list[patches.Circle],
    sim: rebound.Simulation,
    t: float,
    idx: int,
    pad_factor: float = 1.10,
    min_half: float = 1.0,
) -> np.ndarray:
    """
    Update artists and view limits, rasterise, return a ``(H, W, 3)`` frame.
    """
    for c, p in zip(circles, sim.particles, strict=False):
        c.center = (p.x, p.y)

    half_extent = compute_half_extent(sim, pad_factor, min_half)
    ax.set_xlim(-half_extent, half_extent)
    ax.set_ylim(-half_extent, half_extent)

    ax.set_title(
        f"t = {t:7.1f} s  ({t / 60:5.2f} min)  |  "
        f"±{half_extent / 1000:6.1f} km  |  frame {idx:03d}"
    )
    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba())
    return buf[..., :3].copy()


def close_canvas(fig: Figure) -> None:
    """
    Release the matplotlib figure.
    """
    plt.close(fig)


# ==================================================================================== #
# Stage 4: frames -> npz, npz -> GIF
# ==================================================================================== #


def save_frames_npz(frames: list[np.ndarray], path: Path | str) -> tuple[int, ...]:
    """
    Persist a list of ``(H, W, 3)`` uint8 arrays to a single ``.npz``.

    Returns the resulting array shape ``(N, H, W, 3)``.
    """
    arr = np.stack(frames, axis=0)
    np.savez(path, frames=arr)
    return arr.shape


def load_frames_npz(path: Path | str) -> np.ndarray:
    """
    Read back an array written by :func:`save_frames_npz`.
    """
    with np.load(path) as data:
        return data["frames"]


def build_gif_from_chunks(
    chunk_paths: list[Path],
    out_path: Path | str,
    duration_ms: int = DURATION_MS,
) -> None:
    """
    Build a GIF by reading chunk ``.npz`` files one at a time.

    Each frame is converted to paletted P-mode (1 byte per pixel) before
    being appended to the PIL frame list.
    """
    if not chunk_paths:
        print("\n          No chunks to build gif.\n")
        return

    pil_frames: list[Image.Image] = []
    total = 0
    print()
    for i, p in enumerate(chunk_paths):
        arr = load_frames_npz(p)
        for f in arr:
            img = Image.fromarray(f).quantize(colors=128)
            pil_frames.append(img)
            total += 1
        print(f"         GIF assembly: chunk {i + 1}/{len(chunk_paths)} {total} frames")

    pil_frames[0].save(
        str(out_path),
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    print(f"\n         Saved GIF: {out_path}  ({total} frames)")


# ==================================================================================== #
# Main
# ==================================================================================== #


def main() -> None:
    """
    Entry point: settle -> collide -> render -> encode GIF.
    """
    args = parse_args()
    DATA_DIR.mkdir(exist_ok=True)
    MEDIA_DIR.mkdir(exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    frames_npz = DATA_DIR / f"collision_frames_{run_id}.npz"
    gif_path = MEDIA_DIR / f"{GIF_STEM}_{run_id}.gif"

    # Stage 1
    print("Stage 1: build and settle 3 asteroids")

    planet_defs: list[tuple[int, str]] = [
        (N1, "host 1"),
        (N2, "host 2"),
        (N3, "bullet"),
    ]
    settled_paths: list[Path] = []

    for n, label in planet_defs:
        p = settled_path(n)
        settled_paths.append(p)
        print()
        print(f"         {label}: N={n} -> {p.name}")

        if p.exists() and not args.overwrite_settled:
            print(f"         [skip] reusing {p}")
            continue
        if p.exists():
            print(f"         [overwrite] rebuilding {p}")

        process_asteroid(n, p)
        print()

    print("         Asteroids process complete")

    s1, s2, s3 = settled_paths
    state1, extent1 = load_settled_asteroid(s1)
    state2, extent2 = load_settled_asteroid(s2)
    state3, extent3 = load_settled_asteroid(s3)

    check_powerlaw(state1[:, 6], SIZE_SLOPE, R_MIN, R_MAX, label="host 1")
    check_powerlaw(state2[:, 6], SIZE_SLOPE, R_MIN, R_MAX, label="host 2")
    check_powerlaw(state3[:, 6], SIZE_SLOPE, R_MIN, R_MAX, label="bullet")

    print()
    print("         Settled asteroids state and extent parsed")
    print()

    # Stage 2
    print("Stage 2: build collision scene")
    a_extent, separation, box_size_hit = get_scene_params(extent1, extent2, extent3)

    print(f"         Single extent   : {a_extent:.0f} m")
    print(f"         COM separation  : {separation:.0f} m")
    print(f"         Rel bullet speed: {BULLET_SPEED} m/s")
    print(f"         Simulation box  : {box_size_hit:.0f} m")

    sim_hit = build_collision_case(state1, state2, state3, separation, box_size_hit)

    print(f"         Particles total : {sim_hit.N}")
    print(f"         Total mass      : {sum(p.m for p in sim_hit.particles):.3e} kg")
    print(f"         dt              : {sim_hit.dt} s")
    print()

    # Stage 3
    print("Stage 3: run sim")
    approach_time = (separation - a_extent) / BULLET_SPEED
    t_max_hit = 2.0 * approach_time + POST_IMPACT_TAIL
    expected_frames = int(np.ceil(t_max_hit / (STEPS_PER_FRAME_HIT * DT_HIT))) + 1

    fig, ax, canvas, circles = make_canvas(sim_hit)
    canva_w, canva_h = canvas.get_width_height()

    print(f"         Predicted approach time : {approach_time:.0f} s")
    print(f"         Total simulated time    : {t_max_hit:.0f} s")
    print(f"         dt                      : {DT_HIT} s")
    print(f"         Frame every             : {STEPS_PER_FRAME_HIT * DT_HIT:.0f} s")
    print(f"         Expected frames         : ~{expected_frames}")
    print(f"         Frame size              : {canva_w}*{canva_h} px {RENDER_DPI}dpi")
    print(f"         Frames npz              : {frames_npz}")
    print(f"         GIF output              : {gif_path}")
    print()

    # per-run chunk directory
    chunks_dir = DATA_DIR / f"collision_frames_{run_id}"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    frames_buf: list[np.ndarray] = []
    chunk_paths: list[Path] = []
    chunk_idx = 0
    frame_idx = 0
    _t0 = time.time()

    def _flush_chunk(
        buf: list[np.ndarray],
        idx: int,
        paths: list[Path],
    ) -> None:
        """
        Write the current frame buffer as a numbered chunk and clear it.
        """
        p = chunks_dir / f"chunk_{idx:04d}.npz"
        save_frames_npz(buf, p)
        paths.append(p)
        size_mb = p.stat().st_size / 1e6
        print(f"         Chunk {idx:4d} {size_mb:6.2f} MB -> {p.name}")
        buf.clear()

    while True:
        frames_buf.append(
            render_frame_rgb(canvas, ax, circles, sim_hit, sim_hit.t, frame_idx)
        )

        if sim_hit.t >= t_max_hit - 1e-6:
            break

        t_next = min(sim_hit.t + STEPS_PER_FRAME_HIT * DT_HIT, t_max_hit)
        sim_hit.integrate(t_next)

        t_eta_wall: float = -1.0

        if frame_idx == 100:
            elapsed = time.time() - _t0
            rate = elapsed / frame_idx
            t_remaining = rate * (expected_frames - frame_idx)
            t_eta_wall = time.time() + t_remaining
            print(f"         Rate         : {rate:6.3f} s/frame")
            print(f"         Elapsed      : {format_hms(elapsed)}")
            print(
                f"         ETA (finish) : {format_hms(t_remaining)} remaining"
                f"         done at {format_clock(t_eta_wall)}"
            )

        if frame_idx % 100 == 0:
            print(
                f"         Frame {frame_idx:4d}  t = {format_hms(sim_hit.t)}"
                f"wall = {format_hms(time.time() - _t0)}"
                f"eta  = {format_hms(t_eta_wall - time.time())}"
            )

        frame_idx += 1

        if len(frames_buf) >= args.chunk_size:
            _flush_chunk(frames_buf, chunk_idx, chunk_paths)
            chunk_idx += 1

    # flush the trailing partial chunk
    if frames_buf:
        _flush_chunk(frames_buf, chunk_idx, chunk_paths)
        chunk_idx += 1

    close_canvas(fig)

    # Stage 4
    total_frames = frame_idx + 1
    print(
        f"\n         Collected {total_frames} frames over {len(chunk_paths)} chunk(s)\n"
    )

    build_gif_from_chunks(chunk_paths, str(gif_path), DURATION_MS)

    if not args.keep_frames:
        shutil.rmtree(chunks_dir, ignore_errors=True)
        print(f"         Removed chunk dir: {chunks_dir}")


if __name__ == "__main__":
    main()

"""
Iniitial condition utils

    - scan: samples initial conditions at contact, at `lookback * T_bin` snapshots
    - plot: renders results distributions and KS diagnostics
    - read: parse one realisation to main if needed

Note
    - Always in SI units
    - States are `(N, 7)` arrays

Contact geometry is sampled either uniformly (design scan) or with flux
weighting `p(mu) = 2 mu`, where `mu = -dot(v_dir, offset_dir)`.  The
two modes are written to every scan so they can be compared directly.

CLI
    python param_utils.py scan \
        --n 10000 --out data/param_scan_test/ --workers 16 --chunk 2000 --overwrite
    python param_utils.py plot data/param_scan_test \
        --out media/param_scan_test.png \
        --diagnostics media/param_scan_test_diag.png \
        --ks-tests 
    python param_utils.py read data/param_scan_test --index 42

CLI - test case
    python param_utils.py scan --test
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import multiprocessing as mp
import sys
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import config as ring_config
import matplotlib
import numpy as np
import rebound
from lumberyard import setup_logging
from scipy.constants import G as _G
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==================================================================================== #
# Constants and types
# ==================================================================================== #

G_SI: float = float(_G)
"""Grav const from scipy"""

# cfg build
_CFG = ring_config.load_config()


def _cfg[T](section: str, key: str, default: T) -> T:
    """Read one value from config.toml with a built-in fallback"""
    return cast(T, ring_config.value(_CFG, section, key, default))


# logger
LOGGER = logging.getLogger("ringforge.param_utils")


def log_info(message: str) -> None:
    """Logger - info"""
    LOGGER.info("%s", message)


def log_debug(message: str) -> None:
    """Logger - debug"""
    LOGGER.debug("%s", message)


# version
VERSION: str = "param_utils 0.0.1"

# const parse
M1_DEFAULT: float = float(_cfg("bodies", "ma", 1.0e15))  # kg
M2_DEFAULT: float = float(_cfg("bodies", "mb", 1.0e13))  # kg
M3_DEFAULT: float = float(_cfg("bodies", "mc", 1.0e12))  # kg
RHO_DEFAULT: float = float(_cfg("bodies", "density", 3000.0))  # kg/m3
A_BIN_DEFAULT: float = float(_cfg("binary", "a_bin", 1.0e4))  # m
E_BIN_DEFAULT: float = float(_cfg("binary", "e_bin", 0.0))
INC_BIN_DEFAULT: float = float(_cfg("binary", "inc_bin", 0.0))  # rad
V_REL_DEFAULT: float = float(_cfg("binary", "v_rel", 100.0))  # m/s
_LOOKBACKS = _cfg("scan", "lookbacks", [0.01, 0.1, 1.0, 10.0])
LOOKBACK_DEFAULT: tuple[float, ...] = tuple(float(x) for x in _LOOKBACKS)

# types
type ScalarValue = float | int | bool
type ParamValue = ScalarValue | np.ndarray
type SampleReturn = ParamValue | dict[str, object]
type RealisationDict = dict[str, object]

# Keys written for every successful realisation.
# `mu` and `b_contact` are derived from the sampled geometry.
SAMPLED_SCHEMA: list[tuple[str, tuple[int, ...]]] = [
    ("ma", ()),  # kg
    ("mb", ()),  # kg
    ("mc", ()),  # kg
    ("rho", ()),  # kg/m3
    ("a_bin", ()),  # m
    ("e_bin", ()),
    ("inc_bin", ()),  # rad
    ("v_rel", ()),  # m/s at contact
    ("binary_phase", ()),  # rad
    ("offset_dir", (3,)),  # unit vector, mb -> mc at contact
    ("v_dir", (3,)),  # unit vector, v_mc - v_mb at contact
]
DERIVED_SCHEMA: list[tuple[str, tuple[int, ...]]] = [
    ("mu", ()),  # -dot(v_dir, offset_dir), in (0, 1]
    ("b_contact", ()),  # m, geometric impact parameter at contact
]
PARAM_SCHEMA: list[tuple[str, tuple[int, ...]]] = SAMPLED_SCHEMA + DERIVED_SCHEMA

# ==================================================================================== #
# Formatting
# ==================================================================================== #


def fmt_num(value: object) -> str:
    """Format a number compactly for terminal output"""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        v = float(value)
        if v == 0.0:
            return "0.0"
        if 1e-3 <= abs(v) < 1e3:
            return f"{v:g}"
        return f"{v:.3e}"
    if isinstance(value, np.ndarray):
        return "[" + ", ".join(fmt_num(x) for x in value.tolist()) + "]"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(fmt_num(x) for x in value) + "]"
    return str(value)


def _utc_now() -> str:
    """Get current UTC time"""
    return datetime.now(UTC).isoformat()


def _git_hash() -> str:
    """Read the current git hash"""
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


# ==================================================================================== #
# Sampler classes
# ==================================================================================== #


class Param:
    """Base param sampler cls"""

    def sample(self, rng: np.random.Generator, index: int) -> SampleReturn:
        raise NotImplementedError

    def describe(self) -> str:
        return self.__class__.__name__


class Fixed(Param):
    """Param sampler cls with const val return"""

    def __init__(self, value: ParamValue) -> None:
        self.value: ParamValue = value

    def sample(self, rng: np.random.Generator, index: int) -> ParamValue:
        value = self.value
        if isinstance(value, np.ndarray):
            return value.copy()
        return value

    def describe(self) -> str:
        return f"Fixed({fmt_num(self.value)})"


class Uniform(Param):
    """Uniform scalar on `[lo, hi]`"""

    def __init__(self, lo: float, hi: float) -> None:
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            raise ValueError(f"Uniform requires finite lo < hi, got ({lo}, {hi})")
        self.lo = float(lo)
        self.hi = float(hi)

    def sample(self, rng: np.random.Generator, index: int) -> float:
        return float(rng.uniform(self.lo, self.hi))

    def describe(self) -> str:
        return f"Uniform({fmt_num(self.lo)}, {fmt_num(self.hi)})"


class ImpactGeometry(Param):
    """
    Sample bullet contact and approach directions at `t = 0`
    """

    def __init__(self, mode: str = "uniform", max_tries: int = 1_000_000) -> None:
        if mode not in ("uniform", "flux"):
            raise ValueError(f"unknown geometry mode: {mode!r}")
        self.mode = mode
        self.max_tries = int(max_tries)

    def sample(self, rng: np.random.Generator, index: int) -> dict[str, object]:
        off = rng.normal(size=3)
        off /= np.linalg.norm(off)

        for _ in range(self.max_tries):
            v = rng.normal(size=3)
            v /= np.linalg.norm(v)
            mu = float(-np.dot(v, off))
            if mu <= 0.0:
                continue
            if self.mode == "flux" and rng.uniform() >= mu:
                continue
            return {"offset_dir": off, "v_dir": v, "mu": mu}

        raise RuntimeError(
            f"ImpactGeometry({self.mode}) failed to draw a valid pair "
            f"in {self.max_tries} tries"
        )

    def describe(self) -> str:
        if self.mode == "flux":
            return "ImpactGeometry(flux-weighted, p(mu)=2mu)"
        return "ImpactGeometry(uniform contact + inward v_rel)"


# ==================================================================================== #
# Parameter space
# ==================================================================================== #


@dataclass(frozen=True)
class ScanConfig:
    """Fixed parameters of one scan"""

    ma: float = M1_DEFAULT
    mb: float = M2_DEFAULT
    mc: float = M3_DEFAULT
    rho: float = RHO_DEFAULT
    a_bin: float = A_BIN_DEFAULT
    e_bin: float = E_BIN_DEFAULT
    inc_bin: float = INC_BIN_DEFAULT
    v_rel: float = V_REL_DEFAULT
    geometry: str = str(_cfg("scan", "geometry", "uniform"))
    phase: str = str(_cfg("scan", "phase", "random"))
    lookback_periods: tuple[float, ...] = LOOKBACK_DEFAULT
    master_seed: int = int(_cfg("run", "seed", 42))


def make_params(config: ScanConfig) -> dict[str, Param]:
    """Build the ordered sampler dictionary for `config`"""
    if config.phase == "random":
        phase_sampler: Param = Uniform(0.0, 2.0 * np.pi)
    elif config.phase == "fixed":
        phase_sampler = Fixed(0.0)
    else:
        raise ValueError(f"unknown phase mode: {config.phase!r}")

    return {
        "ma": Fixed(config.ma),
        "mb": Fixed(config.mb),
        "mc": Fixed(config.mc),
        "rho": Fixed(config.rho),
        "a_bin": Fixed(config.a_bin),
        "e_bin": Fixed(config.e_bin),
        "inc_bin": Fixed(config.inc_bin),
        "v_rel": Fixed(config.v_rel),
        "binary_phase": phase_sampler,
        "geometry": ImpactGeometry(config.geometry),
    }


def describe_space(params: dict[str, Param], master_seed: int) -> None:
    """Print the param space and the master seed"""
    log_info("=" * 88)
    log_info(f"{VERSION}   master_seed = {master_seed}")
    for name, param in params.items():
        log_info(f"  {name:14s}  {param.describe()}")
    log_info("=" * 88)


def sample(
    index: int,
    master_seed: int = 42,
    params: dict[str, Param] | None = None,
) -> RealisationDict:
    """
    Build one realisation as a pure function of `index, master_seed, params`

    A dict valued sampler (`ImpactGeometry`) merges its sub-keys into the result
    The result always contains `index`
    """
    if params is None:
        params = make_params(ScanConfig(master_seed=master_seed))

    seed_seq = np.random.SeedSequence([int(master_seed), int(index)])
    children = seed_seq.spawn(len(params))

    out: RealisationDict = {}
    for (name, param), child in zip(params.items(), children, strict=False):
        rng = np.random.default_rng(child)
        value: SampleReturn = param.sample(rng, int(index))
        if isinstance(value, dict):
            out.update(value)
        else:
            out[name] = value
    out["index"] = int(index)
    return out


# ==================================================================================== #
# One realisation: build, place, integrate backwards
# ==================================================================================== #


def sphere_radius(mass: float, rho: float) -> float:
    """Radius of a uniform sphere of `mass` at density `rho` (m)"""
    if mass <= 0 or rho <= 0:
        raise ValueError("mass and rho must be positive")
    return float((3.0 * mass / (4.0 * np.pi * rho)) ** (1.0 / 3.0))


def _extract(sim: rebound.Simulation) -> np.ndarray:
    """Return the state of every particle as an `(N, 6)` array"""
    return np.array([[p.x, p.y, p.z, p.vx, p.vy, p.vz] for p in sim.particles])


def run_one(
    real: RealisationDict,
    lookback_periods: tuple[float, ...] = LOOKBACK_DEFAULT,
) -> RealisationDict:
    """
    Build one three-body encounter in SI units and integrate it backwards
    """
    ma = cast(float, real["ma"])
    mb = cast(float, real["mb"])
    mc = cast(float, real["mc"])
    rho = cast(float, real["rho"])
    a_bin = cast(float, real["a_bin"])
    e_bin = cast(float, real["e_bin"])
    inc_bin = cast(float, real["inc_bin"])
    v_rel = cast(float, real["v_rel"])
    phase = cast(float, real.get("binary_phase", 0.0))
    off = cast(np.ndarray, real["offset_dir"])
    v_dir = cast(np.ndarray, real["v_dir"])

    for name, value in (
        ("ma", ma),
        ("mb", mb),
        ("mc", mc),
        ("rho", rho),
        ("a_bin", a_bin),
        ("v_rel", v_rel),
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive, got {value!r}")

    r_a = sphere_radius(ma, rho)
    r_b = sphere_radius(mb, rho)
    r_c = sphere_radius(mc, rho)
    contact = r_b + r_c

    sim = rebound.Simulation()
    sim.units = ("s", "m", "kg")
    sim.add(m=ma)
    sim.add(m=mb, a=a_bin, e=e_bin, inc=inc_bin, l=phase)
    sim.move_to_com()

    # Internal binary period (SI).
    t_bin = float(2.0 * np.pi * np.sqrt(a_bin**3 / (sim.G * (ma + mb))))

    # Bullet at contact with mb at t = 0.
    pb = sim.particles[1]
    sim.add(
        m=mc,
        x=pb.x + contact * off[0],
        y=pb.y + contact * off[1],
        z=pb.z + contact * off[2],
        vx=pb.vx + v_rel * v_dir[0],
        vy=pb.vy + v_rel * v_dir[1],
        vz=pb.vz + v_rel * v_dir[2],
    )

    state_0 = _extract(sim)

    # Diagnostics in the rest frame of mb.
    v_mb = state_0[1, 3:6]
    v_mb_hat = v_mb / (np.linalg.norm(v_mb) + 1e-30)
    # cos_contact =
    #   +1 means mc sits on the side mb is moving towards the leading edge
    #   -1 is the trailing edge
    cos_contact = float(np.dot(off, v_mb_hat))
    # cos_vrel =
    #   +1 means the bullet moves with mb (rear-end)
    #   -1 is head-on.
    cos_vrel = float(np.dot(v_dir, v_mb_hat))
    mu = float(-np.dot(off, v_dir))
    b_contact = float(contact * np.sqrt(max(0.0, 1.0 - mu * mu)))

    # Backward integration
    # IAS15 is time-reversible and its adaptive step handles the close contact
    # sim.dt is inert for IAS15
    sim.integrator = "ias15"
    sim.dt = t_bin / 200.0

    nl = len(lookback_periods)
    snapshots = np.zeros((nl, 3, 6))
    for i, lookback in enumerate(lookback_periods):
        sim.integrate(-float(lookback) * t_bin)
        snapshots[i] = _extract(sim)

    params_out: dict[str, object] = {
        name: real[name] for name, _ in SAMPLED_SCHEMA if name in real
    }
    params_out["mu"] = mu
    params_out["b_contact"] = b_contact

    return {
        "index": real["index"],
        "success": True,
        "params": params_out,
        "state_0": state_0,
        "snapshots": snapshots,
        "T_bin": t_bin,
        "R_a": r_a,
        "R_b": r_b,
        "R_c": r_c,
        "contact": contact,
        "cos_contact": cos_contact,
        "cos_vrel": cos_vrel,
        "mu": mu,
        "b_contact": b_contact,
    }


def worker(
    real: RealisationDict,
    lookback_periods: tuple[float, ...],
) -> RealisationDict:
    """Pool-safe wrapper for `run_one`"""
    try:
        return run_one(real, lookback_periods)
    except Exception as exc:
        return {"index": real["index"], "success": False, "error": repr(exc)}


# ==================================================================================== #
# Chunk I/O
# ==================================================================================== #


def chunk_sort_key(path: Path) -> int:
    """Sort `chunk_xxx.npz` numerically"""
    try:
        return int(Path(path).stem.split("_")[-1])
    except (ValueError, IndexError):
        return 10**18


def save_chunk(
    results: list[RealisationDict],
    out_dir: str | Path,
    chunk_id: int,
    meta: dict[str, Any],
    *,
    overwrite: bool = False,
) -> Path:
    """Write one chunk"""
    n = len(results)
    nl = len(meta["lookback_periods"])

    out: dict[str, np.ndarray] = {}
    for name, shape in PARAM_SCHEMA:
        out[name] = np.full((n, *shape), np.nan) if shape else np.full(n, np.nan)
    out["state_0"] = np.full((n, 3, 6), np.nan)
    out["snapshots"] = np.full((n, nl, 3, 6), np.nan)
    out["index"] = np.zeros(n, dtype=np.int64)
    out["success"] = np.zeros(n, dtype=bool)
    out["T_bin"] = np.full(n, np.nan)
    out["R_a"] = np.full(n, np.nan)
    out["R_b"] = np.full(n, np.nan)
    out["R_c"] = np.full(n, np.nan)
    out["contact"] = np.full(n, np.nan)
    out["cos_contact"] = np.full(n, np.nan)
    out["cos_vrel"] = np.full(n, np.nan)
    out["lookback_periods"] = np.asarray(meta["lookback_periods"], dtype=float)

    for i, result in enumerate(results):
        out["index"][i] = cast(int, result["index"])
        if not bool(result.get("success", False)):
            continue
        out["success"][i] = True
        params = cast(dict[str, object], result["params"])
        for name, _ in PARAM_SCHEMA:
            if name in params:
                out[name][i] = cast(ParamValue, params[name])
        out["state_0"][i] = cast(np.ndarray, result["state_0"])
        out["snapshots"][i] = cast(np.ndarray, result["snapshots"])
        out["T_bin"][i] = cast(float, result["T_bin"])
        out["R_a"][i] = cast(float, result["R_a"])
        out["R_b"][i] = cast(float, result["R_b"])
        out["R_c"][i] = cast(float, result["R_c"])
        out["contact"][i] = cast(float, result["contact"])
        out["cos_contact"][i] = cast(float, result["cos_contact"])
        out["cos_vrel"][i] = cast(float, result["cos_vrel"])
        out["mu"][i] = cast(float, result["mu"])
        out["b_contact"][i] = cast(float, result["b_contact"])

    out["meta_json"] = np.array(json.dumps(meta, sort_keys=True))
    out["schema_json"] = np.array(
        json.dumps([(name, list(shape)) for name, shape in PARAM_SCHEMA])
    )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    path = out_path / f"chunk_{chunk_id:06d}.npz"
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists; use --overwrite or choose a new --start"
        )

    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, allow_pickle=False, **out)
    tmp.replace(path)
    return path


def load_chunks(directory: str | Path) -> tuple[dict[str, np.ndarray], int]:
    """
    Join every `chunk_xxx.npz` in `directory` along axis 0

    `lookback_periods` and the JSON metadata are scalars
    Returns `(data, n_files)`
    """
    files = sorted(Path(directory).glob("chunk_*.npz"), key=chunk_sort_key)
    if not files:
        raise FileNotFoundError(f"no chunk_*.npz files in {directory}")

    scalar_keys = {"lookback_periods", "meta_json", "schema_json"}
    scalars: dict[str, np.ndarray] = {}
    chunks: dict[str, list[np.ndarray]] = {}

    for path in files:
        with np.load(path, allow_pickle=False) as data:
            for key in data.files:
                array = data[key]
                if key in scalar_keys:
                    scalars.setdefault(key, array.copy())
                else:
                    chunks.setdefault(key, []).append(array.copy())

    out = {key: np.concatenate(value, axis=0) for key, value in chunks.items()}
    out.update(scalars)
    return out, len(files)


def list_indices(directory: str | Path) -> np.ndarray:
    """Return the sorted global indices of successful realisations"""
    files = sorted(Path(directory).glob("chunk_*.npz"), key=chunk_sort_key)
    if not files:
        raise FileNotFoundError(f"no chunk_*.npz files in {directory}")
    collected: list[np.ndarray] = []
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            if "index" not in data or "success" not in data:
                continue
            success = np.asarray(data["success"], bool)
            collected.append(np.asarray(data["index"])[success])
    if not collected:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(collected))


def load_realisation(
    directory: str | Path, index: int
) -> dict[str, np.ndarray | float | int | dict[str, Any]]:
    """
    Load one successful realisation by its global `index`

    API for main sim
    Returns the recorded `state_0`, `snapshots`, `lookback_periods`, masses, `T_bin`,
        `a_bin`, and the stored metadata
    """
    files = sorted(Path(directory).glob("chunk_*.npz"), key=chunk_sort_key)
    if not files:
        raise FileNotFoundError(f"no chunk_*.npz files in {directory}")

    for path in files:
        with np.load(path, allow_pickle=False) as data:
            if "index" not in data:
                continue
            rows = np.flatnonzero(np.asarray(data["index"]) == int(index))
            if not len(rows):
                continue
            row = int(rows[0])
            if not bool(np.asarray(data["success"])[row]):
                raise ValueError(f"realisation {index} failed in {path}")

            result: dict[str, np.ndarray | float | int | dict[str, Any]] = {}
            for key in data.files:
                if key == "lookback_periods":
                    result[key] = data[key].copy()
                elif key in ("meta_json", "schema_json"):
                    if key == "meta_json":
                        result["meta"] = json.loads(data[key].item())
                elif data[key].ndim > 0 and data[key].shape[0] == len(data["index"]):
                    result[key] = data[key][row].copy()
            return result

    raise KeyError(f"successful realisation index {index} not found in {directory}")


def pick_realisation(
    directory: str | Path,
    seed: int | None = None,
    index: int | None = None,
) -> tuple[dict[str, np.ndarray | float | int | dict[str, Any]], int]:
    """Load `index` if given, otherwise choose one successful index at random"""
    if index is not None:
        return load_realisation(directory, int(index)), int(index)
    indices = list_indices(directory)
    if not len(indices):
        raise RuntimeError(f"no successful realisations in {directory}")
    rng = np.random.default_rng(42 if seed is None else int(seed))
    chosen = int(rng.choice(indices))
    return load_realisation(directory, chosen), chosen


def select_state(
    realisation: dict[str, Any],
    lookback: float,
) -> tuple[np.ndarray, int, float]:
    """
    Return `point_states, snapshot_index, actual_lookback` for the snapshot nearest to
        `lookback` (units of `T_bin`)
    """
    lookbacks = np.asarray(realisation["lookback_periods"], dtype=float)
    if not lookbacks.size:
        raise ValueError("realisation has no lookback_periods")
    index = int(np.argmin(np.abs(lookbacks - float(lookback))))
    states = np.asarray(realisation["snapshots"], dtype=float)
    if states.shape[1:] != (3, 6):
        raise ValueError(f"Expected snapshots (n, 3, 6), got {states.shape}")
    return states[index], index, float(lookbacks[index])


# ==================================================================================== #
# Scan driver
# ==================================================================================== #


def run_batch(
    n: int,
    out_dir: str | Path,
    config: ScanConfig,
    *,
    workers: int | None = None,
    chunk: int = 2000,
    start: int = 0,
    overwrite: bool = False,
    ks_tests: bool = False,
    show_progress: bool = True,
) -> None:
    """Run `n` realisations, writing atomic chunks and an optional KS output"""
    if n <= 0:
        raise ValueError("n must be positive")
    if chunk <= 0:
        raise ValueError("chunk must be positive")
    if workers is None:
        workers = mp.cpu_count()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    params = make_params(config)
    describe_space(params, config.master_seed)
    log_info(
        f"Running {n} realisations  (start={start}, workers={workers}, "
        f"chunk={chunk}, geometry={config.geometry}, phase={config.phase})"
    )
    log_info("")

    meta: dict[str, Any] = {
        "version": VERSION,
        "git_hash": _git_hash(),
        "created_utc": _utc_now(),
        "config": asdict(config),
        "master_seed": int(config.master_seed),
        "lookback_periods": [float(x) for x in config.lookback_periods],
        "rebound": rebound.__version__,
        "numpy": np.__version__,
        "python": sys.version.split()[0],
    }

    def gen() -> Iterator[RealisationDict]:
        for index in range(int(start), int(start) + int(n)):
            yield sample(index, config.master_seed, params)

    chunk_id = int(start) // int(chunk)
    buffer: list[RealisationDict] = []
    t0 = time.time()
    produced = 0

    pool_worker = functools.partial(worker, lookback_periods=config.lookback_periods)

    with mp.Pool(workers) as pool:
        iterator = pool.imap_unordered(pool_worker, gen(), chunksize=4)
        progress = tqdm(
            total=n,
            unit="case",
            ncols=88,
            desc="simulating",
            smoothing=0.05,
            disable=not show_progress,
        )
        try:
            for result in iterator:
                buffer.append(result)
                produced += 1
                progress.update(1)
                elapsed = time.time() - t0
                rate = produced / elapsed if elapsed > 0 else 0.0
                progress.set_postfix(rate=f"{rate:.1f}/s", refresh=False)
                if len(buffer) >= chunk:
                    save_chunk(buffer, out_dir, chunk_id, meta, overwrite=overwrite)
                    chunk_id += 1
                    buffer = []
        finally:
            progress.close()

    if buffer:
        save_chunk(buffer, out_dir, chunk_id, meta, overwrite=overwrite)

    if ks_tests:
        data, _ = load_chunks(out_dir)
        run_sampling_tests(data)

    log_info(
        f"\nDone. {produced} realisations in {time.time() - t0:.1f} s -> {out_dir}/"
    )


# ==================================================================================== #
# Plotting and statistics
# ==================================================================================== #


def choose_unit(max_meters: float) -> tuple[float, str]:
    """Choose a length unit with readable axis values"""
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


def parse_cos_vrel(spec: str | None) -> tuple[float | None, float | None] | None:
    """Parse `'0.5:'`, `':-0.5'`, or `'0.5:1.0'`"""
    if spec is None:
        return None
    if ":" not in spec:
        raise ValueError("--cos-vrel must contain ':' e.g. '0.5:' or ':-0.5'")
    lo_s, hi_s = spec.split(":", 1)
    return (float(lo_s) if lo_s else None, float(hi_s) if hi_s else None)


def apply_cos_vrel_filter(
    data: dict[str, np.ndarray],
    spec: tuple[float | None, float | None] | None,
) -> tuple[dict[str, np.ndarray], str]:
    """Keep only successful realisations inside the given `cos_vrel` range"""
    if spec is None:
        return data, "all"
    lo, hi = spec
    cos_vrel = np.asarray(data["cos_vrel"])
    mask = np.asarray(data["success"], bool).copy()
    if lo is not None:
        mask &= cos_vrel >= lo
    if hi is not None:
        mask &= cos_vrel <= hi
    lo_text = lo if lo is not None else "-inf"
    hi_text = hi if hi is not None else "+inf"
    label = f"cos_vrel in ({lo_text}, {hi_text})"
    out = {
        key: (
            value
            if (
                key in ("lookback_periods", "meta_json", "schema_json")
                or value.ndim == 0
            )
            else value[mask]
        )
        for key, value in data.items()
    }
    return out, label


def plot_origins(
    data: dict[str, np.ndarray],
    out_png: str | Path,
    *,
    label: str = "all",
    n_scatter: int = 5000,
    seed: int = 0,
) -> None:
    """Scatter the bullet position relative to `ma` at each lookback"""
    ok = np.asarray(data["success"], bool)
    snapshots = np.asarray(data["snapshots"])[ok]
    if not snapshots.size:
        raise RuntimeError("no successful realisations to plot")

    v_rel = np.asarray(data["v_rel"])[ok]
    lookbacks = np.asarray(data["lookback_periods"], float)
    n_ok = snapshots.shape[0]

    rel_pos = snapshots[:, :, 2, :3] - snapshots[:, :, 0, :3]
    sep = np.linalg.norm(rel_pos, axis=-1)
    scale, unit = choose_unit(float(np.nanmax(sep)))

    rng = np.random.default_rng(seed)
    selected = rng.choice(n_ok, size=min(n_scatter, n_ok), replace=False)
    classes = np.unique(v_rel)
    colors = {c: f"C{i}" for i, c in enumerate(classes)}

    n_lb = len(lookbacks)
    fig, axes = plt.subplots(2, n_lb, figsize=(4 * n_lb, 8), squeeze=False)

    for i, lookback in enumerate(lookbacks):
        ax = axes[0, i]
        xy = rel_pos[selected, i, :2] / scale
        for value in classes:
            mask = v_rel[selected] == value
            if not mask.any():
                continue
            ax.scatter(
                xy[mask, 0],
                xy[mask, 1],
                s=1,
                alpha=0.2,
                c=colors[value],
                label=f"{fmt_num(value)} m/s" if i == 0 else None,
            )
        ax.plot(0, 0, marker="P", color="k", ms=12, alpha=0.8, label="ma")
        ax.set_aspect("equal")
        ax.set_title(f"Bullet position at t = -{int(lookback)} T_bin")
        ax.set_xlabel(f"x_mc - x_ma  [{unit}]")
        ax.set_ylabel(f"y_mc - y_ma  [{unit}]")
        if i == 0:
            ax.legend(loc="lower left", fontsize=8, frameon=True)

    for i, lookback in enumerate(lookbacks):
        ax = axes[1, i]
        radius = sep[:, i] / scale
        positive = radius[radius > 0]
        if positive.size:
            lo = max(float(positive.min()), 1e-3 * float(radius.max()))
            hi = float(radius.max())
        else:
            lo, hi = 1e-3, 1.0
        if hi <= lo:
            hi = lo * 10.0
        bins = np.logspace(np.log10(lo), np.log10(hi), 100)
        for value in classes:
            mask = v_rel == value
            if mask.any():
                ax.hist(
                    radius[mask],
                    bins=bins,
                    alpha=0.6,
                    color=colors[value],
                    label=f"{fmt_num(value)} m/s" if i == 0 else None,
                )
        ax.set_title(f"Separation at t = -{int(lookback)} T_bin")
        ax.set_xlabel(f"|x_mc - x_ma|  [{unit}]")
        ax.set_ylabel("Count")
        if i == 0:
            ax.legend(loc="lower left", fontsize=8, frameon=True)

    fig.suptitle(f"param_utils scan origins [{label}]")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log_info(f"Saved {out_png}  ({n_ok} cases, axes in {unit})")


def _mu(data: dict[str, np.ndarray]) -> np.ndarray:
    """Return `mu` from the data, computing it from the raw vectors if absent"""
    if "mu" in data:
        return np.asarray(data["mu"], float)
    off = np.asarray(data["offset_dir"], float)
    v_dir = np.asarray(data["v_dir"], float)
    return -np.einsum("ij,ij->i", off, v_dir)


def prograde_mask(data: dict[str, np.ndarray], lookback_index: int = -1) -> np.ndarray:
    """
    Boolean mask: bullet's angular momentum about the binary COM is +z

    The binary plane is the x-y plane (inc_bin = 0)
    Using the COM rather than ma or mb avoids the origin-dependent drift caused by
        the binary's internal angular momentum
    """
    snapshots = np.asarray(data["snapshots"], float)
    ma = np.asarray(data["ma"], float)
    mb = np.asarray(data["mb"], float)
    mc = np.asarray(data["mc"], float)
    masses = np.stack([ma, mb, mc], axis=1)  # (n, 3)
    total = masses.sum(axis=1)

    states = snapshots[:, lookback_index, :, :]
    positions = np.einsum("nij,nj->ni", states[..., :3], masses) / total[:, None]
    velocities = np.einsum("nij,nj->ni", states[..., 3:6], masses) / total[:, None]
    rel_r = states[:, 2, :3] - positions
    rel_v = states[:, 2, 3:6] - velocities
    h_z = np.cross(rel_r, rel_v)[:, 2]
    return h_z > 0.0


def plot_diagnostics(
    data: dict[str, np.ndarray],
    out_png: str | Path,
    *,
    label: str = "all",
) -> None:
    """Histograms of speed, impact geometry, contact point, and mu"""
    ok = np.asarray(data["success"], bool)
    if not ok.any():
        raise RuntimeError("no successful realisations")

    v_rel = np.asarray(data["v_rel"])[ok]
    cos_vrel = np.asarray(data["cos_vrel"])[ok]
    cos_contact = np.asarray(data["cos_contact"])[ok]
    off = np.asarray(data["offset_dir"])[ok]
    mu = _mu(data)[ok]

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))

    ax = axes[0, 0]
    classes = np.unique(v_rel)
    if len(classes) > 1:
        ax.hist(v_rel, bins=30, alpha=0.8)
    else:
        ax.axvline(float(classes[0]), color="C0", lw=2)
    ax.set_xlabel("v_rel [m/s]")
    ax.set_ylabel("Count")
    ax.set_title("Bullet speed at contact")

    ax = axes[0, 1]
    ax.hist(cos_vrel, bins=100, alpha=0.8)
    ax.axvline(0.5, color="r", ls="--", lw=1, label="rear-end")
    ax.axvline(-0.5, color="r", ls=":", lw=1, label="head-on")
    ax.set_xlabel("cos(v_dir, v_mb)")
    ax.set_ylabel("Count")
    ax.set_title("Approach direction vs mb velocity")
    ax.legend(loc="lower left", fontsize=8)

    ax = axes[0, 2]
    ax.hist(cos_contact, bins=100, alpha=0.8)
    ax.set_xlabel("cos(offset_dir, v_mb)")
    ax.set_ylabel("Count")
    ax.set_title("Contact point: +1 leading, -1 trailing")

    ax = axes[1, 0]
    ax.scatter(off[:, 0], off[:, 1], s=2, alpha=0.3, c=mu, cmap="viridis")
    ax.set_aspect("equal")
    ax.set_xlabel("offset_dir_x")
    ax.set_ylabel("offset_dir_y")
    ax.set_title("Contact direction (colour = mu)")

    ax = axes[1, 1]
    ax.hist(mu, bins=100, density=True, alpha=0.8, label="sample")
    grid = np.linspace(0.0, 1.0, 200)
    ax.plot(grid, np.ones_like(grid), "k--", lw=1, label="uniform: 1")
    ax.plot(grid, 2.0 * grid, "r-", lw=1, label="flux: 2 mu")
    ax.set_xlabel("mu = -dot(v_dir, offset_dir)")
    ax.set_ylabel("Probability density")
    ax.set_title("Incidence cosine")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[1, 2]
    if "b_contact" in data:
        b = np.asarray(data["b_contact"])[ok]
        ax.hist(b, bins=100, alpha=0.8)
        ax.set_xlabel("b_contact [m]")
        ax.set_ylabel("Count")
        ax.set_title("Geometric impact parameter")
    else:
        ax.axis("off")

    fig.suptitle(f"param_utils diagnostics [{label}]")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log_info(f"Saved {out_png}")


def ks_uniform_report(values: np.ndarray, label: str, lo: float, hi: float) -> None:
    """KS test against `Uniform(lo, hi)`, ignoring non-finite values"""
    from scipy.stats import kstest

    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if not x.size:
        log_info(f"            {label}: no finite samples")
        return
    if hi <= lo:
        raise ValueError("uniform upper bound must exceed lower bound")
    result = kstest(x, "uniform", args=(lo, hi - lo))
    log_info(
        f"            {label}: n={x.size}, D={result.statistic:.5g}, "
        f"p={result.pvalue:.5g} (Uniform[{lo:g}, {hi:g}])"
    )


def run_sampling_tests(
    data: dict[str, np.ndarray],
    reference: dict[str, np.ndarray] | None = None,
) -> None:
    """
    KS diagnostics for the sampled geometry.

    Single sample tests check the marginal uniformities that the sampler should produce

    `mu`:
        a uniform design scan should follow `Uniform(0, 1)`
        a flux-weighted scan should follow `Beta(2, 1)`, whose density is 2*mu

    A two-sample test compares the filtered sample (if a `--cos-vrel` cut was applied)
        with the full sample, and prograde with retrograde subsets about the binary COM.
    """
    from scipy.stats import ks_2samp, kstest

    ok = np.asarray(data["success"], bool)
    if not ok.any():
        log_info("KS diagnostics: no successful samples")
        return

    log_info("Sampling goodness-of-fit (KS tests; interpret p-values with care):")
    mu = _mu(data)[ok]
    if reference is None:
        for name in ("offset_dir", "v_dir"):
            if name not in data:
                continue
            vectors = np.asarray(data[name])[ok]
            for axis_index, axis in enumerate("xyz"):
                ks_uniform_report(vectors[:, axis_index], f"{name}.{axis}", -1.0, 1.0)

        for name in ("cos_contact", "cos_vrel"):
            if name in data:
                ks_uniform_report(np.asarray(data[name])[ok], name, -1.0, 1.0)

        if mu.size:
            result = kstest(mu[np.isfinite(mu)], "uniform", args=(0.0, 1.0))
            log_info(
                f"            mu vs Uniform(0,1): n={mu.size}, "
                f"D={result.statistic:.5g}, p={result.pvalue:.5g}"
            )
            result = kstest(mu[np.isfinite(mu)], "beta", args=(2.0, 1.0))
            log_info(
                f"            mu vs Beta(2,1)   : n={mu.size}, "
                f"D={result.statistic:.5g}, p={result.pvalue:.5g}"
            )
    else:
        log_info(
            "            filtered sample: one-sample uniform tests skipped "
            "(conditioning changes the marginals); two-sample test below"
        )

    if reference is not None:
        ref_mu = _mu(reference)[np.asarray(reference["success"], bool)]
        sample_mu = mu
        if sample_mu.size and ref_mu.size:
            result = ks_2samp(sample_mu, ref_mu)
            log_info(
                f"            mu filtered vs full: n={sample_mu.size}/"
                f"{ref_mu.size}, D={result.statistic:.5g}, "  # type: ignore
                f"p={result.pvalue:.5g}"  # type: ignore
            )

    try:
        prograde = prograde_mask(data)
        if prograde.size == mu.size and prograde.any() and (~prograde).any():
            result = ks_2samp(mu[prograde], mu[~prograde])
            log_info(
                f"            mu prograde vs retrograde: "
                f"n={int(prograde.sum())}/{int((~prograde).sum())}, "
                f"D={result.statistic:.5g}, p={result.pvalue:.5g} "  # type: ignore
                f"(fraction prograde={prograde.mean():.4f})"
            )
    except (KeyError, ValueError) as exc:
        log_info(f"            prograde/retrograde test skipped: {exc!r}")


# ==================================================================================== #
# CLI
# ==================================================================================== #


def _build_scan_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Build cli flags - scan"""
    parser = subparsers.add_parser("scan", help="run the backward-integration scan")
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="number of realisations (default for --test: 10000)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="output directory (default for --test: data/param_scan_test/)",
    )
    parser.add_argument("--seed", type=int, default=int(_cfg("run", "seed", 42)))
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="pool size (default for --test: from config [scan].workers)",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=int(_cfg("scan", "chunk", 2000)),
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--test",
        action="store_true",
        help="run the standard 10k test scan and then the standard plots+KS",
    )
    parser.add_argument("--plot-out", type=str, default=None)
    parser.add_argument("--plot-diagnostics", type=str, default=None)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress INFO console output",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=str(_cfg("run", "log_dir", "logs")),
    )
    parser.add_argument(
        "--geometry",
        choices=("uniform", "flux"),
        default=str(_cfg("scan", "geometry", "uniform")),
    )
    parser.add_argument(
        "--phase",
        choices=("random", "fixed"),
        default=str(_cfg("scan", "phase", "random")),
    )
    parser.add_argument("--ma", type=float, default=M1_DEFAULT)
    parser.add_argument("--mb", type=float, default=M2_DEFAULT)
    parser.add_argument("--mc", type=float, default=M3_DEFAULT)
    parser.add_argument("--rho", type=float, default=RHO_DEFAULT)
    parser.add_argument("--a-bin", type=float, default=A_BIN_DEFAULT)
    parser.add_argument("--e-bin", type=float, default=E_BIN_DEFAULT)
    parser.add_argument("--inc-bin", type=float, default=INC_BIN_DEFAULT)
    parser.add_argument("--v-rel", type=float, default=V_REL_DEFAULT)
    parser.add_argument(
        "--lookback",
        type=float,
        nargs="+",
        default=list(LOOKBACK_DEFAULT),
        help="lookback times in units of T_bin",
    )
    parser.add_argument("--ks-tests", action="store_true")


def _build_plot_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Build cli flags - plot"""
    parser = subparsers.add_parser("plot", help="plot a scan and run KS tests")
    parser.add_argument("dir", type=str)
    parser.add_argument("--out", type=str, default="media/param_scan_test.png")
    parser.add_argument("--diagnostics", type=str, default=None)
    parser.add_argument("--ks-tests", action="store_true")
    parser.add_argument("--cos-vrel", type=str, default=None)
    parser.add_argument("--n-scatter", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress INFO console output",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=str(_cfg("run", "log_dir", "logs")),
    )


def _build_read_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Build cli flags - read"""
    parser = subparsers.add_parser("read", help="load one realisation")
    parser.add_argument("dir", type=str)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lookback", type=float, default=None)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress INFO console output",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=str(_cfg("run", "log_dir", "logs")),
    )


def summarise_scan(
    directory: str | Path,
    out_png: str | Path,
    diagnostics_png: str | Path | None = None,
    *,
    ks_tests: bool = False,
    cos_vrel: str | None = None,
    n_scatter: int = 10000,
    seed: int = 42,
) -> None:
    """Load scan, run KS tests if requested, then save figs"""
    data, n_files = load_chunks(directory)
    log_info(f"param_utils: loaded {n_files} chunk file(s) from {directory}")
    spec = parse_cos_vrel(cos_vrel)
    data_f, label = apply_cos_vrel_filter(data, spec)
    log_info(f"            filter: {label}")
    if ks_tests:
        run_sampling_tests(
            data_f,
            reference=data if spec is not None else None,
        )
    log_info("")
    if not np.asarray(data_f["success"], bool).any():
        log_info("no cases pass the filter; skipping figures")
        return
    plot_origins(
        data_f,
        out_png,
        label=label,
        n_scatter=n_scatter,
        seed=seed,
    )
    if diagnostics_png is not None:
        plot_diagnostics(data_f, diagnostics_png, label=label)


def main(argv: list[str] | None = None) -> None:
    """Main cli for scan / plot / read"""
    parser = argparse.ArgumentParser(description=f"{VERSION}: scan utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _build_scan_parser(subparsers)
    _build_plot_parser(subparsers)
    _build_read_parser(subparsers)
    args = parser.parse_args(argv)
    _, log_path = setup_logging(
        "ringforge.param_utils",
        log_dir=args.log_dir,
        console_level=logging.WARNING if args.quiet else logging.INFO,
    )
    show_progress = not args.quiet
    log_debug(f"log file: {log_path}")
    log_debug(f"argv: {argv if argv is not None else sys.argv}")

    if args.command == "scan":
        if args.n is None and not args.test:
            raise SystemExit("scan requires --n unless --test is used")
        if args.out is None and not args.test:
            raise SystemExit("scan requires --out unless --test is used")
        n = int(args.n) if args.n is not None else int(_cfg("scan", "n", 10000))
        out_dir = (
            args.out
            if args.out is not None
            else str(_cfg("scan", "out", "data/param_scan_test"))
        )
        workers = (
            int(args.workers)
            if args.workers is not None
            else (int(_cfg("scan", "workers", 16)) if args.test else mp.cpu_count())
        )
        overwrite = bool(args.overwrite or args.test)
        config = ScanConfig(
            ma=float(args.ma),
            mb=float(args.mb),
            mc=float(args.mc),
            rho=float(args.rho),
            a_bin=float(args.a_bin),
            e_bin=float(args.e_bin),
            inc_bin=float(args.inc_bin),
            v_rel=float(args.v_rel),
            geometry=str(args.geometry),
            phase=str(args.phase),
            lookback_periods=tuple(float(x) for x in args.lookback),
            master_seed=int(args.seed),
        )
        run_batch(
            n,
            out_dir,
            config,
            workers=workers,
            chunk=args.chunk,
            start=args.start,
            overwrite=overwrite,
            ks_tests=args.ks_tests and not args.test,
            show_progress=show_progress,
        )
        if args.test:
            summarise_scan(
                out_dir,
                args.plot_out or "media/param_scan_test.png",
                args.plot_diagnostics or "media/param_scan_test_diag.png",
                ks_tests=True,
                n_scatter=10000,
                seed=args.seed,
            )
        return

    if args.command == "plot":
        summarise_scan(
            args.dir,
            args.out,
            args.diagnostics,
            ks_tests=args.ks_tests,
            cos_vrel=args.cos_vrel,
            n_scatter=args.n_scatter,
            seed=args.seed,
        )
        return

    if args.command == "read":
        realisation, chosen = pick_realisation(
            args.dir, seed=args.seed, index=args.index
        )
        print(f"realisation index: {chosen}")
        for key in (
            "ma",
            "mb",
            "mc",
            "rho",
            "a_bin",
            "v_rel",
            "T_bin",
            "mu",
            "b_contact",
            "cos_contact",
            "cos_vrel",
        ):
            if key in realisation:
                print(f"  {key:14s} {fmt_num(realisation[key])}")
        if "lookback_periods" in realisation:
            print(
                "  lookback:     "
                + fmt_num(np.asarray(realisation["lookback_periods"]))
                + " T_bin"
            )
        if args.lookback is not None:
            states, snapshot_index, actual = select_state(
                realisation, float(args.lookback)
            )
            print(
                f"  selected snapshot {snapshot_index} at -{actual:g} T_bin; "
                f"state shape {states.shape}"
            )
            print(states)
        return

    raise RuntimeError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    main()

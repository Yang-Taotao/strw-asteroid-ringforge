"""
Param space scanner. Work in progress.

Pipeline
    Stage 1 - build a stable asteroid binary host (ma > mb) in SI units
    Stage 2 - place bullet (mc < mb) in contact with mb at t=0
    Stage 3 - sample impact geometry and relative velocity from t=0
    Stage 4 - backwards integrate N binary orbital periods for bullet initial conditions
    Stage 5 - write per-realisation states to chunked `npz` files
    Stage 6 - visualize the pre-impact shells from cached `npz` files

Typical CLI
    run 100000 realisations across 16 cores that outputs to data/param_scan_00
    with chunk size 1000, seed 42 for reproducibility

    python param_scan.py \\
        --n 100000 --out data/param_scan_00 --workers 16 --chunk 1000 --seed 42

CLI for resume from crash
    resume from a potential crash that was writing to data/run_00 with seed 42
    first find the idx where it went boom

    ls data/run_00/chunk_*.npz | tail -1

    we then resume from (idx+1)*chunk_size

    python param_scan.py \\
        --n 100000 --start ((idx+1)*chunk_size) --out data/run_01 --seed 42

Multi-node CLI
    run across machines, this requires fixed seed
    node 0

    python param_scan.py \\
        --n 100000 --start 0 --out data/shard00 --seed 42

    node 1
    python param_scan.py \\
        --n 100000 --start 100000 --out data/shard01 --seed 42
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import numpy as np
import rebound
from tqdm import tqdm

# ==================================================================================== #
# Sampler types
# ==================================================================================== #

type ScalarValue = float | int | bool
"""A scalar value that a sampler can return."""

type ParamValue = ScalarValue | np.ndarray
"""The value of one sampled parameter."""

type SampleReturn = ParamValue | dict[str, np.ndarray]
"""The value that one sampler returns for one realisation."""

type RealisationDict = dict[str, object]
"""A flat dict that holds one sampled parameter set."""

# ==================================================================================== #
# Sampler classes
# ==================================================================================== #


def fmt_num(v: object) -> str:
    """
    Format a number for the output of describe().

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


class Param:
    """
    Base class for a parameter sampler.

    A sampler draws a value from a random number generator (RNG).
    The sampler can return a scalar or a dict in the form {subname: value}.
    To open a parameter later, swap the sampler object. Do not change the pipeline.
    """

    def sample(
        self,
        rng: np.random.Generator,
        index: int,
    ) -> SampleReturn:
        """
        Draw one value for the realisation with this index.

        Parameters
            rng
                The RNG for this parameter.
                The function sample() makes it in a fixed way from (master_seed, index).
            index
                The realisation index.
                A fixed sampler can use this index to give equal coverage.
                Other samplers ignore it.

        Returns
            A scalar value or a dict of {subname: value}.

        Raises
            NotImplementedError
                Always, in the base class. A subclass must override this method.
        """
        raise NotImplementedError

    def describe(self) -> str:
        """
        Give a one-line description for describe_space().

        Returns
            The name of the class as text.
        """
        return self.__class__.__name__


class Fixed(Param):
    """
    Sampler that always gives the same value.
    """

    def __init__(self, value: ParamValue) -> None:
        """
        Store the value that sample() returns each time.

        Parameters
            value
                The value that sample() returns.
        """
        self.value: ParamValue = value

    def sample(
        self,
        rng: np.random.Generator,
        index: int,
    ) -> ParamValue:
        """
        Give the stored value. The RNG and the index are not used.

        Parameters
            rng
                Not used.
            index
                Not used.

        Returns
            The stored value. A NumPy array is copied.
        """
        v = self.value
        if isinstance(v, np.ndarray):
            return v.copy()
        return v

    def describe(self) -> str:
        """
        Give the text Fixed(<value>).

        Returns
            The description text.
        """
        return f"Fixed({fmt_num(self.value)})"


class Uniform(Param):
    """
    Sampler that gives a uniform scalar in [lo, hi].
    """

    def __init__(self, lo: float, hi: float) -> None:
        """
        Store the bounds of the interval.

        Parameters
            lo
                Lower bound of the uniform distribution.
            hi
                Upper bound of the uniform distribution.
        """
        self.lo = lo
        self.hi = hi

    def sample(
        self,
        rng: np.random.Generator,
        index: int,
    ) -> float:
        """
        Draw one uniform value in [lo, hi].

        Parameters
            rng
                The RNG for this parameter.
            index
                Not used.

        Returns
            The sampled value.
        """
        return float(rng.uniform(self.lo, self.hi))

    def describe(self) -> str:
        """
        Give the text Uniform(<lo>, <hi>).

        Returns
            The description text.
        """
        return f"Uniform({fmt_num(self.lo)}, {fmt_num(self.hi)})"


class LogUniform(Param):
    """
    Sampler that gives a log-uniform scalar in [lo, hi].

    Both bounds must be positive.
    """

    def __init__(self, lo: float, hi: float) -> None:
        """
        Store the bounds and their logarithms.

        Parameters
            lo
                Lower bound. Must be positive.
            hi
                Upper bound. Must be larger than lo.
        """
        self.lo = lo
        self.hi = hi
        self._log_lo = float(np.log(lo))
        self._log_hi = float(np.log(hi))

    def sample(
        self,
        rng: np.random.Generator,
        index: int,
    ) -> float:
        """
        Draw one log-uniform value in [lo, hi].

        Parameters
            rng
                The RNG for this parameter.
            index
                Not used.

        Returns
            The sampled value.
        """
        return float(np.exp(rng.uniform(self._log_lo, self._log_hi)))

    def describe(self) -> str:
        """
        Give the text LogUniform(<lo>, <hi>).

        Returns
            The description text.
        """
        return f"LogUniform({fmt_num(self.lo)}, {fmt_num(self.hi)})"


class Choice(Param):
    """
    Sampler that picks one value from a fixed list at random.
    """

    def __init__(self, values: list[ParamValue]) -> None:
        """
        Store the pool of values.

        Parameters
            values
                A non-empty list of values.
                The sampler draws from this list with equal probability.
        """
        self.values: list[ParamValue] = list(values)

    def sample(
        self,
        rng: np.random.Generator,
        index: int,
    ) -> ParamValue:
        """
        Draw one value from the pool with equal probability.

        Parameters
            rng
                The RNG for this parameter.
            index
                Not used.

        Returns
            One value from the pool.
        """
        return self.values[int(rng.integers(len(self.values)))]

    def describe(self) -> str:
        """
        Give the text Choice([...]).

        Returns
            The description text.
        """
        return f"Choice({fmt_num(self.values)})"


class CyclicChoice(Param):
    """
    Sampler that steps through a list in order.

    The sampler gives values[index % N].
    """

    def __init__(self, values: list[ParamValue]) -> None:
        """
        Store the pool of values.

        Parameters
            values
                A non-empty list of values.
                The sampler steps through this list in order.
        """
        self.values: list[ParamValue] = list(values)

    def sample(
        self,
        rng: np.random.Generator,
        index: int,
    ) -> ParamValue:
        """
        Give values[index % len(values)].

        Parameters
            rng
                Not used.
            index
                The realisation index.

        Returns
            One value from the pool.
        """
        return self.values[int(index) % len(self.values)]

    def describe(self) -> str:
        """
        Give the text CyclicChoice([...]).

        Returns
            The description text.
        """
        return f"CyclicChoice({fmt_num(self.values)})"


class ImpactGeometry(Param):
    """
    Sampler for the contact and approach directions of the bullet.

    The sampler returns the dict {"offset_dir": (3,), "v_dir": (3,)}.
    Both values are unit vectors in the inertial frame at t = 0.
    Force the physical condition for mc moving into mb: v_dir * offset_dir < 0.
    """

    def sample(
        self,
        rng: np.random.Generator,
        index: int,
    ) -> dict[str, np.ndarray]:
        """
        Draw one pair {offset_dir, v_dir}.

        Parameters
            rng
                The RNG for this parameter.
            index
                Not used.

        Returns
            A dict with the two unit vectors.
        """
        offset_dir = rng.normal(size=3)
        offset_dir /= np.linalg.norm(offset_dir)
        v_dir = rng.normal(size=3)
        v_dir /= np.linalg.norm(v_dir)
        if np.dot(v_dir, offset_dir) > -0.1:
            v_dir -= 2.0 * np.dot(v_dir, offset_dir) * offset_dir
            v_dir /= np.linalg.norm(v_dir)
        return {"offset_dir": offset_dir, "v_dir": v_dir}

    def describe(self) -> str:
        """
        Sampler description print.

        Returns
            The description text.
        """
        return "ImpactGeometry(uniform contact + inward v_rel)"


# ==================================================================================== #
# Parameter space
# ==================================================================================== #


PARAMS: dict[str, Param] = {
    # Masses of the host binary and the bullet in kg.
    # We set ma > mb > mc.
    "ma": Fixed(1.0e15),
    "mb": Fixed(1.0e13),
    "mc": Fixed(1.0e12),
    # Bulk density in kg/m^3.
    # Rubble pile ~ 3000.
    # Ice-rich ~ 900 to 1500.
    # Iron-rich ~ 5000 to 8000.
    "rho": Fixed(3000.0),
    # Binary semi-major axis in m.
    "a_bin": Fixed(1.0e5),
    # Binary eccentricity and inclination.
    "e_bin": Fixed(0.0),
    "inc_bin": Fixed(0.0),
    # Bullet relative velocity at contact in m/s.
    "v_rel": CyclicChoice([100.0]),
    # Impact geometry that gives {'offset_dir', 'v_dir'}.
    # Force the physical condition for mc moving into mb: v_dir * offset_dir < 0.
    # Later: use (impact parameter b, angle theta) to stratify (head on etc).
    # Then the samples are not uniform on the sphere.
    "geometry": ImpactGeometry(),
}

# Schema for the sampled parameters of one realisation.
PARAM_SCHEMA: list[tuple[str, tuple[int, ...]]] = [
    ("ma", ()),  # scalar, kg
    ("mb", ()),  # scalar, kg
    ("mc", ()),  # scalar, kg
    ("rho", ()),  # scalar, kg/m^3
    ("a_bin", ()),  # scalar, m
    ("e_bin", ()),  # scalar, dimensionless
    ("inc_bin", ()),  # scalar, rad
    ("v_rel", ()),  # scalar, m/s
    ("offset_dir", (3,)),  # unit vector from mb centre to mc centre at t=0
    ("v_dir", (3,)),  # unit vector of (v_mc - v_mb) at t=0
]

LOOKBACK_PERIODS: np.ndarray = np.array([10, 25, 50, 100])  # in units of T_bin

# ==================================================================================== #
# Sampler
# ==================================================================================== #


def sample(
    index: int,
    master_seed: int = 42,
    params: dict[str, Param] = PARAMS,
) -> RealisationDict:
    """
    Build the realisation for one index and seed in a fixed way.

    The result is a pure function of (index, master_seed, params).
    A second run with the same three inputs gives the same dict.

    Parameters
        index
            The realisation index.
        master_seed
            The top-level seed.
        params
            A dict {name: sampler} that describes the parameter space.

    Returns
        A flat realisation dict.
        A scalar sampler adds one key.
        A dict sampler, such as ImpactGeometry, merges its sub-keys into the result.
        The result always holds the key "index".
    """
    ss = np.random.SeedSequence([int(master_seed), int(index)])
    children = ss.spawn(len(params))

    out: RealisationDict = {}
    for (name, p), cs in zip(params.items(), children, strict=False):
        rng = np.random.default_rng(cs)
        val: SampleReturn = p.sample(rng, int(index))
        if isinstance(val, dict):
            out.update(val)
        else:
            out[name] = val
    out["index"] = int(index)
    return out


# ==================================================================================== #
# Single realisation
# ==================================================================================== #


def sphere_radius(m: float, rho: float) -> float:
    """
    Give the radius of a uniform sphere.

    Parameters
        m
            The mass in kg.
        rho
            The bulk density in kg/m^3.

    Returns
        The radius in m.
    """
    return (3.0 * m / (4.0 * np.pi * rho)) ** (1.0 / 3.0)


def _extract(sim: rebound.Simulation) -> np.ndarray:
    """
    Take a snapshot of the state of every particle.

    Parameters
        sim
            A REBOUND simulation. The particle order does not change.

    Returns
        An array of shape (N, 6). Each row is [x, y, z, vx, vy, vz].
    """
    return np.array([[p.x, p.y, p.z, p.vx, p.vy, p.vz] for p in sim.particles])


def run_one(real: RealisationDict) -> RealisationDict:
    """
    Build, sample, and integrate one realisation in SI units.

    Place bullet at the surface of mb at t = 0.
    It uses the sampled geometry and the sampled relative velocity.
    Then integrates backwards through each lookback time in `LOOKBACK_PERIODS`.

    Parameters
        real
            The flat realisation dict from sample().

    Returns
        A dict with the sampled parameters, plus state_0, snapshots, T_bin, radii,
        contact, cos_contact, and cos_vrel.
        The dict always holds the key "index".
        When failure, worker() catches the exception before this function returns.
    """
    ma = cast(float, real["ma"])
    mb = cast(float, real["mb"])
    mc = cast(float, real["mc"])
    rho = cast(float, real["rho"])
    a_bin = cast(float, real["a_bin"])
    e_bin = cast(float, real["e_bin"])
    inc_bin = cast(float, real["inc_bin"])
    v_rel = cast(float, real["v_rel"])
    off = cast(np.ndarray, real["offset_dir"])
    vdir = cast(np.ndarray, real["v_dir"])

    r_a = sphere_radius(ma, rho)
    r_b = sphere_radius(mb, rho)
    r_c = sphere_radius(mc, rho)
    contact = r_b + r_c

    # Binary in SI units.
    sim = rebound.Simulation()
    sim.units = ("s", "m", "kg")
    sim.add(m=ma)
    sim.add(m=mb, a=a_bin, e=e_bin, inc=inc_bin)
    sim.move_to_com()

    t_bin = 2.0 * np.pi * np.sqrt(a_bin**3 / (sim.G * (ma + mb)))

    # Bullet at t = 0. The bullet touches mb.
    pb = sim.particles[1]
    sim.add(
        m=mc,
        x=pb.x + contact * off[0],
        y=pb.y + contact * off[1],
        z=pb.z + contact * off[2],
        vx=pb.vx + v_rel * vdir[0],
        vy=pb.vy + v_rel * vdir[1],
        vz=pb.vz + v_rel * vdir[2],
    )

    state_0 = _extract(sim)

    # Classify the impact in the rest frame of mb.
    v_mb = state_0[1, 3:6]
    v_mb_hat = v_mb / (np.linalg.norm(v_mb) + 1e-30)
    cos_contact = float(np.dot(off, v_mb_hat))  # 1 -> trailing edge of mb
    cos_vrel = float(np.dot(vdir, v_mb_hat))  # 1 -> rear-end

    # Backward integration.
    sim.integrator = "ias15"
    sim.dt = t_bin / 200.0

    snapshots = np.zeros((len(LOOKBACK_PERIODS), 3, 6))
    for i, lb in enumerate(LOOKBACK_PERIODS):
        sim.integrate(-lb * t_bin)
        snapshots[i] = _extract(sim)

    params_out: dict[str, object] = {k: real[k] for k, _ in PARAM_SCHEMA}

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
    }


def worker(real: RealisationDict) -> RealisationDict:
    """
    Call run_one() in a pool-safe way.

    The function catches every exception from run_one() and gives back a failed result.
    One bad realisation cannot stop the pool.
    The failed dict holds the original index and an "error" string.

    Parameters
        real
            The flat realisation dict from sample().

    Returns
        The result from run_one(), or the dict
        {"index": ..., "success": False, "error": ...}.
    """
    try:
        return run_one(real)
    except Exception as e:
        return {"index": real["index"], "success": False, "error": repr(e)}


# ==================================================================================== #
# Chunks writer
# ==================================================================================== #


def save_chunk(
    results: list[RealisationDict],
    out_dir: str | Path,
    chunk_id: int,
) -> Path:
    """
    Write one chunk of results to chunk_<id>.npz in one step.

    Parameters
        results
            A list of per-realisation dicts from worker().
        out_dir
            The directory for the `npz` file.
        chunk_id
            The integer chunk id.

    Returns
        The path of the written `npz` file.
    """
    n = len(results)
    nl = len(LOOKBACK_PERIODS)

    out: dict[str, np.ndarray] = {}
    # Per-realisation sampled parameters. The schema gives the shape of each value.
    for name, shape in PARAM_SCHEMA:
        if shape == ():
            out[name] = np.full(n, np.nan)
        else:
            out[name] = np.full((n, *shape), np.nan)

    # State arrays.
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

    for i, r in enumerate(results):
        out["index"][i] = cast(int, r["index"])
        if not bool(r.get("success", False)):
            continue
        out["success"][i] = True
        params = cast(dict[str, object], r["params"])
        for name, _ in PARAM_SCHEMA:
            out[name][i] = cast(ParamValue, params[name])
        out["state_0"][i] = cast(np.ndarray, r["state_0"])
        out["snapshots"][i] = cast(np.ndarray, r["snapshots"])
        out["T_bin"][i] = cast(float, r["T_bin"])
        out["R_a"][i] = cast(float, r["R_a"])
        out["R_b"][i] = cast(float, r["R_b"])
        out["R_c"][i] = cast(float, r["R_c"])
        out["contact"][i] = cast(float, r["contact"])
        out["cos_contact"][i] = cast(float, r["cos_contact"])
        out["cos_vrel"][i] = cast(float, r["cos_vrel"])

    out["lookback_periods"] = LOOKBACK_PERIODS

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    path = out_path / f"chunk_{chunk_id:06d}.npz"
    tmp = path.with_suffix(".npz.tmp")

    # An open handle stops NumPy from adding its own .npz suffix.
    with Path.open(tmp, "wb") as f:
        np.savez_compressed(f, allow_pickle=False, **out)

    tmp.rename(path)
    return path


# ==================================================================================== #
# Parallization
# ==================================================================================== #


def describe_space(
    params: dict[str, Param] = PARAMS,
    master_seed: int = 42,
) -> None:
    """
    Print the parameter space and its top-level seed.

    Parameters
        params
            A dict {name: sampler} to describe.
        master_seed
            The seed shown in the header.
    """
    print("-" * 68)
    print(f"Parameter space   (master_seed = {master_seed})")
    for name, p in params.items():
        print(f"  {name:10s}  {p.describe()}")
    print("-" * 68)


def run_batch(
    n: int,
    out_dir: str | Path,
    master_seed: int = 42,
    workers: int | None = None,
    chunk: int = 2000,
    start: int = 0,
    params: dict[str, Param] = PARAMS,
) -> None:
    """
    Run n realisations across mp pool.
    Make realisations one at a time and send to mp for batch processing.

    Parameters
        n
            The number of realisations.
        out_dir
            The directory for the chunks.
        master_seed
            The seed passed to `sample()`.
        workers
            The pool size. The default is `multiprocessing.cpu_count()`.
        chunk
            The number of results per `npz` file.
        start
            The first realisation index.
            Use this value to resume after a crash or to shard a run across machines.
        params
            A dict {name: sampler} that describes the param space.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if workers is None:
        workers = mp.cpu_count()

    describe_space(params, master_seed)
    print(
        f"Running {n} realisations  (start={start}, workers={workers}, chunk={chunk})"
    )
    print()

    def gen() -> Iterator[RealisationDict]:
        """
        Give one realisation dict for each index in [start, start + n).

        Returns
            An iterator over the realisation dicts.
        """
        for i in range(start, start + n):
            yield sample(i, master_seed, params)

    chunk_id = start // chunk
    buf: list[RealisationDict] = []
    t0 = time.time()

    with (
        mp.Pool(workers) as pool,
        tqdm(
            total=n, unit="case", ncols=100, desc="simulating", smoothing=0.05
        ) as pbar,
    ):
        for res in pool.imap_unordered(worker, gen(), chunksize=4):
            buf.append(res)
            pbar.update(1)

            # Rolling stats in the postfix.
            elapsed = time.time() - t0
            done = pbar.n
            rate = done / elapsed if elapsed > 0 else 0.0
            pbar.set_postfix(rate=f"{rate:.1f}/s", chunk=chunk_id, refresh=False)

            if len(buf) >= chunk:
                save_chunk(buf, out_dir, chunk_id)
                chunk_id += 1
                buf = []
                pbar.set_postfix(rate=f"{rate:.1f}/s", chunk=chunk_id, refresh=True)

    if buf:
        save_chunk(buf, out_dir, chunk_id)

    print(f"\nDone.  {n} realisations in {time.time() - t0:.1f} s  ->  {out_dir}/")


# ==================================================================================== #
# Main
# ==================================================================================== #


def main() -> None:
    """
    Entry: get cli args and do parallelized realizations.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=mp.cpu_count())
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()
    run_batch(args.n, args.out, args.seed, args.workers, args.chunk, args.start)


if __name__ == "__main__":
    main()

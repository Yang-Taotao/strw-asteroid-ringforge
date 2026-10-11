<div align="center">

# strw-asteroid-ringforge

**Rubble-pile asteroid collisions, simulated with REBOUND and rendered to GIF.**

[![Python](https://img.shields.io/badge/python-3.13-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Conda](https://img.shields.io/badge/env-conda--forge%2Fminiforge-green?logo=conda-forge&logoColor=white)](https://github.com/conda-forge/miniforge)
[![REBOUND](https://img.shields.io/badge/backend-REBOUND-orange)](https://rebound.readthedocs.io/)
[![AMUSE](https://img.shields.io/badge/interface-AMUSE-purple)](https://amusecode.github.io/)
[![Status](https://img.shields.io/badge/status-active%20development-yellow)](https://shouldideploy.today/)

*Settle, collide, and animate self-gravitating rubble piles to probe ring formation.*

</div>

---

## Table of contents

- [Overview](#overview)
- [Pipeline](#pipeline)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Project layout](#project-layout)
- [Design notes](#design-notes)
- [Roadmap](#roadmap)
- [Housekeeping](#housekeeping)

---

## Overview

`strw-asteroid-ringforge` builds self-gravitating rubble-pile asteroids.
This follows a power-law size distribution.
We relax the rubble piles under their own gravity.
Then we smash them together and renders the aftermath to an animated `gif`.

> [!NOTE]
> The name is a palce holder, we are investigating if rings around asteroids can form.
>
> Current sim written with native [`REBOUND`](https://github.com/hannorein/rebound).
>
> We aim to use `AMUSE` interface down the road for easy testing across multiple codes.

---

## Pipeline

```text
   ┌────────────────┐     ┌────────────────┐     ┌────────────────┐
   │  Build clump   │───▶│     Settle     │───▶│  Cache state   │
   │  power-law r   │     │  REBOUND IAS15 │     │  hashed .npz   │
   └────────────────┘     └────────────────┘     └────────────────┘
                                                         │
                                               with N unique asteroids
                                                         │
                                                         ▼
   ┌────────────────┐     ┌────────────────┐     ┌────────────────┐
   │   Render GIF   │◀───│  Chunked npz   │◀───│  Collision     │
   │   PIL + Agg    │     │  frame store   │     │  scene builder │
   └────────────────┘     └────────────────┘     └────────────────┘
```

Each stage is independently runnable and cached, so parameter sweeps can
skip the expensive settlement step entirely once a state is on disk.

---

## Quick start

### 1. Clone and enter the repository

```sh
git clone https://github.com/Yang-Taotao/strw-asteroid-ringforge.git
cd strw-asteroid-ringforge
```

### 2. Create and activate the conda environment

```sh
# Typical env when amuse interface is needed
conda create -f env.yml -n amuse-py313
conda activate amuse-py313

# Currently, we rely on native REBOUND
# Simply install REBOUND to your env and activate
```

### 3. Run sim

```sh
# Full pipeline: settle, collide, render, encode
python main.py

# Force re-settlement of every asteroid
python main.py --overwrite-settled

# Keep the per-run chunk directory for post-mortem inspection
python main.py --keep-frames

# Control the chunk write size (frames per .npz)
python main.py --chunk-size 100
```

> [!TIP]
> Settlement is the slow stage (~hours for N ≈ 2000).
> The hash-keyed cache means a re-run with identical parameters skips it entirely.
> We then reuse the cache and thus iterate on the collision and rendering stages freely.

### 4. Find results

| Artifact                    | Location                                            |
|---                          |---                                                  |
| Settled asteroid states     | `./data/settled_N<N>_<hash>.npz`                    |
| Per-run frame chunks        | `./data/collision_frames_<run_id>/chunk_XXXX.npz`   |
| Final animation             | `./media/gif_<run_id>.gif`                          |

---

## Configuration

The simulator is driven entirely by module-level constants at the top of `main.py`.
These are the ones you are most likely to touch:

| Constant            | Purpose                                     | Default          |
|---                  |---                                          |---               |
| `N1, N2, N3`        | Particle counts for host 1, host 2, bullet  | `2100, 900, 300` |
| `SIZE_SLOPE`        | Power-law index `dN/dr \sim r^slope`        | `-3.0`           |
| `R_MIN, R_MAX`      | Radius sampling range \[m\]                 | `50, 1000`       |
| `SETTLE_TIME`       | Relaxation duration \[s\]                   | `108000`         |
| `BULLET_SPEED`      | Relative approach speed \[m/s\]             | `100`            |
| `SEPARATION_FACTOR` | COM separation in asteroid extents          | `6.0`            |
| `POST_IMPACT_TAIL`  | Extra time after first contact \[s\]        | `6000`           |
| `RENDER_DPI`        | Frame rasterisation DPI                     | `100`            |
| `FRAMES_CHUNK_SIZE` | Frames per chunk `.npz`                     | `100`            |
| `OFFSET_1..3`       | Body offsets in units of separation         | per body         |
| `BLKVEL_1..3`       | Body bulk vel in units of `BULLET_SPEED`    | per body         |

> [!IMPORTANT]
> Every parameter that influences the settled state is hashed into the
> cache filename. Change `SIZE_SLOPE` and you get a fresh file; change
> only `BULLET_SPEED` and the settlement is reused.

---

## Project layout

```text
.
├── main.py                    # Entry point: pipeline orchestration
├── env.yml                    # Conda environment snapshot
├── src/ringforge              # Work in progress, module placeholder
├── LICENSE
├── README.md
├── data/
│   ├── settled_*.npz          # Cached asteroid states
│   └── collision_frames_*/    # Per-run chunked frame store
└── media/
    └── gif_*.gif              # Rendered animations
```

---

## Design notes

### Stage 1 — Settle a single asteroid

1. Sample radii from a power law `p(r) \sim r^slope` on `[R_MIN, R_MAX]`.
2. Reject-sample positions inside a sphere of radius `ball_radius_for(N)`.
   1. Repeat until no two particles overlap (with a small buffer).
3. Integrate with **REBOUND's IAS15** integrator for `SETTLE_TIME`.
   1. Use tree gravity plus hardsphere collisions.
4. Verify that the post-settle radius distribution slope.
   1. Compare input slope with post-settle results (PDF and survival-function fits).
5. Save the state as a `(N, 7)` array of `[x, y, z, vx, vy, vz, r]`.

### Stage 2 — Build the three-body scene

Given three settled asteroids `A`, `B`, `C` with `N_A > N_B > N_C`:

- `A` and `B` form a pseudo-binary at the origin.
- `C` is launched from the left as an impactor.
  - With a rear-ending trajectory that injects energy into the system.

> [!NOTE]
> Offsets are expressed as unit vectors scaled by `separation`
>
> With bulk velocities as unit vectors scaled by `BULLET_SPEED`.

This keeps the impact scene geometry legible:

```python
add_settled_asteroid(
    sim,
    state3,
    offset=OFFSET_3 * separation,
    bulk_vel=BLKVEL_3 * BULLET_SPEED,
)
```

### Stage 3 — Render

The renderer holds a single matplotlib figure and one `Circle` artist per particle.
We update the centres at each frame and rasterise through an explicit `FigureCanvasAgg`.
This is bound to a `numpy` buffer.
This keeps the per-frame cost dominated by the integration, not by artist construction.

Frames are accumulated in memory and flushed to numbered chunk files `chunk_XXXX.npz`.
With `FRAMES_CHUNK_SIZE`, we aim to be efficient in memory usage.

### Stage 4 — Encode

Chunk files are read back one at a time.
Each RGB frame is converted to paletted P-mode (1 byte/pixel).
Then we append to the `PIL` frame list.
Peak memory during encoding is then bounded by the P-mode size, not by the raw RGB size.

---

## Roadmap

### Primary

- [ ] Implement `tidymess` backend with the AMUSE interface, following
      the current REBOUND design.
- [ ] Add a parallelisation module for HPC deployments.

### Intermediate

- [ ] Backwards calculation from the moment of impact for init param sweep constraints.
- [ ] Parameter sweep utility, driven by the hash-keyed cache.
- [ ] Collision-outcome and stopping criteria derived from bounding
      energy rather than fixed `POST_IMPACT_TAIL`.

---

## Housekeeping

### Environment

Requires env management via [**conda**](https://github.com/conda-forge/miniforge).

The `env.yml` file is generated with:

```sh
conda env export --no-builds | grep -v "^prefix:" > env.yml
```

To reproduce and activate this environment under a custom name such as `amuse-py313`:

```sh
conda create -f env.yml -n amuse-py313
conda activate amuse-py313
```

### REBOUND

> [!WARNING]
> The current implementation relies on a **native** `REBOUND` build.
> This is different from the common `AMUSE` interface usage for `REBOUND`.

### Conventions

- Frames are stored as `(N, H, W, 3)` `uint8` arrays in uncompressed `.npz` files.
- Every unique cached artifact carries a short `SHA1` fingerprint of the input params.

---

<div align="center">

**Built with coffee and tea at [STRW](https://www.strw.leidenuniv.nl/)**

</div>

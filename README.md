# strw-asteroid-rings

This is the README document for strw-asteroid-rings

**Dated:** 2026-September-02

## housekeeping

### env

Requires environment management via `conda` ([conda-forge/miniforge](https://github.com/conda-forge/miniforge))

The `env.yml` file is generated via:

```sh
conda env export --no-builds | grep -v "^prefix:" > env.yml
```

To reproduce and activate this environment under a custom name such as `amuse-py313`, use:

```sh
conda create -f env.yml -n amuse-py313
conda activate amuse-py313
```

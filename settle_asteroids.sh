#!/usr/bin/env bash
# Build and settle the three collision bodies, then exit before any impact run.
#
# Defaults are the paper masses at 1,000 particles each
# override any value in the environment, for example:
#
#   M1=1e17 M2=5e16 M3=1e15 N1=2000 N2=2000 N3=2000 ./settle_asteroids.sh
#
# Existing caches are validated and reused. Set OVERWRITE=1 to force new ones.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONDA_ENV="${CONDA_ENV:-amuse-py313}"

M1="${M1:-1e17}"
M2="${M2:-5e16}"
M3="${M3:-1e15}"
N1="${N1:-1000}"
N2="${N2:-1000}"
N3="${N3:-1000}"
DENSITY="${DENSITY:-500}"
SETTLE_TIME="${SETTLE_TIME:-108000}"
DT_SETTLE="${DT_SETTLE:-10}"
SOFTENING="${SOFTENING:-20}"
PLACEMENT_SEED="${PLACEMENT_SEED:-42}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-$SCRIPT_DIR/logs/settle_asteroids_${STAMP}.log}"

if ! command -v conda >/dev/null 2>&1; then
	echo "conda is not on PATH" >&2
	exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$SCRIPT_DIR/.mplconfig}"
mkdir -p "$MPLCONFIGDIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

overwrite_args=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
	overwrite_args+=(--overwrite-settled)
fi

mkdir -p "$(dirname "$LOG_FILE")"
python main.py \
	--settle-only \
	--resolve-primary \
	--ma "$M1" --mb "$M2" --mc "$M3" \
	--n1 "$N1" --n2 "$N2" --n3 "$N3" \
	--density "$DENSITY" \
	--settle-time "$SETTLE_TIME" \
	--dt-settle "$DT_SETTLE" \
	--softening "$SOFTENING" \
	--placement-seed "$PLACEMENT_SEED" \
	--no-media \
	--no-progress \
	--log-file "$LOG_FILE" \
	"${overwrite_args[@]}"

echo "settled bodies; log: $LOG_FILE"

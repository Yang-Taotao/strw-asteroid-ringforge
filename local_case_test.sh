#!/usr/bin/env bash
# Quick local check:
#   1. plot the existing parameter scan and diagnostics
#   2. run 9 real parameter-scan cases at 1000 particles, suppress media gen
#   3. merge into impact-scan plot
#
#
# Usage:
#   ./local_case_test.sh [seed]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/run_cases.sh" ]]; then
	ROOT="$SCRIPT_DIR"
else
	ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
cd "$ROOT"

CONDA_ENV="${CONDA_ENV:-amuse-py313}"
SEED="${1:-42}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="media/local_test_${STAMP}"
SCAN_DIR="data/param_scan_1m"

if ! command -v conda >/dev/null 2>&1; then
	echo "conda is not on PATH" >&2
	exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT/.mplconfig}"
mkdir -p "$MPLCONFIGDIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

mkdir -p "$OUT"

if [[ -d "$SCAN_DIR" ]]; then
	python param_utils.py plot "$SCAN_DIR" \
		--out "$OUT/param_scan.png" \
		--diagnostics "$OUT/param_scan_diagnostics.png" \
		--ks-tests
else
	echo "no $SCAN_DIR; running the standard 10k test scan first"
	python param_utils.py scan --test \
		--out "$SCAN_DIR" \
		--plot-out "$OUT/param_scan.png" \
		--plot-diagnostics "$OUT/param_scan_diagnostics.png" \
		--ks-tests
fi

CASE_DIR="$OUT/cases" \
	N_CASES=9 \
	N_REPEATS=1 \
	N_JOBS=6 \
	N_PARTICLES=1000 \
	DENSITY=500 \
	POST_IMPACT_TAIL=10000 \
	DT_HIT=1 \
	IMPACT_TARGETS="-2 -1.5 -1 -0.5 0 0.5 1 1.5 2" \
	LOG_FILE="$OUT/run_cases.log" \
	PLOT_OUT="$OUT/impact_scan.png" \
	JSON_OUT="$OUT/impact_scan.json" \
	./run_cases.sh "$SEED"

echo
echo "Local test complete"
echo "  parameter scan      : $OUT/param_scan.png"
echo "  diagnostics         : $OUT/param_scan_diagnostics.png"
echo "  impact scan         : $OUT/impact_scan.png"
echo "  case rows           : $OUT/cases"
echo "  batch log           : $OUT/run_cases.log"

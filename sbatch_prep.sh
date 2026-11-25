#!/usr/bin/env bash
# Build/settle asteroids and generate param scan for each new experiment.
# Run once before submitting jobs.
#
# Existing settled caches are reused.
# Existing scan directories are skipped unless OVERWRITE_SCAN=1.
#
# Usage:
#   ./prepare_experiments.sh
#
# Environment:
#   N_SCAN=1000000, SCAN_WORKERS=32, SETTLE_TIME=108000, DT_SETTLE=10,
#   SOFTENING=20, A_BIN=1e4, V_REL=100, OVERWRITE_SCAN=0.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONDA_ENV="${CONDA_ENV:-amuse-py313}"
N_SCAN="${N_SCAN:-1000000}"
SCAN_WORKERS="${SCAN_WORKERS:-32}"
A_BIN="${A_BIN:-1e4}"
V_REL="${V_REL:-100}"

if ! command -v conda >/dev/null 2>&1; then
	echo "conda is not on PATH" >&2
	exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$SCRIPT_DIR/.mplconfig}"
mkdir -p "$MPLCONFIGDIR" logs

targets=($(seq -f '%.2f' -2 0.01 2))

# tag|m1|m2|m3|density
specs=(
	"paper_icy|1e17|5e16|1e15|500"
	"paper_rocky|1e17|5e16|1e15|3000"
	"equal_rocky|1e15|1e15|1e15|3000"
	"equal_icy|1e15|1e15|1e15|500"
)

for spec in "${specs[@]}"; do
	IFS='|' read -r tag m1 m2 m3 rho <<<"$spec"
	scan="data/param_scan_${tag}"
	echo
	echo "=== ${tag}: M=(${m1}, ${m2}, ${m3}) kg  rho=${rho} kg/m3"

	LOG_FILE="$SCRIPT_DIR/logs/settle_${tag}.log" \
		M1="$m1" M2="$m2" M3="$m3" N1=1000 N2=1000 N3=1000 \
		DENSITY="$rho" SETTLE_TIME="${SETTLE_TIME:-108000}" \
		DT_SETTLE="${DT_SETTLE:-10}" SOFTENING="${SOFTENING:-20}" \
		./settle_asteroids.sh

	if [[ -d "$scan" && "${OVERWRITE_SCAN:-0}" != "1" ]]; then
		echo "scan exists, skipping: $scan"
		continue
	fi
	if [[ "${OVERWRITE_SCAN:-0}" == "1" ]]; then
		rm -rf "$scan"
	fi

	python param_utils.py scan \
		--n "$N_SCAN" \
		--out "$scan" \
		--ma "$m1" --mb "$m2" --mc "$m3" \
		--rho "$rho" \
		--a-bin "$A_BIN" \
		--v-rel "$V_REL" \
		--impact-parameters "${targets[@]}" \
		--workers "$SCAN_WORKERS" \
		--seed 42 \
		--log-file "logs/param_scan_${tag}.log"
	echo "scan ready: $scan"
done

echo
echo "Experiments prepared:"
for spec in "${specs[@]}"; do
	IFS='|' read -r tag _ _ _ _ <<<"$spec"
	echo "  ${tag}: data/param_scan_${tag}"
done

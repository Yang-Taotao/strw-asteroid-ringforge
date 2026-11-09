#!/usr/bin/env bash
# Select and run a fixed sample of parameter-scan cases, then plot their rows.
#
# Per-case results are saved as JSON under data/run_cases/<seed>/.
# Finished cases are skipped on a rerun, so a long batch can be resumed.
# Native Python (case_scan.py) does selection and plotting; this script only
# activates the environment, runs the jobs, and calls those subcommands.
#
# Usage:
#   ./run_cases.sh [seed]
#
# Environment overrides: SCAN_DIR, N_CASES, N_JOBS.

set -euo pipefail

# env
CONDA_ENV="${CONDA_ENV:-amuse-py313}"
if ! command -v conda >/dev/null 2>&1; then
	echo "conda is not on PATH" >&2
	exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# cfg
SCAN_DIR="${SCAN_DIR:-data/param_scan_test}"
N_CASES="${N_CASES:-12}"
N_JOBS="${N_JOBS:-6}"
CASE_SEED="${1:-42}"
CASE_DIR="data/run_cases/${CASE_SEED}"
INDICES="${CASE_DIR}/indices.txt"

mkdir -p "$CASE_DIR" logs/run_cases media

python case_scan.py select \
	--scan-dir "$SCAN_DIR" \
	--count "$N_CASES" \
	--seed "$CASE_SEED" \
	--out "$INDICES"

run_one() {
	local idx="$1"
	local row="${CASE_DIR}/case_${idx}.json"
	if [[ -s "$row" && -f "${row}.done" ]]; then
		echo "=== [$(date +'%F %T')] skip   param-index=$idx"
		return 0
	fi

	echo ">>> [$(date +'%F %T')] start  param-index=$idx"
	if python main.py \
		--param-path "$SCAN_DIR" \
		--param-index "$idx" \
		--seed "$CASE_SEED" \
		--particles-per-body 1000 \
		--start-at contact \
		--no-parallel-settle \
		--media-format mp4 \
		--steps-per-frame 10 \
		--fig-size-in 6 --render-dpi 80 \
		--quiet --no-progress \
		--log-dir logs/run_cases \
		--case-scan-out "$row"; then
		touch "${row}.done"
		echo "<<< [$(date +'%F %T')] done   param-index=$idx"
	else
		local rc=$?
		echo "!!! [$(date +'%F %T')] FAILED param-index=$idx (exit=$rc)"
		return "$rc"
	fi
}

export SCAN_DIR CASE_DIR CASE_SEED
export -f run_one

mapfile -t CASES <"$INDICES"
if ((${#CASES[@]} == 0)); then
	echo "no cases selected" >&2
	exit 1
fi

printf '%s\n' "${CASES[@]}" |
	xargs -P "$N_JOBS" -I {} bash -c 'run_one "$1"' _ {}

python case_scan.py merge \
	--case-dir "$CASE_DIR" \
	--count "$N_CASES" \
	--out media/impact_scan.png

#!/usr/bin/env bash
# Select and run a fixed sample of parameter-scan cases, then plot their rows.
#
# Selection groups cases by impact parameter: `N_CASES / N_REPEATS` targets,
# with `N_REPEATS` nearby realisations per target
# Each target is chosen from the successful parameter-scan rows.
#
# Finished cases are skipped on a rerun, so a long batch can be resumed.
# Native Python (case_scan.py) does selection and plotting; this script only
# activates the environment, runs the jobs, and calls those subcommands.
#
# Usage:
#   ./run_cases.sh [seed]
#
# Environment overrides:
# SCAN_DIR, CASE_DIR, N_CASES, N_REPEATS, N_JOBS,
# N_PARTICLES, DENSITY, POST_IMPACT_TAIL, DT_HIT, PLOT_OUT, JSON_OUT,
# LOG_FILE, SURFACE_DENSITY_TARGET, SURFACE_DENSITY_OUT,
# SURFACE_DENSITY_DIFFUSION.

set -euo pipefail

# env
CONDA_ENV="${CONDA_ENV:-amuse-py313}"
if ! command -v conda >/dev/null 2>&1; then
	echo "conda is not on PATH" >&2
	exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.mplconfig}"
mkdir -p "$MPLCONFIGDIR"

# cfg
SCAN_DIR="${SCAN_DIR:-data/param_scan_1m}"
N_CASES="${N_CASES:-9}"
N_REPEATS="${N_REPEATS:-1}"
N_JOBS="${N_JOBS:-6}"
N_PARTICLES="${N_PARTICLES:-1000}"
DENSITY="${DENSITY:-500}"
CASE_SEED="${1:-42}"
CASE_DIR="${CASE_DIR:-data/run_cases/${CASE_SEED}}"
POST_IMPACT_TAIL="${POST_IMPACT_TAIL:-10000}"
DT_HIT="${DT_HIT:-1}"
IMPACT_TARGETS="${IMPACT_TARGETS:--2 -1.5 -1 -0.5 0 0.5 1 1.5 2}"
PLOT_OUT="${PLOT_OUT:-media/impact_scan.png}"
JSON_OUT="${JSON_OUT:-data/impact_scan_run_cases.json}"
SURFACE_DENSITY_TARGET="${SURFACE_DENSITY_TARGET:-}"
SURFACE_DENSITY_OUT="${SURFACE_DENSITY_OUT:-media/surface_density.png}"
SURFACE_DENSITY_DIFFUSION="${SURFACE_DENSITY_DIFFUSION:-0.05}"
SELECTION="${CASE_DIR}/selection.tsv"
LOG_FILE="${LOG_FILE:-logs/run_cases/batch_${CASE_SEED}_$(date +%Y%m%d_%H%M%S).log}"

mkdir -p "$CASE_DIR" media
mkdir -p "$(dirname "$LOG_FILE")"

if [[ -n "$IMPACT_TARGETS" ]]; then
	python case_scan.py select \
		--scan-dir "$SCAN_DIR" \
		--count "$N_CASES" \
		--repeats "$N_REPEATS" \
		--seed "$CASE_SEED" \
		--impact-parameters $IMPACT_TARGETS \
		--out "$SELECTION"
else
	python case_scan.py select \
		--scan-dir "$SCAN_DIR" \
		--count "$N_CASES" \
		--repeats "$N_REPEATS" \
		--seed "$CASE_SEED" \
		--out "$SELECTION"
fi

run_one() {
	local target="$1"
	local idx="$2"
	local row="${CASE_DIR}/case_${idx}.json"
	if [[ -s "$row" && -f "${row}.done" ]]; then
		echo "=== [$(date +'%F %T')] skip   target=$target param-index=$idx"
		return 0
	fi

	local surface_args=()
	if [[ -n "$SURFACE_DENSITY_TARGET" && "$target" == "$SURFACE_DENSITY_TARGET" ]]; then
		surface_args=(
			--surface-density "$SURFACE_DENSITY_OUT"
			--surface-density-diffusion "$SURFACE_DENSITY_DIFFUSION"
		)
	fi

	echo ">>> [$(date +'%F %T')] start  target=$target param-index=$idx"
	if python main.py \
		--param-path "$SCAN_DIR" \
		--param-index "$idx" \
		--seed "$CASE_SEED" \
		--particles-per-body "$N_PARTICLES" \
		--density "$DENSITY" \
		--start-at contact \
		--no-parallel-settle \
		--no-media \
		--steps-per-frame 10 \
		--post-impact-tail "$POST_IMPACT_TAIL" \
		--dt-hit "$DT_HIT" \
		--fig-size-in 6 --render-dpi 80 \
		--quiet --no-progress \
		--log-file "$LOG_FILE" \
		--no-run-json \
		--case-scan-target "$target" \
		"${surface_args[@]}" \
		--case-scan-out "$row"; then
		touch "${row}.done"
		echo "<<< [$(date +'%F %T')] done   target=$target param-index=$idx"
	else
		local rc=$?
		echo "!!! [$(date +'%F %T')] FAILED target=$target param-index=$idx (exit=$rc)"
		return "$rc"
	fi
}

export SCAN_DIR CASE_DIR CASE_SEED N_PARTICLES DENSITY POST_IMPACT_TAIL DT_HIT PLOT_OUT JSON_OUT
export LOG_FILE SURFACE_DENSITY_TARGET SURFACE_DENSITY_OUT SURFACE_DENSITY_DIFFUSION
export -f run_one

TASKS=()
while IFS=$'\t' read -r target idx _b; do
	[[ "$target" == \#* ]] && continue
	TASKS+=("$target $idx")
done <"$SELECTION"

if ((${#TASKS[@]} == 0)); then
	echo "no cases selected" >&2
	exit 1
fi

printf '%s\n' "${TASKS[@]}" |
	xargs -P "$N_JOBS" -n 2 bash -c 'run_one "$1" "$2"' _

echo "batch log: $LOG_FILE"
python case_scan.py merge \
	--case-dir "$CASE_DIR" \
	--count "$N_CASES" \
	--out "$PLOT_OUT" \
	--json-out "$JSON_OUT"

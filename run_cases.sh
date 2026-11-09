#!/usr/bin/env bash
# Select and run a fixed sample of parameter-scan cases, then plot their rows.
#
# Selection groups cases by impact parameter: `N_CASES / N_REPEATS` targets,
# with `N_REPEATS` nearby realisations per target
# Each target is chosen from the successful parameter-scan rows.
#
# Per-case results are saved as JSON under data/run_cases/<seed>/.
# Finished cases are skipped on a rerun, so a long batch can be resumed.
# Native Python (case_scan.py) does selection and plotting; this script only
# activates the environment, runs the jobs, and calls those subcommands.
#
# Usage:
#   ./run_cases.sh [seed]
#
# Environment overrides: SCAN_DIR, CASE_DIR, N_CASES, N_REPEATS, N_JOBS,
# N_PARTICLES, POST_IMPACT_TAIL, DT_HIT, PLOT_OUT, JSON_OUT.

set -euo pipefail

# env
CONDA_ENV="${CONDA_ENV:-amuse-py313}"
if ! command -v conda >/dev/null 2>&1; then
	echo "conda is not on PATH" >&2
	exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/ringforge-mpl}"
mkdir -p "$MPLCONFIGDIR"

# cfg
SCAN_DIR="${SCAN_DIR:-data/param_scan_test}"
N_CASES="${N_CASES:-24}"
N_REPEATS="${N_REPEATS:-3}"
N_JOBS="${N_JOBS:-6}"
N_PARTICLES="${N_PARTICLES:-10000}"
CASE_SEED="${1:-42}"
CASE_DIR="${CASE_DIR:-data/run_cases/${CASE_SEED}}"
POST_IMPACT_TAIL="${POST_IMPACT_TAIL:-18000}"
DT_HIT="${DT_HIT:-1}"
PLOT_OUT="${PLOT_OUT:-media/impact_scan.png}"
JSON_OUT="${JSON_OUT:-data/impact_scan_run_cases.json}"
SELECTION="${CASE_DIR}/selection.tsv"

mkdir -p "$CASE_DIR" logs/run_cases media

python case_scan.py select \
	--scan-dir "$SCAN_DIR" \
	--count "$N_CASES" \
	--repeats "$N_REPEATS" \
	--seed "$CASE_SEED" \
	--out "$SELECTION"

run_one() {
	local target="$1"
	local idx="$2"
	local row="${CASE_DIR}/case_${idx}.json"
	if [[ -s "$row" && -f "${row}.done" ]]; then
		echo "=== [$(date +'%F %T')] skip   target=$target param-index=$idx"
		return 0
	fi

	echo ">>> [$(date +'%F %T')] start  target=$target param-index=$idx"
	if python main.py \
		--param-path "$SCAN_DIR" \
		--param-index "$idx" \
		--seed "$CASE_SEED" \
		--particles-per-body "$N_PARTICLES" \
		--start-at contact \
		--no-parallel-settle \
		--no-media \
		--steps-per-frame 10 \
		--post-impact-tail "$POST_IMPACT_TAIL" \
		--dt-hit "$DT_HIT" \
		--fig-size-in 6 --render-dpi 80 \
		--quiet --no-progress \
		--log-dir logs/run_cases \
		--case-scan-target "$target" \
		--case-scan-out "$row"; then
		touch "${row}.done"
		echo "<<< [$(date +'%F %T')] done   target=$target param-index=$idx"
	else
		local rc=$?
		echo "!!! [$(date +'%F %T')] FAILED target=$target param-index=$idx (exit=$rc)"
		return "$rc"
	fi
}

export SCAN_DIR CASE_DIR CASE_SEED N_PARTICLES POST_IMPACT_TAIL DT_HIT PLOT_OUT JSON_OUT
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

python case_scan.py merge \
	--case-dir "$CASE_DIR" \
	--count "$N_CASES" \
	--out "$PLOT_OUT" \
	--json-out "$JSON_OUT"

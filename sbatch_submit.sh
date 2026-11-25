#!/usr/bin/env bash
# Submit the prepared impact-scan jobs, one per experiment.
# Run prepare_experiments.sh first.
#
# Usage:
#   ./submit_experiments.sh
#   DRY_RUN=1 ./submit_experiments.sh
#
# Each job uses 20 cores on one node and the 24 h limit in cluster_impact_scan.sh.
# The jobs run simultaneously.
# The default is 3208 cases (8 per b, ~5 h).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p logs media

# tag|scan|density
specs=(
	"paper_icy|1e17|5e16|1e15|500"
	"paper_rocky|data/param_scan_paper_rocky|3000"
	"equal_rocky|data/param_scan_equal_rocky|3000"
	"equal_icy|data/param_scan_equal_icy|500"
)

for spec in "${specs[@]}"; do
	IFS='|' read -r tag scan density <<<"$spec"
	if [[ ! -d "$scan" ]]; then
		echo "missing scan: $scan; run ./prepare_experiments.sh first" >&2
		exit 1
	fi
	cmd=(
		sbatch
		--export=ALL
		--job-name="impact_${tag}"
		--output="logs/impact_${tag}_%j.out"
		--error="logs/impact_${tag}_%j.err"
		cluster_impact_scan.sh
	)
	if [[ "${DRY_RUN:-0}" == "1" ]]; then
		printf 'TAG=%q SCAN_DIR=%q DENSITY=%q N_CASES=%q N_REPEATS=%q %q' \
			"$tag" "$scan" "$density" "${N_CASES:-3208}" "${N_REPEATS:-8}" "${cmd[0]}"
		printf ' %q' "${cmd[@]:1}"
		printf '\n'
	else
		TAG="$tag" SCAN_DIR="$scan" DENSITY="$density" \
			N_CASES="${N_CASES:-3208}" N_REPEATS="${N_REPEATS:-8}" \
			"${cmd[@]}"
	fi
done

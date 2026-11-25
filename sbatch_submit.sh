#!/usr/bin/env bash
# Submit the preparation job and then the four impact-scan jobs.
#
# Usage:
#   ./sbatch_submit.sh
#   DRY_RUN=1 ./sbatch_submit.sh
#
# The impact jobs wait for the prep job with --dependency=afterok.
# If preparation was already submitted or completed:
#   PREP_JOBID=<jobid> ./sbatch_submit.sh
#   SKIP_PREP=1 ./sbatch_submit.sh   # scans already exist

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p logs media

# Submit/identify the preparation job first.
prep_job="${PREP_JOBID:-}"
if [[ "${SKIP_PREP:-0}" == "1" ]]; then
	prep_job=""
elif [[ -n "$prep_job" ]]; then
	echo "using existing prep job: $prep_job"
elif [[ "${DRY_RUN:-0}" == "1" ]]; then
	echo "sbatch sbatch_prep.sh"
	prep_job="PREP_JOBID"
else
	prep_job=$(sbatch --parsable sbatch_prep.sh)
	echo "prep job: $prep_job"
fi

# tag|scan|density
specs=(
	"paper_icy|data/param_scan_paper_icy|500"
	"paper_rocky|data/param_scan_paper_rocky|3000"
	"equal_rocky|data/param_scan_equal_rocky|3000"
	"equal_icy|data/param_scan_equal_icy|500"
)

for spec in "${specs[@]}"; do
	IFS='|' read -r tag scan density <<<"$spec"
	if [[ "${SKIP_PREP:-0}" == "1" && ! -d "$scan" ]]; then
		echo "missing scan: $scan; run sbatch_prep.sh first" >&2
		exit 1
	fi

	cmd=(
		sbatch
		--parsable
		--export=ALL
		--job-name="impact_${tag}"
		--output="logs/impact_${tag}_%j.out"
		--error="logs/impact_${tag}_%j.err"
	)
	if [[ -n "$prep_job" ]]; then
		cmd+=(--dependency="afterok:${prep_job}")
	fi
	cmd+=(cluster_impact_scan.sh)

	if [[ "${DRY_RUN:-0}" == "1" ]]; then
		printf 'TAG=%q SCAN_DIR=%q DENSITY=%q N_CASES=%q N_REPEATS=%q' \
			"$tag" "$scan" "$density" "${N_CASES:-3208}" "${N_REPEATS:-8}"
		printf ' %q' "${cmd[@]}"
		printf '\n'
	else
		TAG="$tag" SCAN_DIR="$scan" DENSITY="$density" \
			N_CASES="${N_CASES:-3208}" N_REPEATS="${N_REPEATS:-8}" \
			"${cmd[@]}"
	fi
done

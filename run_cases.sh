#!/usr/bin/env bash
# Run several collision cases in parallel
# Init condition from param scan results

set -u

# env
CONDA_ENV="amuse-py313"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# cfg
N_JOBS=6 # num of concurrent param idx
START=0  # first param idx
END=5    # last param idx

# case - run_one case
# log start and finish with exit code
run_one() {
	local idx="$1"
	echo ">>> [$(date +'%F %T')] start  param-index=$idx"
	python main.py \
		--param-path data/param_scan_test \
		--param-index "$idx" \
		--particles-per-body 1000 \
		--start-at contact \
		--no-parallel-settle \
		--media-format mp4 \
		--steps-per-frame 10 \
		--fig-size-in 6 --render-dpi 80 \
		--quiet \
		--log-dir logs/run_cases
	local rc=$?
	if ((rc == 0)); then
		echo "<<< [$(date +'%F %T')] done   param-index=$idx"
	else
		echo "!!! [$(date +'%F %T')] FAILED param-index=$idx (exit=$rc)"
	fi
	return $rc
}

export -f run_one

# exec - run all cases
# one case per idx, max N_JOBS
seq "$START" "$END" |
	xargs -P "$N_JOBS" -I {} bash -c 'run_one "$1"' _ {}

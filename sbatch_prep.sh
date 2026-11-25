#!/bin/bash
#SBATCH --job-name=prep_impacts
#SBATCH --export=ALL
#SBATCH --output=logs/prep_%j.out
#SBATCH --error=logs/prep_%j.err
#SBATCH --mail-user="yangtaotao0524@gmail.com"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --time=04:00:00
#SBATCH --partition=cpu-zen4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G

# Prepare the dust-collision experiments on a compute node:
#   - settle the three bodies for each experiment
#   - write the direct-b param_scan for each experiment
#
# Submit from the repository root:
#   sbatch sbatch_prep.sh
#
# Existing settled caches are reused. Existing scan directories are skipped
# unless OVERWRITE_SCAN=1 (which replaces that scan directory).

set -euo pipefail

module purge
module load ALICE/default
module load Miniforge3

cd "${SLURM_SUBMIT_DIR:-$PWD}"

if command -v conda >/dev/null 2>&1; then
	source "$(conda info --base)/etc/profile.d/conda.sh"
fi
conda activate "${CONDA_ENV:-amuse-py313}"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.mplconfig}"
mkdir -p "$MPLCONFIGDIR" logs media

N_SCAN="${N_SCAN:-1000000}"
SCAN_WORKERS="${SCAN_WORKERS:-32}"
A_BIN="${A_BIN:-1e4}"
V_REL="${V_REL:-100}"

targets=($(seq -f '%.2f' -2 0.01 2))

# tag|m1|m2|m3|density
specs=(
	"paper_icy|1e17|5e16|1e15|500"
	"paper_rocky|1e17|5e16|1e15|3000"
	"equal_rocky|1e15|1e15|1e15|3000"
	"equal_icy|1e15|1e15|1e15|500"
)

echo "[$SHELL] prep on $(hostname), ${SLURM_CPUS_ON_NODE:-?} CPUs, ${SLURM_MEM_PER_NODE:-?} MB"
echo "[$SHELL] scan: N=$N_SCAN workers=$SCAN_WORKERS impact values=${#targets[@]}"

for spec in "${specs[@]}"; do
	IFS='|' read -r tag m1 m2 m3 rho <<<"$spec"
	scan="data/param_scan_${tag}"
	echo
	echo "=== ${tag}: M=(${m1}, ${m2}, ${m3}) kg  rho=${rho} kg/m3"

	LOG_FILE="$PWD/logs/settle_${tag}.log" \
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
echo "[$SHELL] done"

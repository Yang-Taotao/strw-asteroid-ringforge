#!/bin/bash
#SBATCH --job-name=param_scan
#SBATCH --export=ALL
#SBATCH --output=logs/param_scan_%j.out
#SBATCH --error=logs/param_scan_%j.err
#SBATCH --mail-user="yangtaotao0524@gmail.com"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --time=04:00:00
#SBATCH --partition=cpu-zen4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G

# Initial parameter scan for the Matt-paper masses, using the direct signed
# impact-parameter sampler:
#   - 401 impact parameters, -2.00..+2.00 step 0.01
#   - 1,000,000 realisations
#   - 100 chunks of 10,000
#   - masses 1e17 / 5e16 / 1e15 kg, density 500 kg/m3
#
# This only writes the scan; it does not run any impact simulations.
# Set OVERWRITE=1 to replace an existing output directory.
#
# Submit from the repository root:
#   sbatch cluster_param_scan.sh

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

CONDA_ENV="${CONDA_ENV:-amuse-py313}"
N="${N:-1000000}"
OUT="${OUT:-data/param_scan_1m}"
WORKERS="${WORKERS:-32}"
CHUNK="${CHUNK:-10000}"
SEED="${SEED:-42}"
JOB_ID="${SLURM_JOB_ID:-manual}"
LOG_FILE="${LOG_FILE:-logs/param_scan_${JOB_ID}.log}"
PLOT_LOG="${PLOT_LOG:-logs/param_scan_plot_${JOB_ID}.log}"

mapfile -t IMPACT_PARAMETERS < <(seq -f '%.2f' -2 0.01 2)

if [[ -d "$OUT" && "${OVERWRITE:-0}" != "1" ]]; then
	echo "refusing to overwrite $OUT; set OVERWRITE=1 to replace it" >&2
	exit 1
fi

overwrite_args=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
	overwrite_args+=(--overwrite)
fi

echo "[$SHELL] param scan: N=$N workers=$WORKERS chunk=$CHUNK"
echo "[$SHELL] output: $OUT"
echo "[$SHELL] impact parameters: ${#IMPACT_PARAMETERS[@]} values"
echo "[$SHELL] scan log: $LOG_FILE"

python param_utils.py scan \
	--n "$N" \
	--out "$OUT" \
	--workers "$WORKERS" \
	--chunk "$CHUNK" \
	--seed "$SEED" \
	--geometry uniform \
	--phase random \
	--impact-parameters "${IMPACT_PARAMETERS[@]}" \
	--ma 1e17 --mb 5e16 --mc 1e15 \
	--rho 500 \
	--a-bin 1e4 \
	--v-rel 100 \
	--quiet \
	--log-file "$LOG_FILE" \
	"${overwrite_args[@]}"

python param_utils.py plot "$OUT" \
	--out media/param_scan_1m.png \
	--diagnostics media/param_scan_1m_diag.png \
	--ks-tests \
	--quiet \
	--log-file "$PLOT_LOG"

echo "[$SHELL] scan data: $OUT"
echo "[$SHELL] plot: media/param_scan_1m.png"
echo "[$SHELL] diagnostics: media/param_scan_1m_diag.png"
echo "[$SHELL] log: $LOG_FILE"

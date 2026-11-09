#!/bin/bash
#SBATCH --job-name=impact_scan
#SBATCH --export=NONE
#SBATCH --output=logs/impact_scan_%j.out
#SBATCH --error=logs/impact_scan_%j.err
#SBATCH --mail-user="yangtaotao0524@gmail.com"
#SBATCH --mail-type=END,FAIL
#SBATCH --time=24:00:00
#SBATCH --partition=cpu-zen4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G

# Production impact scan:
#   - 40 impact-parameter targets * 3 realisations = 120 cases
#   - 10,000 particles per resolved body
#   - no media gen, one JSON row per case
#   - merged into media/impact_scan_cluster_<jobid>.png
#
# Submit from the repo root:
#   sbatch cluster_impact_scan.sh

set -euo pipefail

module purge
module load ALICE/default
module load Miniforge3

cd "$SLURM_SUBMIT_DIR"

export CWD=$(pwd)
export DATE=$(date)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "[$SHELL] #### Starting impact-scan array"
echo "[$SHELL] ## USER: $SLURM_JOB_USER | ID: $SLURM_JOB_ID"
echo "[$SHELL] ## current working directory: $CWD"
echo "[$SHELL] ## cases: $SLURM_JOB_NUM_NODES node(s), $SLURM_NTASKS task(s)"

CASE_SEED="${CASE_SEED:-42}"
CASE_DIR="data/run_cases/cluster_${SLURM_JOB_ID}"
PLOT="media/impact_scan_cluster_${SLURM_JOB_ID}.png"

mkdir -p logs media
export CASE_DIR

CASE_DIR="$CASE_DIR" \
	N_CASES=120 \
	N_REPEATS=3 \
	N_JOBS="$SLURM_NTASKS" \
	N_PARTICLES=10000 \
	POST_IMPACT_TAIL=18000 \
	DT_HIT=1 \
	PLOT_OUT="$PLOT" \
	JSON_OUT="data/impact_scan_cluster_${SLURM_JOB_ID}.json" \
	./run_cases.sh "$CASE_SEED"

echo "[$SHELL] ## impact scan plot: $PLOT"
echo "[$SHELL] ## case rows: $CASE_DIR"
echo "[$SHELL] ## Job done $DATE"

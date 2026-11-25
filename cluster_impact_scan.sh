#!/bin/bash
#SBATCH --job-name=impact_scan
#SBATCH --export=ALL
#SBATCH --output=logs/impact_scan_%j.out
#SBATCH --error=logs/impact_scan_%j.err
#SBATCH --mail-user="yangtaotao0524@gmail.com"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --time=24:00:00
#SBATCH --partition=cpu-zen4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G

# Production impact scan:
#   - 401 impact parameters, -2.00..2.00 step 0.01
#   - 8 realisations per impact parameter = 3208 cases
#   - 1,000 particles per resolved body
#   - direct-b selection from the experiment's SCAN_DIR
#   - no media gen; one case JSON row per case, no per-run manifest
#   - one shared batch log file
#   - merged into media/impact_scan_cluster_<jobid>.png
#   - one surface-density profile at b=-1.00
#   - TAG selects the experiment; SCAN_DIR and DENSITY must match that scan
#
# Submit one job per experiment, for example:
#   TAG=paper_rocky SCAN_DIR=data/param_scan_paper_rocky DENSITY=3000 \
#     sbatch --export=ALL --job-name=impact_paper_rocky cluster_impact_scan.sh

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
TAG="${TAG:-paper_icy}"
CASE_DIR="data/run_cases/${TAG}_${SLURM_JOB_ID}"
PLOT="media/impact_scan_${TAG}_${SLURM_JOB_ID}.png"
SURFACE_OUT="media/surface_density_${TAG}_${SLURM_JOB_ID}.png"
LOG_FILE="logs/run_cases/${TAG}_${SLURM_JOB_ID}.log"
IMPACT_TARGETS="${IMPACT_TARGETS:-$(seq -f '%.2f' -2 0.01 2)}"
SURFACE_DENSITY_TARGET="${SURFACE_DENSITY_TARGET:--1.00}"
SURFACE_DENSITY_OUT="${SURFACE_DENSITY_OUT:-$SURFACE_OUT}"

mkdir -p logs media
export CASE_DIR LOG_FILE IMPACT_TARGETS

CASE_DIR="$CASE_DIR" \
	SCAN_DIR="${SCAN_DIR:-data/param_scan_1m}" \
	DENSITY="${DENSITY:-500}" \
	N_CASES="${N_CASES:-3208}" \
	N_REPEATS="${N_REPEATS:-8}" \
	N_JOBS="$SLURM_NTASKS" \
	N_PARTICLES="${N_PARTICLES:-1000}" \
	POST_IMPACT_TAIL="${POST_IMPACT_TAIL:-10000}" \
	DT_HIT=1 \
	IMPACT_TARGETS="$IMPACT_TARGETS" \
	SURFACE_DENSITY_TARGET="$SURFACE_DENSITY_TARGET" \
	SURFACE_DENSITY_OUT="$SURFACE_DENSITY_OUT" \
	LOG_FILE="$LOG_FILE" \
	PLOT_OUT="$PLOT" \
	JSON_OUT="data/impact_scan_${TAG}_${SLURM_JOB_ID}.json" \
	./run_cases.sh "$CASE_SEED"

echo "[$SHELL] ## impact scan plot: $PLOT"
echo "[$SHELL] ## surface density plot: $SURFACE_DENSITY_OUT"
echo "[$SHELL] ## batch log: $LOG_FILE"
echo "[$SHELL] ## case rows: $CASE_DIR"
echo "[$SHELL] ## Job done $DATE"

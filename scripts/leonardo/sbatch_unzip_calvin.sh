#!/bin/bash
# ==============================================================================
# Unzip + preprocess CALVIN dataset as a SLURM job, not on the login node.
#
# Login nodes throttle/kill long CPU-bound processes; this moves the 1–3 hour
# unzip + the rel_actions preprocess onto a compute node where it can run
# uninterrupted.
#
# Usage:
#   sbatch scripts/leonardo/sbatch_unzip_calvin.sh           # task_D_D (default)
#   CALVIN_TASK=task_ABC_D sbatch scripts/leonardo/sbatch_unzip_calvin.sh
#
# Pre-req: the .zip is already downloaded at $LEONARDO_WORK/data/calvin/<TASK>.zip
# ==============================================================================

#SBATCH --job-name=unzip-calvin
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --account=AIFAC_F02_024

set -euo pipefail

FAST="${LEONARDO_FAST:-${FAST:?Set LEONARDO_FAST or FAST}}"
WORK="${LEONARDO_WORK:-${WORK:?Set LEONARDO_WORK or WORK}}"

TASK="${CALVIN_TASK:-task_D_D}"
CALVIN_DIR="$WORK/data/calvin"
CODE_DIR="$FAST/project/flower_vla_calvin"
VENV_DIR="$WORK/venvs/flower_vla_calvin"

ZIP="$CALVIN_DIR/${TASK}.zip"
[[ -f "$ZIP" ]] || { echo "ERROR: $ZIP not found. Download it first."; exit 1; }

echo "============================================"
echo "CALVIN unzip + preprocess job"
echo "============================================"
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $(hostname)"
echo "Task:     $TASK"
echo "Zip:      $ZIP ($(du -h "$ZIP" | awk '{print $1}'))"
echo "Dest:     $CALVIN_DIR/$TASK"
echo "Start:    $(date)"
echo "============================================"

cd "$CALVIN_DIR"

# NOTE: if a previous unzip was interrupted, the partial $CALVIN_DIR/$TASK dir
# is incomplete — delete it before running this job:
#   rm -rf $LEONARDO_WORK/data/calvin/$TASK
# (Or set FORCE_REEXTRACT=1 below to handle this automatically.)
if [[ -d "$CALVIN_DIR/$TASK" && "${FORCE_REEXTRACT:-0}" != "1" ]]; then
    echo "WARN: $CALVIN_DIR/$TASK already exists — skipping unzip."
    echo "      If this is a partial extraction, kill this job, rm -rf the dir,"
    echo "      and resubmit (or set FORCE_REEXTRACT=1)."
else
    [[ "${FORCE_REEXTRACT:-0}" == "1" ]] && rm -rf "$CALVIN_DIR/$TASK"
    echo "Unzipping $ZIP ..."
    unzip -q "$ZIP"
    echo "Unzip done at $(date)"
fi

echo ""
echo "Removing zip to free space..."
rm -f "$ZIP"

# Preprocess: extract rel_actions
echo ""
echo "Preprocessing CALVIN data (extracting rel_actions)..."
module purge
module load profile/deeplrn
module load cuda/12.1
source "$VENV_DIR/bin/activate"

python "$CODE_DIR/preprocess/extract_by_key.py" \
    -i "$CALVIN_DIR" \
    --in_task "$TASK" \
    --in_split all \
    -k rel_actions

echo ""
echo "=== CALVIN $TASK ready at $CALVIN_DIR/$TASK ==="
echo "End: $(date)"

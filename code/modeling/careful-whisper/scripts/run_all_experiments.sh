#!/bin/bash
#SBATCH --job-name=launch-experiments
#SBATCH --partition=standard
#SBATCH --account=dbic
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=submit_scripts/logs/launch-experiments-%j.txt
#
# Launcher script — generates DSQ joblists and submits them.
# Does not need GPU; runs on a CPU node.
#
# Full pipeline for avspeech, lrs3, and voxceleb2.
#
# Stages:
#   stage1    Token fusion (submit first)
#   stage2    Non-audiovisual models + controls (can run in parallel with stage1)
#   stage2b   Audiovisual models + controls (requires stage1 done + utils.py updated)
#   stage3    Voxceleb2 subsets (non-AV; can run in parallel with stage1)
#
# Usage (interactive):
#   bash run_all_experiments.sh stage1
#   bash run_all_experiments.sh stage2
#
# Usage (submit as SLURM job):
#   sbatch run_all_experiments.sh stage1
#   sbatch run_all_experiments.sh stage2

set -e

module load cuda/12.3
module load openmpi
source /optnfs/common/miniconda3/etc/profile.d/conda.sh
conda activate prosody

# SLURM copies the script to its spool dir, so BASH_SOURCE[0] is unreliable.
# Use SLURM_SUBMIT_DIR when running as a job, fall back to BASH_SOURCE for interactive use.
SCRIPTS_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
DSQ_DIR="$SCRIPTS_DIR/submit_scripts/dsq"
cd "$SCRIPTS_DIR"

DATASETS=(avspeech lrs3 voxceleb2)

# Models that don't depend on token fusion — can start immediately
NON_AV_MODELS=(text audio prosody audio-control prosody-control)

# Models that require a token fusion checkpoint — run after stage1 completes
AV_MODELS=(audiovisual audiovisual-control)

# Subset models (non-AV only; voxceleb2 only)
SUBSET_MODELS=(text audio prosody audiovisual)

JOBLIST_DIR="$SCRIPTS_DIR/submit_scripts/joblists"
LOGS_DIR="$SCRIPTS_DIR/submit_scripts/logs"

# Shared SLURM settings for GPU jobs
DSQ_SLURM_ARGS="--partition=v100_preemptable --account=dbic --time=3-00:00:00
    --nodes=1 --ntasks=1 --ntasks-per-node=1 --gres=gpu:1
    --cpus-per-task=16 --mem-per-cpu=8G --exclude=p04"

# Helper: generate joblists, combine, call DSQ, and sbatch
# Usage: submit_stage <stage_name> <joblist1> [<joblist2> ...]
submit_stage() {
    local stage_name="$1"; shift
    local combined="$JOBLIST_DIR/${stage_name}_careful_whisper_experiments.txt"
    local dsq_name="dsq_${stage_name}_careful_whisper_experiments"
    local dsq_batch="$DSQ_DIR/${dsq_name}"
    local dsq_out="$LOGS_DIR/${dsq_name}"
    local job_num=0

    # Concatenate all provided joblists into one
    > "$combined"
    for jl in "$@"; do
        cat "$jl" >> "$combined"
        job_num=$((job_num + $(wc -l < "$jl")))
    done

    local width=${#job_num}
    mkdir -p "$dsq_out"

    dsq --job-file "$combined" --batch-file "${dsq_batch}.sh" \
        --status-dir "$dsq_out" \
        --output="${dsq_out}/${dsq_name}-%A_%${width}a-%N.txt" \
        $DSQ_SLURM_ARGS

    sbatch "${dsq_batch}.sh"
    echo "Submitted: ${dsq_name} ($job_num jobs) — monitor with: dsqa -j <job_id>"
}

# ============================================================
# STAGE 1: Token fusion
# ============================================================
stage1() {
    echo "=== STAGE 1: Token fusion ==="
    for DATASET in "${DATASETS[@]}"; do
        echo "--- Generating: $DATASET ---"
        python batch_run_token_fusion.py -d "$DATASET"
        sbatch "$DSQ_DIR/dsq_${DATASET}_token-fusion.sh"
    done
    echo ""
    echo "Token fusion jobs submitted."
    echo "Once complete:"
    echo "  1. Check the best checkpoint epoch for each dataset in the training logs"
    echo "  2. Update DATASET_CONFIG in scripts/utils.py with the correct epoch filenames"
    echo "  3. Run: sbatch run_all_experiments.sh stage2b"
}

# ============================================================
# STAGE 2: Non-audiovisual models + controls
# (text, audio, prosody, audio-control, prosody-control)
# Safe to run in parallel with stage1.
# ============================================================
stage2() {
    echo "=== STAGE 2: Non-audiovisual models + controls ==="
    local joblists=()
    for DATASET in "${DATASETS[@]}"; do
        for MODEL in "${NON_AV_MODELS[@]}"; do
            python batch_careful-whisper_experiments.py -d "$DATASET" -m "$MODEL"
            joblists+=("$JOBLIST_DIR/${DATASET}-${MODEL}_careful_whisper_experiments.txt")
        done
    done
    submit_stage "stage2" "${joblists[@]}"
}

# ============================================================
# STAGE 2b: Audiovisual models + controls
# PREREQUISITE: stage1 complete + utils.py updated with new checkpoint epochs
# ============================================================
stage2b() {
    echo "=== STAGE 2b: Audiovisual models + controls ==="
    local joblists=()
    for DATASET in "${DATASETS[@]}"; do
        for MODEL in "${AV_MODELS[@]}"; do
            python batch_careful-whisper_experiments.py -d "$DATASET" -m "$MODEL"
            joblists+=("$JOBLIST_DIR/${DATASET}-${MODEL}_careful_whisper_experiments.txt")
        done
    done
    submit_stage "stage2b" "${joblists[@]}"
}

# ============================================================
# STAGE 3: Voxceleb2 subsets (non-AV)
# Safe to run in parallel with stage1.
# ============================================================
stage3() {
    echo "=== STAGE 3: Voxceleb2 subsets ==="
    local joblists=()
    for MODEL in "${SUBSET_MODELS[@]}"; do
        python batch_careful-whisper_experiments.py -d voxceleb2 -m "$MODEL" --subsets 1
        joblists+=("$JOBLIST_DIR/voxceleb2-${MODEL}_careful_whisper_experiments.txt")
    done
    submit_stage "stage3" "${joblists[@]}"
}

# ============================================================
# Dispatch
# ============================================================
case "${1:-}" in
    stage1)  stage1  ;;
    stage2)  stage2  ;;
    stage2b) stage2b ;;
    stage3)  stage3  ;;
    *)
        echo "Usage: {bash|sbatch} run_all_experiments.sh {stage1|stage2|stage2b|stage3}"
        echo ""
        echo "  stage1    Token fusion (avspeech, lrs3, voxceleb2)"
        echo "  stage2    Non-AV models + controls  [parallel with stage1]"
        echo "  stage2b   Audiovisual models + controls  [after stage1 + utils.py update]"
        echo "  stage3    Voxceleb2 subsets (non-AV)  [parallel with stage1]"
        exit 1
        ;;
esac

import os, sys

## primary directories
BASE_DIR = '/dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/code/modeling/careful-whisper'
DATASETS_DIR = '/dartfs/rc/lab/F/FinnLab/datasets/'

## secondary directories
SUBMIT_DIR = os.path.join(BASE_DIR, 'scripts/submit_scripts')
LOGS_DIR = os.path.join(SUBMIT_DIR, 'logs')
SCRATCH_DIR = '/dartfs-hpc/scratch/f003rjw'

# Strings to append at start of run for dSQ
DSQ_MODULES	 = [
    'module load cuda/12.3',
    'module load openmpi',
    'source /optnfs/common/miniconda3/etc/profile.d/conda.sh',
    'conda activate prosody'
]

EXTRA_COMMANDS = [
    # # Send some noteworthy information to the output log
    'echo "Running on node: $(hostname)"',
    'echo "In directory:    $(pwd)"',
    'echo "Starting on:     $(date)"',
    'echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"',
    'echo "Cuda devices: " ${CUDA_VISIBLE_DEVICES}',
]

DSQ_MODULES += EXTRA_COMMANDS

DSQ_MODULES = ''.join([f'{module}; ' for module in DSQ_MODULES])
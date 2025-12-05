#!/bin/bash

# Run within BIDS code/ directory:
# sbatch slurm_download_avspeech.sh

# Set partition
#SBATCH --partition=standard

# How long is job (in minutes)?
#SBATCH --time=3-00:00:00

# How much memory to allocate (in MB)?
#SBATCH --nodes=1 --ntasks-per-node=1 --ntasks=1 --cpus-per-task=8 --mem-per-cpu=4G

# Name of jobs?
#SBATCH --job-name=download_avspeech

# Where to output log files?
#SBATCH --output='./DownloadAVSpeech-%A.txt'

source /optnfs/common/miniconda3/etc/profile.d/conda.sh
conda activate prosody

python download_avspeech.py
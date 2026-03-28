#!/bin/bash
#SBATCH --output /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/code/modeling/careful-whisper/scripts/submit_scripts/logs/dsq_token-fusion/dsq_token-fusion-%A_%1a-%N.txt
#SBATCH --array 0
#SBATCH --job-name dsq-voxceleb2_token-fusion_experiments
#SBATCH --partition=v100_preemptable --time=2-12:00:00 --account=dbic --nodes=1 --gres=gpu:1 --ntasks-per-node=1 --ntasks=1 --exclude= --cpus-per-task=16 --mem-per-cpu=8G

# DO NOT EDIT LINE BELOW
/optnfs/common/dSQ/dSQ-1.05/dSQBatch.py --job-file /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/code/modeling/careful-whisper/scripts/submit_scripts/joblists/voxceleb2_token-fusion_experiments.txt --status-dir /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/code/modeling/careful-whisper/scripts/submit_scripts/logs/dsq_token-fusion


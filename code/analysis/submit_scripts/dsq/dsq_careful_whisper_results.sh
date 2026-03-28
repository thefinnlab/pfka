#!/bin/bash
#SBATCH --output /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/derivatives/logs/careful-whisper/results/dsq_careful_whisper_results/dsq_careful_whisper_results-%A_%1a-%N.txt
#SBATCH --array 0-6
#SBATCH --job-name dsq_careful_whisper_results
#SBATCH --partition=v100_preemptable --time=1-00:00:00 --account=dbic --nodes=1 --gres=gpu:1 --ntasks-per-node=1 --ntasks=1 --exclude= --cpus-per-task=16 --mem-per-cpu=8G

# DO NOT EDIT LINE BELOW
/optnfs/common/dSQ/dSQ-1.05/dSQBatch.py --job-file /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/code/analysis/submit_scripts/joblists/careful_whisper_results.txt --status-dir /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/derivatives/logs/careful-whisper/results/dsq_careful_whisper_results

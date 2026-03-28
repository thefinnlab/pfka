#!/bin/bash
#SBATCH --output submit_scripts/logs/dsq_lrs3-text-a100_careful_whisper_experiments/dsq_lrs3-text-a100_careful_whisper_experiments-%A_%1a-%N.txt
#SBATCH --array 0
#SBATCH --job-name dsq-lrs3-text-a100_careful_whisper_experiments
#SBATCH --partition=a100_preemptable --account=dbic --time=1-00:00:00 --nodes=1 --ntasks=1 --ntasks-per-node=1 --gres=gpu:1 --cpus-per-task=16 --mem-per-cpu=8G

# DO NOT EDIT LINE BELOW
/optnfs/common/dSQ/dSQ-1.05/dSQBatch.py --job-file /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/code/modeling/careful-whisper/scripts/submit_scripts/joblists/lrs3-text-a100_careful_whisper_experiments.txt --status-dir /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/code/modeling/careful-whisper/scripts/submit_scripts/logs/dsq_lrs3-text-a100_careful_whisper_experiments


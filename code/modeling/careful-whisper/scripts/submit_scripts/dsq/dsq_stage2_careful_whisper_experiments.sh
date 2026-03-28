#!/bin/bash
#SBATCH --output /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/code/modeling/careful-whisper/scripts/submit_scripts/logs/dsq_stage2_careful_whisper_experiments/dsq_stage2_careful_whisper_experiments-%A_%2a-%N.txt
#SBATCH --array 0-14
#SBATCH --job-name dsq-stage2_careful_whisper_experiments
#SBATCH --partition=v100_preemptable --account=dbic --time=3-00:00:00 --nodes=1 --ntasks=1 --ntasks-per-node=1 --gres=gpu:1 --cpus-per-task=16 --mem-per-cpu=8G --exclude=p04

# DO NOT EDIT LINE BELOW
/optnfs/common/dSQ/dSQ-1.05/dSQBatch.py --job-file /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/code/modeling/careful-whisper/scripts/submit_scripts/joblists/stage2_careful_whisper_experiments.txt --status-dir /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/code/modeling/careful-whisper/scripts/submit_scripts/logs/dsq_stage2_careful_whisper_experiments


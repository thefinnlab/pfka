#!/bin/bash
#SBATCH --output submit_scripts/logs/dsq_lrs3-text-gv01_careful_whisper_experiments/dsq_lrs3-text-gv01-%A_%1a-%N.txt
#SBATCH --array 0
#SBATCH --job-name dsq-lrs3-text-gv01_careful_whisper_experiments
#SBATCH --partition=v100_preemptable --account=dbic --time=3-00:00:00 --nodes=1 --ntasks=1 --ntasks-per-node=1 --gres=gpu:1 --cpus-per-task=16 --mem-per-cpu=8G --nodelist=gv01

# DO NOT EDIT LINE BELOW
/optnfs/common/dSQ/dSQ-1.05/dSQBatch.py --job-file /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/code/modeling/careful-whisper/scripts/submit_scripts/joblists/lrs3-text-gv01_careful_whisper_experiments.txt --status-dir /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/code/modeling/careful-whisper/scripts/submit_scripts/logs/dsq_lrs3-text-gv01_careful_whisper_experiments


#!/bin/bash
#SBATCH --output /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/code/modeling/careful-whisper/scripts/submit_scripts/logs/dsq_voxceleb2-prosody-control_careful_whisper_experiments/dsq_voxceleb2-prosody-control_careful_whisper_experiments-%A_%1a-%N.txt
#SBATCH --array 0
#SBATCH --job-name dsq-voxceleb2-prosody-control_careful_whisper_experiments
#SBATCH --partition=v100_preemptable --time=3-00:00:00 --account=dbic --nodes=1 --gres=gpu:1 --ntasks-per-node=1 --ntasks=1 --exclude=p04 --cpus-per-task=16 --mem-per-cpu=8G

# DO NOT EDIT LINE BELOW
/optnfs/common/dSQ/dSQ-1.05/dSQBatch.py --job-file /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/code/modeling/careful-whisper/scripts/submit_scripts/joblists/voxceleb2-prosody-control_careful_whisper_experiments.txt --status-dir /dartfs/rc/lab/F/FinnLab/tommy/isc_asynchrony_behavior/code/modeling/careful-whisper/scripts/submit_scripts/logs/dsq_voxceleb2-prosody-control_careful_whisper_experiments


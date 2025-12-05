import sys, os
import argparse
import subprocess
from pathlib import Path
import glob

sys.path.append('../../../utils/')

from config import *
from dataset_utils import attempt_makedirs

sys.path.append('../utils/')

import utils 

PARTITION = 'preemptable'
TIME = '8:00:00'
N_NODES = 1
N_TASKS_PER_NODE = 1
N_TASKS = 1
CPUS_PER_TASK = 31
MEM_PER_CPU = '8G'
GPU_INFO = ''

EXCLUDE = ''
ACCOUNT = 'dbic'

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str)
    parser.add_argument('-o', '--overwrite', type=int, default=0)
    p = parser.parse_args()

    logs_dir = os.path.join(BASE_DIR, 'derivatives/logs/modeling/')
    dsq_dir =  os.path.join(BASE_DIR, 'code/submit_scripts/modeling/dsq')
    joblists_dir = os.path.join(BASE_DIR, 'code/submit_scripts/modeling/joblists')

    attempt_makedirs(logs_dir)
    attempt_makedirs(dsq_dir)
    attempt_makedirs(joblists_dir)

    all_cmds = []
    job_num = 0

    dataset_config = utils.DATASET_CONFIGS[p.dataset]
    output_dir = os.path.join(DATASETS_DIR, 'nlp-datasets', p.dataset)

    if p.dataset in ['lrs3', 'voxceleb2', 'avspeech']:
        models = utils.DATASET_TYPES['video']
    else:
        models = utils.DATASET_TYPES['audio']

    model_types = ' '.join([f"--{k} {v} " for k, v in models.items()])

    for split in dataset_config['splits']:
        
        print(f'Making job for: {p.dataset} {split}', flush=True)

        cmd = [
            f"{DSQ_MODULES.replace('dark_matter', 'prosody')} ",
            f"python compile_metadata.py --dataset {p.dataset} --output_dir {output_dir} --split {split} "
            f"{model_types} --overwrite {p.overwrite}; ", 
        ]

        cmd = "".join(cmd)
        all_cmds.append(cmd)
        job_num += 1
    # break

    if not all_cmds:
        print(f'No matching audio and text files found', flush=True)
        sys.exit(0)

    joblist_fn = os.path.join(joblists_dir, f'dsq_{p.dataset}_compile_metadata.txt')

    with open(joblist_fn, 'w') as f:
        for cmd in all_cmds:
            f.write(f"{cmd}\n")

    dsq_base_string = f'{p.dataset}_compile_metadata'
    dsq_batch_fn = os.path.join(dsq_dir, dsq_base_string)
    dsq_out_dir = os.path.join(logs_dir, dsq_base_string)
    array_fmt_width = len(str(job_num))

    if not os.path.exists(dsq_out_dir):
        os.makedirs(dsq_out_dir)

    subprocess.run(f"dsq --job-file {joblist_fn} --batch-file {dsq_batch_fn}.sh "
        f"--status-dir {dsq_out_dir} --partition={PARTITION} --output={dsq_out_dir}/{dsq_base_string}-%A_%{array_fmt_width}a-%N.txt "
        f"--time={TIME} --nodes={N_NODES} {GPU_INFO} --account={ACCOUNT} --ntasks-per-node={N_TASKS_PER_NODE} --ntasks={N_TASKS} "
        f"--cpus-per-task={CPUS_PER_TASK} --mem-per-cpu={MEM_PER_CPU} --exclude={EXCLUDE}", shell=True)

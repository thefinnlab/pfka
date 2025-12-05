import sys, os
import argparse
import subprocess
from pathlib import Path
import glob

sys.path.append('../../../../utils/')

from config import *
from dataset_utils import attempt_makedirs

sys.path.append('../../')

import utils 

PARTITION = 'preemptable'
TIME = '5-00:00:00'
N_NODES = 1
N_TASKS_PER_NODE = 1
N_TASKS = 1
CPUS_PER_TASK = 8
MEM_PER_CPU = '8G'

TIME = '2-12:00:00'
CPUS_PER_TASK = 8
MEM_PER_CPU = '8G'
PARTITION = 'gpuq'
GPU_INFO = '--gres=gpu:1'
NODE_LIST = ''#--nodelist=a03,a04'
EXCLUDE = ''
ACCOUNT = 'dbic'

if __name__ == "__main__":

    NUM_SHARDS = 10

    # Set up dataset paths
    base_dir = os.path.join(DATASETS_DIR, 'nlp-datasets/voxceleb2/')

    logs_dir = os.path.join(BASE_DIR, 'derivatives/logs/modeling/')
    dsq_dir =  os.path.join(BASE_DIR, 'code/submit_scripts/modeling/dsq')
    joblists_dir = os.path.join(BASE_DIR, 'code/submit_scripts/modeling/joblists')

    attempt_makedirs(logs_dir)
    attempt_makedirs(dsq_dir)
    attempt_makedirs(joblists_dir)

    all_cmds = []
    job_num = 0

    for shard in range(NUM_SHARDS):
        cmd = [
            f"{DSQ_MODULES.replace('dark_matter', 'prosody')} ",
            f"python classify_voxceleb2.py --base_dir {base_dir} --num_shards {NUM_SHARDS} --current_shard {shard}; ",
        ]

        cmd = "".join(cmd)
        all_cmds.append(cmd)
        job_num += 1

    if not all_cmds:
        print(f'No splits for voxceleb2', flush=True)
        sys.exit(0)

    joblist_fn = os.path.join(joblists_dir, f'voxceleb2_classify_language.txt')

    with open(joblist_fn, 'w') as f:
        for cmd in all_cmds:
            f.write(f"{cmd}\n")

    dsq_base_string = f'voxceleb2_classify_language'
    dsq_batch_fn = os.path.join(dsq_dir, dsq_base_string)
    dsq_out_dir = os.path.join(logs_dir, dsq_base_string)
    array_fmt_width = len(str(job_num))

    if not os.path.exists(dsq_out_dir):
        os.makedirs(dsq_out_dir)

    subprocess.run(f"dsq --job-file {joblist_fn} --batch-file {dsq_batch_fn}.sh "
        f"--status-dir {dsq_out_dir} --partition={PARTITION} --output={dsq_out_dir}/{dsq_base_string}-%A_%{array_fmt_width}a-%N.txt "
        f"--time={TIME} --nodes={N_NODES} {GPU_INFO} --account={ACCOUNT} --ntasks-per-node={N_TASKS_PER_NODE} --ntasks={N_TASKS} "
        f"--cpus-per-task={CPUS_PER_TASK} --mem-per-cpu={MEM_PER_CPU} --exclude={EXCLUDE}", shell=True)

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
TIME = '5-00:00:00'
N_NODES = 1
N_TASKS_PER_NODE = 1
N_TASKS = 1
CPUS_PER_TASK = 4
MEM_PER_CPU = '8G'
GPU_INFO = ''

TIME = '2-12:00:00'
CPUS_PER_TASK = 8
MEM_PER_CPU = '8G'
PARTITION = 'v100_preemptable'
GPU_INFO = '--gres=gpu:1'
NODE_LIST = ''#--nodelist=a03,a04'
EXCLUDE = ''
ACCOUNT = 'dbic'

if __name__ == "__main__":

    # DATASETS = ['voxceleb2'] #'lrs3'
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str)
    parser.add_argument('-o', '--overwrite', type=int, default=0)
    p = parser.parse_args()

    video_model = 'data2vec'

    logs_dir = os.path.join(BASE_DIR, 'derivatives/logs/modeling/')
    dsq_dir =  os.path.join(BASE_DIR, 'code/submit_scripts/modeling/dsq')
    joblists_dir = os.path.join(BASE_DIR, 'code/submit_scripts/modeling/joblists')

    attempt_makedirs(logs_dir)
    attempt_makedirs(dsq_dir)
    attempt_makedirs(joblists_dir)

    all_cmds = []
    script_fn = os.path.join(os.getcwd(), 'extract_dataset_features.py')
    job_num = 0

    counter = 0

    # for dataset in DATASETS:

    dataset_config = utils.DATASET_CONFIGS[p.dataset]
    splits = dataset_config['splits']
    output_dir = os.path.join(DATASETS_DIR, 'nlp-datasets', p.dataset)

    if p.dataset in ['lrs3', 'voxceleb2', 'avspeech']:
        models = utils.DATASET_TYPES['video']
        models['video_model'] = video_model
    else:
        models = utils.DATASET_TYPES['audio']

    model_types = ' '.join([f"--{k} {v} " for k, v in models.items()])

    for split in splits[::-1]:

        # Number of subdatasets for efficient processing
        if split == 'train':
            N_SHARDS = 50
        else:
            N_SHARDS = 10
        # else:
            # N_SHARDS = 1

        if split != 'train':
            continue
        
        for shard in range(N_SHARDS):

            if counter < 35:
                counter += 1
                continue

            counter += 1
            # if shard not in [3]:
            #   continue

            print(f'Making job for: {p.dataset} {split}, {shard+1}/{N_SHARDS} shards', flush=True)

            cmd = [
                f"{DSQ_MODULES.replace('dark_matter', 'prosody')} ",
                f"python extract_dataset_features.py --dataset {p.dataset} --output_dir {output_dir} --split {split} "
                f"{model_types} --num_shards {N_SHARDS} --current_shard {shard} --overwrite {p.overwrite}; ", 
            ]

            cmd = "".join(cmd)
            all_cmds.append(cmd)
            job_num += 1
        #     break
        # break
    # break

    if not all_cmds:
        print(f'No matching audio and text files found', flush=True)
        sys.exit(0)

    joblist_fn = os.path.join(joblists_dir, f'{p.dataset}_extract_dataset_features.txt')

    with open(joblist_fn, 'w') as f:
        for cmd in all_cmds:
            f.write(f"{cmd}\n")

    dsq_base_string = f'{p.dataset}_extract_dataset_features'
    dsq_batch_fn = os.path.join(dsq_dir, dsq_base_string)
    dsq_out_dir = os.path.join(logs_dir, dsq_base_string)
    array_fmt_width = len(str(job_num))

    if not os.path.exists(dsq_out_dir):
        os.makedirs(dsq_out_dir)

    subprocess.run(f"dsq --job-file {joblist_fn} --batch-file {dsq_batch_fn}.sh "
        f"--status-dir {dsq_out_dir} --partition={PARTITION} --output={dsq_out_dir}/{dsq_base_string}-%A_%{array_fmt_width}a-%N.txt "
        f"--time={TIME} --nodes={N_NODES} {GPU_INFO} --account={ACCOUNT} --ntasks-per-node={N_TASKS_PER_NODE} --ntasks={N_TASKS} "
        f"--cpus-per-task={CPUS_PER_TASK} --mem-per-cpu={MEM_PER_CPU} --exclude={EXCLUDE}", shell=True)

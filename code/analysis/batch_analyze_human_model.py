import os, sys, glob
import json
import numpy as np
import pandas as pd
import argparse
from itertools import product
import subprocess

sys.path.append('../utils/')

from config import *
import dataset_utils as utils
from tommy_utils.nlp import MLM_MODELS_DICT, CLM_MODELS_DICT

PARTITION = 'standard'
TIME = '12:00:00'
N_NODES = 1
N_TASKS_PER_NODE = 1
N_TASKS = 1
CPUS_PER_TASK = 8
MEM_PER_CPU = '32G'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # type of analysis we're running --> linked to the name of the regressors
    parser.add_argument('-t', '--task', type=str, nargs='+', default=['black', 'wheretheressmoke', 'howtodraw'])
    parser.add_argument('-careful_whisper', '--careful_whisper', type=int, default=0)
    parser.add_argument('-drop_shortened', '--drop_shortened', type=int, default=0)
    parser.add_argument('-o', '--overwrite', type=int, default=0)
    p = parser.parse_args()

    # model_names = sorted(CLM_MODELS_DICT.keys())
    window_sizes = [25]
    # window_sizes = [
    #     2, 3, 4, 5, 10, 50, 75, 125, 150, 175, 200, 225, 250, 275, 300
    # ]

    # failed_jobs = [
    #     0, 3, 4, 5, 6, 11, 14, 16, 17
    # ]

    # get all MLM models except BERT
    if p.careful_whisper:
        models = sorted(glob.glob(os.path.join(BASE_DIR, f'derivatives/model-predictions/{p.task[0]}/careful-whisper/*')))
        models = [os.path.basename(model) for model in models]
        model_names = ' '.join(models)

        print (f'Loading the following models')
        print (f'Careful Whisper models: {models}')
    else:
        MLM_MODELS = list(MLM_MODELS_DICT.keys())[1:]
        CLM_MODELS = list(CLM_MODELS_DICT.keys()) 
        model_names = CLM_MODELS + MLM_MODELS
        # model_names = ' '.join(model_names)

        remove_models = ['qwen3-8B', 'llama3.1-8B']
        # remove_models = ['qwen3-32B', 'qwen3-8B', 'llama3.1-8B', 'llama3.1-70B', 'gemma3-1b-pt']
        model_names = [model for model in model_names if model not in remove_models]
        model_names = ' '.join(model_names)

        print (f'Loading the following models')
        print (f'MLM models: {MLM_MODELS}')
        print (f'CLM models: {CLM_MODELS}')

    # grab the tasks
    all_cmds = []
    script_fn = os.path.join(os.getcwd(), 'run_analyze_human_model.py')
    job_string = f'{DSQ_MODULES} srun python {script_fn}'
    job_num = 0

    for i, (task, window_size) in enumerate(product(p.task, window_sizes)):

        # if i not in failed_jobs:
        #     continue

        cmd = ''.join([
            f"{job_string} -t {task} -m {model_names} -careful_whisper {p.careful_whisper} -window_size {window_size} -drop_shortened {p.drop_shortened} -o {p.overwrite}"
        ])

        all_cmds.append(cmd)
        job_num += 1

    dsq_base_string = f'dsq_analyze_human_model'
    logs_dir = os.path.join(BASE_DIR, 'derivatives/logs/behavioral/')
    dsq_dir =  os.path.join(BASE_DIR, 'code/submit_scripts/behavioral/dsq')
    joblists_dir = os.path.join(BASE_DIR, 'code/submit_scripts/behavioral/joblists')

    utils.attempt_makedirs(logs_dir)
    utils.attempt_makedirs(dsq_dir)
    utils.attempt_makedirs(joblists_dir)

    joblist_fn = os.path.join(joblists_dir, f'run_analyze_human_model.txt')

    with open(joblist_fn, 'w') as f:
        for cmd in all_cmds:
            f.write(f"{cmd}\n")
    
    dsq_batch_fn = os.path.join(dsq_dir, dsq_base_string)
    dsq_out_dir = os.path.join(logs_dir, dsq_base_string)
    array_fmt_width = len(str(job_num))
    
    if not os.path.exists(dsq_out_dir):
        os.makedirs(dsq_out_dir)
    
    # subprocess.run('module load dSQ', shell=True)
    subprocess.run(f"dsq --job-file {joblist_fn} --batch-file {dsq_batch_fn}.sh "
        f"--status-dir {dsq_out_dir} --partition={PARTITION} --output={dsq_out_dir}/{dsq_base_string}-%A_%{array_fmt_width}a-%N.txt "
        f"--time={TIME} --nodes={N_NODES} --ntasks-per-node={N_TASKS_PER_NODE} --ntasks={N_TASKS} "
        f"--cpus-per-task={CPUS_PER_TASK} --mem-per-cpu={MEM_PER_CPU}", shell=True)
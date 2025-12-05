import os, sys, glob
import json
import numpy as np
import pandas as pd
import argparse
from itertools import product
import subprocess

sys.path.append('../../utils/')

from config import *
import dataset_utils as utils
from tommy_utils import nlp
# import nlp_utils as nlp

PARTITION = 'standard'
TIME = '14-00:00:00'
N_NODES = 1
N_TASKS_PER_NODE = 1
N_TASKS = 1
CPUS_PER_TASK = 12
MEM_PER_CPU = '32G'

GPU_INFO = ''
NODE_LIST = ''#--nodelist=a03,a04'
EXCLUDE = ''
ACCOUNT = 'dbic'

if __name__ == '__main__':

	OVERWRITE = True

	# grab the tasks
	task_dirs = sorted(glob.glob(os.path.join(BASE_DIR, 'stimuli/preprocessed', '*')))
	task_names = [os.path.basename(d) for d in task_dirs if os.path.isdir(d)]

	# remove practice trial and example trial from the list of tasks
	task_names = [task for task in task_names if task not in ['nwp_practice_trial', 'example_trial']] 

	task_names = ['black', 'wheretheressmoke', 'howtodraw'] #'black'] #['demon'] #, 'keats']

	# get all MLM models except BERT
	MLM_MODELS = list(nlp.MLM_MODELS_DICT.keys())[1:]
	CLM_MODELS = list(nlp.CLM_MODELS_DICT.keys()) 
	ALL_MODELS = list(nlp.CLM_MODELS_DICT.keys()) + list(nlp.MLM_MODELS_DICT.keys())

	# model_names = CLM_MODELS + MLM_MODELS
	model_names = [
		# 'qwen3-8B',
		'qwen3-32B',
		# 'llama3.1-8B',
		'llama3.1-70B',
		'gemma3-1b-pt'
	]

	assert all([model in ALL_MODELS for model in model_names])

	# model_names = sorted(CLM_MODELS_DICT.keys())
	window_sizes = [
		# 25, 100
		2, 3, 4, 5, 10, 50, 75, 125, 150, 175, 200, 225, 250, 275, 300
	]
	
	# window_sizes = [100]
	top_ns = [1,5] #, 10]

	all_cmds = []
	script_fn = os.path.join(os.getcwd(), 'run_get_clm_predictions.py')
	job_string = f'{DSQ_MODULES} srun python {script_fn}'
	job_num = 0

	# 
	failed_jobs = [
		16
		# 17, 18, 20, 21, 23, 25
	]

	# failed_jobs += list(np.arange(26, 135))

	for i, (task, model, window) in enumerate(product(task_names, model_names, window_sizes)):

		if i not in failed_jobs:
			continue

		if window in [25, 100]:
			save_logits = 1
		else:
			save_logits = 0

		# This only checks against the first top_n (but all are run inside the script)
		out_dir = os.path.join(BASE_DIR, 'derivatives/model-predictions', task, model, f'window-size-{window}')
		out_fn = os.path.join(out_dir, f'task-{task}_model-{model}_window-size-{window}_top-{top_ns[0]}.csv')
		file_exists = os.path.exists(out_fn)

		if OVERWRITE or not file_exists:
			cmd = ''.join([
				f'{job_string} -t {task} -m {model} -w {window} -n {" ".join(str(v) for v in top_ns)} -s {save_logits}']
			)

			all_cmds.append(cmd)
			job_num += 1

	if not all_cmds:
		print (f'No files needing predictions - overwrite if you want to redo predictions', flush=True)
		sys.exit(0)

	dsq_base_string = f'dsq_run_get_clm_predictions'
	logs_dir = os.path.join(BASE_DIR, 'derivatives/logs/modeling/')
	dsq_dir =  os.path.join(BASE_DIR, 'code/submit_scripts/modeling/dsq')
	joblists_dir = os.path.join(BASE_DIR, 'code/submit_scripts/modeling/joblists')

	utils.attempt_makedirs(logs_dir)
	utils.attempt_makedirs(dsq_dir)
	utils.attempt_makedirs(joblists_dir)

	joblist_fn = os.path.join(joblists_dir, f'run_get_clm_predictions.txt')

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
		f"--time={TIME} --nodes={N_NODES} {GPU_INFO} --account={ACCOUNT} --ntasks-per-node={N_TASKS_PER_NODE} --ntasks={N_TASKS} "
		f"--cpus-per-task={CPUS_PER_TASK} --mem-per-cpu={MEM_PER_CPU} --exclude={EXCLUDE}", shell=True)
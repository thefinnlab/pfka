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

PARTITION = 'preemptable'
TIME = '12:00:00'
N_NODES = 1
N_TASKS_PER_NODE = 1
N_TASKS = 1
CPUS_PER_TASK = 1
MEM_PER_CPU = '8G'

if __name__ == '__main__':

	task_list = ['black', 'wheretheressmoke', 'howtodraw']

	# grab the tasks
	all_cmds = []
	script_fn = os.path.join(os.getcwd(), 'run_add_syntax.py')
	job_string = f'{DSQ_MODULES} srun python {script_fn}'
	job_num = 0

	for i, task in enumerate(task_list):

		cmd = ''.join([
			f"{job_string} -t {task}"
		])

		all_cmds.append(cmd)
		job_num += 1

	dsq_base_string = f'dsq_add_syntax'
	logs_dir = os.path.join(BASE_DIR, 'derivatives/logs/behavioral/')
	dsq_dir =  os.path.join(BASE_DIR, 'code/submit_scripts/behavioral/dsq')
	joblists_dir = os.path.join(BASE_DIR, 'code/submit_scripts/behavioral/joblists')

	utils.attempt_makedirs(logs_dir)
	utils.attempt_makedirs(dsq_dir)
	utils.attempt_makedirs(joblists_dir)

	joblist_fn = os.path.join(joblists_dir, f'run_add_syntax.txt')

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
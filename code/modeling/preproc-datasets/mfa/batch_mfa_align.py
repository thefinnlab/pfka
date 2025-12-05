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
CPUS_PER_TASK = 31
MEM_PER_CPU = '8G'
GPU_INFO = ''

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('-d', '--dataset', type=str)
  p = parser.parse_args()

  logs_dir = os.path.join(BASE_DIR, 'derivatives/logs/modeling/')
  dsq_dir =  os.path.join(BASE_DIR, 'code/submit_scripts/modeling/dsq')
  joblists_dir = os.path.join(BASE_DIR, 'code/submit_scripts/modeling/joblists')

  attempt_makedirs(logs_dir)
  attempt_makedirs(dsq_dir)
  attempt_makedirs(joblists_dir)

  all_cmds = []
  script_fn = os.path.join(os.getcwd(), 'mfa_align.py')
  job_string = f'{DSQ_MODULES} python {script_fn}'
  job_string = job_string.replace('dark_matter', 'mfa')
  job_num = 0

  dataset_config = utils.DATASET_CONFIGS[p.dataset]
  output_dir = os.path.join(DATASETS_DIR, 'nlp-datasets', p.dataset)
  
  # for split in dataset_config['splits']:
  print(f'Making job for: {p.dataset}', flush=True)
  cmd = f'{job_string} --dataset {p.dataset} --output_dir {output_dir} --num_jobs {CPUS_PER_TASK} '
  all_cmds.append(cmd)
  job_num += 1

    # break

  if not all_cmds:
    print(f'No matching audio and text files found', flush=True)
    sys.exit(0)

  joblist_fn = os.path.join(joblists_dir, f'mfa_align_dataset_joblist.txt')

  with open(joblist_fn, 'w') as f:
    for cmd in all_cmds:
      f.write(f"{cmd}\n")

  dsq_base_string = f'mfa_align_dataset'
  dsq_batch_fn = os.path.join(dsq_dir, dsq_base_string)
  dsq_out_dir = os.path.join(logs_dir, dsq_base_string)
  array_fmt_width = len(str(job_num))

  if not os.path.exists(dsq_out_dir):
    os.makedirs(dsq_out_dir)

  # subprocess.run('module load dSQ', shell=True)
  subprocess.run(f"dsq --job-file {joblist_fn} --batch-file {dsq_batch_fn}.sh "
    f"--status-dir {dsq_out_dir} --output={dsq_out_dir}/{dsq_base_string}-%A_%{array_fmt_width}a-%N.txt "
    f"--partition {PARTITION} {GPU_INFO} --time={TIME} --nodes={N_NODES} --ntasks-per-node={N_TASKS_PER_NODE} "
    f"--ntasks={N_TASKS} --cpus-per-task={CPUS_PER_TASK} --mem-per-cpu={MEM_PER_CPU}", shell=True)

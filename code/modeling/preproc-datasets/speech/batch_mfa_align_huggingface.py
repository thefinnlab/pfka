import sys, os
import argparse
import subprocess
from pathlib import Path
import glob

sys.path.append('../utils/')

from config import *
import utils 

PARTITION = 'preemptable'
TIME = '5-00:00:00'
N_NODES = 1
N_TASKS_PER_NODE = 1
N_TASKS = 1
CPUS_PER_TASK = 8
MEM_PER_CPU = '8G'
GPU_INFO = ''


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('-d', '--dataset', type=str)
    # parser.add_argument('--subset_size', type=int, default=None,
    #                   help='Number of samples to process per split (default: all)')
  parser.add_argument('--output_dir', type=str, default=None,
                    help='Base directory for output (default: dataset_name_processing)')
  p = parser.parse_args()

  utils.attempt_makedirs(DSQ_DIR)
  utils.attempt_makedirs(LOGS_DIR)
  utils.attempt_makedirs(JOBLIST_DIR)

  all_cmds = []
  script_fn = os.path.join(os.getcwd(), 'mfa_align_huggingface.py')
  job_string = f'{DSQ_MODULES} python {script_fn}'
  job_num = 0

  print(f'Making job for: {p.dataset}', flush=True)
  cmd = f'{job_string} --dataset {p.dataset} --output_dir {p.output_dir} --num_jobs {CPUS_PER_TASK} '
  all_cmds.append(cmd)
  job_num += 1

    # break

  if not all_cmds:
    print(f'No matching audio and text files found', flush=True)
    sys.exit(0)

  joblist_fn = os.path.join(JOBLIST_DIR, f'{p.dataset}_mfa_align_huggingface_joblist.txt')

  with open(joblist_fn, 'w') as f:
    for cmd in all_cmds:
      f.write(f"{cmd}\n")

  dsq_base_string = f'dsq_mfa_align_huggingface'
  dsq_batch_fn = os.path.join(DSQ_DIR, dsq_base_string)
  dsq_out_dir = os.path.join(LOGS_DIR, dsq_base_string)
  array_fmt_width = len(str(job_num))

  if not os.path.exists(dsq_out_dir):
    os.makedirs(dsq_out_dir)

  # subprocess.run('module load dSQ', shell=True)
  subprocess.run(f"dsq --job-file {joblist_fn} --batch-file {dsq_batch_fn}.sh "
    f"--status-dir {dsq_out_dir} --output={dsq_out_dir}/{dsq_base_string}-%A_%{array_fmt_width}a-%N.txt "
    f"--partition {PARTITION} {GPU_INFO} --time={TIME} --nodes={N_NODES} --ntasks-per-node={N_TASKS_PER_NODE} "
    f"--ntasks={N_TASKS} --cpus-per-task={CPUS_PER_TASK} --mem-per-cpu={MEM_PER_CPU}", shell=True)

import os
import sys
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
CPUS_PER_TASK = 8
MEM_PER_CPU = '8G'
GPU_INFO = ''

TIME = '2-12:00:00'
CPUS_PER_TASK = 16
MEM_PER_CPU = '8G'
PARTITION = 'v100_preemptable'
GPU_INFO = '--gres=gpu:1'
NODE_LIST = ''#--nodelist=a03,a04'
EXCLUDE = ''
ACCOUNT = 'dbic'

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='Process speech datasets to Praat TextGrids')
  parser.add_argument('-d','--dataset', type=str, choices=['lrs3', 'avspeech', 'voxceleb2'])
  parser.add_argument('--overwrite', type=int, default=0,
            help='Force extraction even if files exist')

  p = parser.parse_args()

  logs_dir = os.path.join(BASE_DIR, 'derivatives/logs/modeling/')
  dsq_dir =  os.path.join(BASE_DIR, 'code/submit_scripts/modeling/dsq')
  joblists_dir = os.path.join(BASE_DIR, 'code/submit_scripts/modeling/joblists')

  attempt_makedirs(logs_dir)
  attempt_makedirs(dsq_dir)
  attempt_makedirs(joblists_dir)

  all_cmds = []
  script_fn = os.path.join(os.getcwd(), 'preproc_av_dataset.py')
  job_num = 0


  dataset_config = utils.DATASET_CONFIGS[p.dataset]
  splits = dataset_config['splits']
  splits = splits[::-1]

  output_dir = os.path.join(DATASETS_DIR, 'nlp-datasets', p.dataset)

  for split in splits:

    # Number of subdatasets for efficient processing
    if split == 'train':
        N_SHARDS = 25
        # continue
    else:
        N_SHARDS = 5

    # if split != 'train':
    #   continue

    for shard in range(N_SHARDS):

        # if shard != 4:
        #   continue
          
        print(f'Making job for: {p.dataset} {split}, {shard+1}/{N_SHARDS} shards', flush=True)

        cmd = [
          f"{DSQ_MODULES.replace('dark_matter', 'prosody')} ",
          # f"python normalize_videos.py --dataset {dataset} --output_dir {output_dir} --split {split} --num_shards {N_SHARDS} --current_shard {shard}; ", 
          # f"python extract_dataset_audio.py --dataset {p.dataset} --output_dir {output_dir} --split {split} --num_shards {N_SHARDS} --current_shard {shard} --num_jobs {CPUS_PER_TASK}; ", 
          f"python transcribe_audio.py --dataset {p.dataset} --output_dir {output_dir} --split {split} --batch_size 64 --num_shards {N_SHARDS} --current_shard {shard}; ", 
          f"python prepare_corpus.py --dataset {p.dataset} --output_dir {output_dir} --split {split} --num_shards {N_SHARDS} --current_shard {shard} --num_jobs {CPUS_PER_TASK}", 
        ]

        cmd = "".join(cmd)
        all_cmds.append(cmd)
        job_num += 1

  if not all_cmds:
    print(f'No matching audio and text files found', flush=True)
    sys.exit(0)

  joblist_fn = os.path.join(joblists_dir, f'{p.dataset}_preproc_av_dataset.txt')

  with open(joblist_fn, 'w') as f:
    for cmd in all_cmds:
      f.write(f"{cmd}\n")

  dsq_base_string = f'{p.dataset}_preproc_av_dataset'
  dsq_batch_fn = os.path.join(dsq_dir, dsq_base_string)
  dsq_out_dir = os.path.join(logs_dir, dsq_base_string)
  array_fmt_width = len(str(job_num))

  if not os.path.exists(dsq_out_dir):
    os.makedirs(dsq_out_dir)

  subprocess.run(f"dsq --job-file {joblist_fn} --batch-file {dsq_batch_fn}.sh "
    f"--status-dir {dsq_out_dir} --partition={PARTITION} --output={dsq_out_dir}/{dsq_base_string}-%A_%{array_fmt_width}a-%N.txt "
    f"--time={TIME} --nodes={N_NODES} {GPU_INFO} --account={ACCOUNT} --ntasks-per-node={N_TASKS_PER_NODE} --ntasks={N_TASKS} "
    f"--cpus-per-task={CPUS_PER_TASK} --mem-per-cpu={MEM_PER_CPU} --exclude={EXCLUDE}", shell=True)

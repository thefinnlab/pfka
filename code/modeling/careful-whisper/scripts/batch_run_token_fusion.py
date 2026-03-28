import sys, os
import glob
import argparse
import subprocess
import numpy as np

from config import *
import utils

PARTITION='preemptable'
TIME = '24:00:00'
N_NODES = 1
N_TASKS_PER_NODE = 1
N_TASKS = 1
CPUS_PER_TASK = 8
MEM_PER_CPU = '8G'
GPU_INFO = ''

TIME = '2-12:00:00'
EXCLUDE = ''
CPUS_PER_TASK = 16
MEM_PER_CPU = '8G'
PARTITION = 'gpuq'
GPU_INFO = '--gres=gpu:1'
EXCLUDE = 'p04'
ACCOUNT = 'dbic' #dbic

DATASET_INFO = {
    'lrs3': [
        'data.dataset_name=lrs3',
        'data.data_dir=\${paths.data_dir}/lrs3',
    ],
    'voxceleb2': [
        'data.dataset_name=voxceleb2',
        'data.data_dir=\${paths.data_dir}/voxceleb2',
    ],
    'avspeech': [
        'data.dataset_name=avspeech',
        'data.data_dir=\${paths.data_dir}/avspeech',
    ],
    'av-combined': [
        'data.dataset_name=av-combined',
        'data.data_dir=\${paths.data_dir}/av-combined',
    ]
}

MODEL_CONFIGS = {

    # Whisper w/ CLM integration
    'token-fusion_wav2vec-data2vec': [
        f"model.input_dim1=1024",
        f"model.input_dim2=1024",

        f"model.input_name1=audio_features",
        f"model.input_name2=video_features",
        f"model.hidden_dim=1024",
        # f"model.loss_fn=orthogonality_loss",
        # f"model.optimizer.lr=5e-5"
    ],

}

if __name__ == "__main__":

    EXPERIMENT_NAME = 'token_fusion'
    MODEL_CONFIG_NAME = ''
 
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str)
    p = parser.parse_args()

    # make directories
    dsq_dir = os.path.join(SUBMIT_DIR, 'dsq')
    joblist_dir = os.path.join(SUBMIT_DIR, 'joblists')
    logs_dir = os.path.join(LOGS_DIR)

    utils.attempt_makedirs(dsq_dir)
    utils.attempt_makedirs(joblist_dir)
    utils.attempt_makedirs(logs_dir)

    all_cmds = []
    script_fn = os.path.join(BASE_DIR, 'src/train.py')
    job_string = f'{DSQ_MODULES} srun python {script_fn}'
    job_num = 0

    # Dataset overrides --> set up dataset
    dataset_config = ' '.join(DATASET_INFO[p.dataset])
    counter = 0

    for model_name, model_config in MODEL_CONFIGS.items():

        # if counter not in np.arange(10, 20).tolist():
        #     counter += 1
        #     continue

        # counter += 1
        print (model_name)

        # Model config --> change model configurations
        model_name += MODEL_CONFIG_NAME
        model_config_str = ' '.join(model_config)
        
        # Logger overrides --> change the name based on the dataset
        wandb_config = (
            f"logger.wandb.project={p.dataset}-tokenfusion-audiovisual "
            f"logger.wandb.name={model_name} "
            f"logger.wandb.group={model_name.split('_')[0]} "
        )

        hydra_config = (
            "hydra.run.dir=\${paths.log_dir}/\${task_name}/token-fusion/"
            f"{p.dataset}/"
            f"{model_name}/"
        )

        cmd = (
            f"{job_string} "
            f"experiment={EXPERIMENT_NAME}.yaml "
            f"{model_config_str} "
            f"{dataset_config} "
            f"{wandb_config} "
            f"{hydra_config} "
        )

        all_cmds.append(cmd)
        job_num += 1

    if not all_cmds:
        print (f'No model needing extraction - overwrite if you want to redo extraction', flush=True)
        sys.exit(0)

    joblist_fn = os.path.join(joblist_dir, f'{p.dataset}_token-fusion_experiments.txt')

    with open(joblist_fn, 'w') as f:
        for cmd in all_cmds:
            f.write(f"{cmd}\n")
    
    dsq_base_string = f'dsq_{p.dataset}_token-fusion'
    dsq_batch_fn = os.path.join(dsq_dir, dsq_base_string)
    dsq_out_dir = os.path.join(logs_dir, dsq_base_string)
    array_fmt_width = len(str(job_num)) 

    if not os.path.exists(dsq_out_dir):
        os.makedirs(dsq_out_dir)
    
    subprocess.run(f"dsq --job-file {joblist_fn} --batch-file {dsq_batch_fn}.sh "
        f"--status-dir {dsq_out_dir} --partition={PARTITION} --output={dsq_out_dir}/{dsq_base_string}-%A_%{array_fmt_width}a-%N.txt "
        f"--time={TIME} --account={ACCOUNT} --nodes={N_NODES} {GPU_INFO} --ntasks-per-node={N_TASKS_PER_NODE} --ntasks={N_TASKS} "
        f"--exclude={EXCLUDE} --cpus-per-task={CPUS_PER_TASK} --mem-per-cpu={MEM_PER_CPU}", shell=True)

import sys, os
import argparse
import numpy as np

from config import *
import utils

# SLURM settings
TIME = '3-00:00:00'
EXCLUDE = 'p04'
CPUS_PER_TASK = 16
MEM_PER_CPU = '8G'
PARTITION = 'gpuq'
GPU_INFO = '--gres=gpu:1'
N_NODES = 1
N_TASKS_PER_NODE = 1
N_TASKS = 1
ACCOUNT = 'dbic'

# Usage:
#   Single type:
#     python batch_careful-whisper_experiments.py -d lrs3 -m audio
#     python batch_careful-whisper_experiments.py -d voxceleb2 -m audio --subsets 1
#
#   All types for a dataset (bash):
#     for m in text audio prosody audiovisual audio-control prosody-control audiovisual-control; do
#         python batch_careful-whisper_experiments.py -d lrs3 -m $m
#     done
#
#   Subsets (voxceleb2 only):
#     for m in text audio prosody audiovisual; do
#         python batch_careful-whisper_experiments.py -d voxceleb2 -m $m --subsets 1
#     done
#
# NOTE: audiovisual and audiovisual-control require a token-fusion checkpoint.
#       Update utils.DATASET_CONFIG with the correct epoch after rerunning token fusion.

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
    ],
}

# Each model_type maps to {run_name: [hydra overrides]}.
# run_name is used for wandb.name and hydra.run.dir.
# Audiovisual types have their ckpt_path injected at runtime from utils.DATASET_CONFIG.
MODEL_TYPE_CONFIGS = {

    'text': {
        'text': [
            "model.config.cross_attention=False",
            "model.config.use_causal_cross_attention=False",
        ],
    },

    'audio': {
        'audio': [
            "model.config.cross_attention=True",
            "model.config.use_causal_cross_attention=True",
            "model.config.context_embed_dropout=0.1",
            "model.config.context_pos_embed=True",
        ],
    },

    'prosody': {
        'prosody': [
            "model.config.cross_attention=True",
            "model.config.use_causal_cross_attention=True",
            "model.config.context_type=prominence",
            "model.config.context_dim=1",
            "model.config.context_embed_dropout=0.1",
            "model.config.context_pos_embed=True",
        ],
    },

    'audiovisual': {
        'audiovisual': [
            "model.config.cross_attention=True",
            "model.config.use_causal_cross_attention=True",
            "model.config.context_type=audiovisual_features",
            "model.config.context_embed_dropout=0.1",
            "model.config.context_pos_embed=True",
            "data.token_fusion_method=mlp",
            # ckpt_path injected below
        ],
    },

    'audio-control': {
        'audio-shuffled': [
            "model.config.cross_attention=True",
            "model.config.use_causal_cross_attention=True",
            "model.config.context_embed_dropout=0.1",
            "model.config.context_pos_embed=True",
            "data.shuffle_context_dims=True",
        ],
    },

    'prosody-control': {
        'prosody-shuffled': [
            "model.config.cross_attention=True",
            "model.config.use_causal_cross_attention=True",
            "model.config.context_type=prominence",
            "model.config.context_dim=null",
            "model.config.context_embed_dropout=0.1",
            "model.config.context_pos_embed=True",
            "data.shuffle_context_dims=True",
        ],
    },

    'audiovisual-control': {
        'audiovisual-shuffled': [
            "model.config.cross_attention=True",
            "model.config.use_causal_cross_attention=True",
            "model.config.context_type=audiovisual_features",
            "model.config.context_embed_dropout=0.1",
            "model.config.context_pos_embed=True",
            "data.token_fusion_method=mlp",
            # ckpt_path injected below
            "data.shuffle_context_dims=True",
        ],
    },
}


def create_model_variations(base_configs, subset_percentages=None):
    """Return {run_name: [overrides]} optionally expanded across subset sizes."""
    variations = {}
    for model_name, config in base_configs.items():
        if subset_percentages is not None:
            for subset in subset_percentages:
                subset_name = f"{model_name}_subset-{str(subset).zfill(3)}"
                subset_config = config.copy()
                subset_config.append(f"data.subset_percentage={subset}")
                variations[subset_name] = subset_config
        else:
            variations[model_name] = config.copy()
    return variations


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, required=True,
                        choices=list(DATASET_INFO.keys()))
    parser.add_argument('-m', '--model_type', type=str, required=True,
                        choices=list(MODEL_TYPE_CONFIGS.keys()),
                        help='Model type to generate jobs for')
    parser.add_argument('--subsets', type=int, default=0,
                        help='Generate subset-size jobs instead of full-data jobs (pass 1 to enable)')
    p = parser.parse_args()

    EXPERIMENT_NAME = 'careful_whisper'

    # Deep-copy to avoid mutating the module-level dict
    base_configs = {k: list(v) for k, v in MODEL_TYPE_CONFIGS[p.model_type].items()}

    # Inject token-fusion checkpoint path for audiovisual models
    if p.model_type in ('audiovisual', 'audiovisual-control'):
        ckpt_rel = utils.DATASET_CONFIG[p.dataset]['ckpt_path']
        ckpt_override = (
            "data.ckpt_path=\${paths.log_dir}train/token-fusion/"
            f"{p.dataset}/{ckpt_rel}"
        )
        for name in base_configs:
            base_configs[name].append(ckpt_override)

    # Subset percentages: log-spaced 2–25%, then linear 30–70%
    if p.subsets == 1:
        subset_pcts = np.logspace(0.3, 1.4, 10) / 100
        subset_pcts = np.concatenate((subset_pcts, np.arange(0.3, 1.0, 0.1)))
        subset_percentages = (100 * np.sort(np.round(subset_pcts, 2))).astype(int)
    else:
        subset_percentages = None

    MODEL_CONFIGS = create_model_variations(base_configs, subset_percentages)

    # Make output directories
    dsq_dir = os.path.join(SUBMIT_DIR, 'dsq')
    joblist_dir = os.path.join(SUBMIT_DIR, 'joblists')
    logs_dir = LOGS_DIR

    utils.attempt_makedirs(dsq_dir)
    utils.attempt_makedirs(joblist_dir)
    utils.attempt_makedirs(logs_dir)

    all_cmds = []
    script_fn = os.path.join(BASE_DIR, 'src/train.py')
    job_string = f'{DSQ_MODULES} srun python {script_fn}'
    job_num = 0

    dataset_config = ' '.join(DATASET_INFO[p.dataset])
    wandb_job_type = 'subsets' if p.subsets == 1 else 'full-data'

    for model_name, model_config in MODEL_CONFIGS.items():

        model_config_str = ' '.join(model_config)

        # wandb group = base name without _subset-XXX suffix
        wandb_group = model_name.split('_subset-')[0]

        wandb_config = (
            f"logger.wandb.project={p.dataset}-audiovisual "
            f"logger.wandb.name={model_name} "
            f"logger.wandb.group={wandb_group} "
            f"logger.wandb.job_type={wandb_job_type} "
        )

        hydra_config = (
            "hydra.run.dir=\${paths.log_dir}/\${task_name}/careful-whisper/"
            f"{p.dataset}/{model_name}/"
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
        print('No jobs to submit', flush=True)
        sys.exit(0)

    joblist_fn = os.path.join(
        joblist_dir, f'{p.dataset}-{p.model_type}_careful_whisper_experiments.txt'
    )

    with open(joblist_fn, 'w') as f:
        for cmd in all_cmds:
            f.write(f"{cmd}\n")

    print(f"Joblist written: {joblist_fn} ({job_num} jobs)")

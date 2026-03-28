"""
Batch script for generating SLURM job arrays for CarefulWhisper results.

Subcommands:
  results   Generate inference jobs (one per dataset × model variant) calling
            run_careful_whisper_results.py results
  analysis  Generate a lightweight analysis job calling
            run_careful_whisper_results.py analysis
            (NOTE: subsets are not supported for analysis)

Usage:
  Results (all 7 model variants):
    python batch_careful_whisper_results.py results -d lrs3 avspeech voxceleb2
    python batch_careful_whisper_results.py results -d lrs3 -s test

  Analysis (compute ΔI from saved CSVs):
    python batch_careful_whisper_results.py analysis -d lrs3 avspeech voxceleb2
"""

import sys, os
import argparse
import glob

sys.path.append('../utils/')
sys.path.append('../modeling/careful-whisper/scripts/')

from config import *
from dataset_utils import attempt_makedirs

# Inference jobs require the prosody env (pyrootutils, hydra, lightning)
DSQ_MODULES = DSQ_MODULES.replace('conda activate analysis', 'conda activate prosody')
DSQ_MODULES = DSQ_MODULES.replace('cuda/11.2', 'cuda/12.3')

# -----------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------
MODELING_DIR      = os.path.join(BASE_DIR, 'code/modeling/careful-whisper')
MODELING_LOGS     = os.path.join(MODELING_DIR, 'logs/train/careful-whisper')
TOKEN_FUSION_LOGS = os.path.join(MODELING_DIR, 'logs/train/token-fusion')

# -----------------------------------------------------------------------
# SLURM settings
# -----------------------------------------------------------------------
EXCLUDE          = '' #'p04'
CPUS_PER_TASK    = 16
MEM_PER_CPU      = '8G'
PARTITION        = 'v100_preemptable'
GPU_INFO         = '--gres=gpu:1'
N_NODES          = 1
N_TASKS_PER_NODE = 1
N_TASKS          = 1
ACCOUNT          = 'dbic'
RESULTS_TIME     = '1-00:00:00'
ANALYSIS_TIME    = '0-01:00:00'

# -----------------------------------------------------------------------
# Model variants
# Base models are defined explicitly. Shuffled variants are auto-generated:
#   - shuffle=1, run_dir and model_name_suffix gain '-shuffled'
#   - prosody shuffled: context_dim=1 → context_dim=null
# (data.shuffle_context_dims is handled by the -shuffle_context_dims CLI flag)
# -----------------------------------------------------------------------
_BASE_VARIANTS = {
    'text': {
        'run_dir':           'text',
        'overrides': [
            'model.config.cross_attention=False',
            'model.config.use_causal_cross_attention=False',
        ],
        'context_type':      'audio_features',
        'model_name_suffix': 'text-careful-whisper_no-xattn',
        'audiovisual':       0,
        'has_shuffled':      False,
    },
    'audio': {
        'run_dir':           'audio',
        'overrides': [
            'model.config.cross_attention=True',
            'model.config.use_causal_cross_attention=True',
            'model.config.context_embed_dropout=0.1',
            'model.config.context_pos_embed=True',
        ],
        'context_type':      'audio_features',
        'model_name_suffix': 'audio-careful-whisper_causal-xattn',
        'audiovisual':       0,
        'has_shuffled':      True,
    },
    'prosody': {
        'run_dir':           'prosody',
        'overrides': [
            'model.config.cross_attention=True',
            'model.config.use_causal_cross_attention=True',
            'model.config.context_type=prominence',
            'model.config.context_dim=1',
            'model.config.context_embed_dropout=0.1',
            'model.config.context_pos_embed=True',
        ],
        'context_type':      'prominence',
        'model_name_suffix': 'prosody-careful-whisper_causal-xattn',
        'audiovisual':       0,
        'has_shuffled':      True,
    },
    'audiovisual': {
        'run_dir':           'audiovisual',
        'overrides': [
            'model.config.cross_attention=True',
            'model.config.use_causal_cross_attention=True',
            'model.config.context_type=audiovisual_features',
            'model.config.context_embed_dropout=0.1',
            'model.config.context_pos_embed=True',
        ],
        'context_type':      'audiovisual_features',
        'model_name_suffix': 'audiovisual-careful-whisper_causal-xattn_token-fusion-mlp',
        'audiovisual':       1,
        'has_shuffled':      True,
    },
}


def _make_variants(base):
    """Expand base model configs into full + shuffled variants."""
    variants = {}
    for name, cfg in base.items():
        variants[name] = {k: v for k, v in cfg.items() if k != 'has_shuffled'}
        variants[name]['shuffle'] = 0
        if cfg['has_shuffled']:
            prefix, rest = cfg['model_name_suffix'].split('-', 1)
            shuffled_overrides = [
                o.replace('context_dim=1', 'context_dim=null') for o in cfg['overrides']
            ]
            variants[f'{name}-shuffled'] = {
                'run_dir':           f'{name}-shuffled',
                'overrides':         shuffled_overrides,
                'context_type':      cfg['context_type'],
                'shuffle':           1,
                'model_name_suffix': f'{prefix}-shuffled-{rest}',
                'audiovisual':       cfg['audiovisual'],
            }
    return variants


MODEL_VARIANTS = _make_variants(_BASE_VARIANTS)

SUBSET_PERCENTAGES = [2, 3, 4, 5, 6, 8, 11, 14, 19, 25, 30, 40, 50, 60, 70, 80, 90]


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def find_best_checkpoint(ckpt_dir):
    """Return the best epoch checkpoint in ckpt_dir, falling back to last.ckpt."""
    epoch_ckpts = sorted(glob.glob(os.path.join(ckpt_dir, 'epoch_*.ckpt')))
    if epoch_ckpts:
        return epoch_ckpts[0]
    last = os.path.join(ckpt_dir, 'last.ckpt')
    if os.path.exists(last):
        return last
    raise FileNotFoundError(f'No checkpoint found in {ckpt_dir}')


def find_model_checkpoint(dataset, run_dir_name):
    ckpt_dir = os.path.join(MODELING_LOGS, dataset, run_dir_name, 'checkpoints')
    return find_best_checkpoint(ckpt_dir)


def find_token_fusion_checkpoint(dataset):
    """Dynamically find the best token-fusion checkpoint for a dataset."""
    # token-fusion runs are stored under a single subdirectory per dataset;
    # glob for any run dir containing a checkpoints/ folder
    pattern = os.path.join(TOKEN_FUSION_LOGS, dataset, '*', 'checkpoints')
    ckpt_dirs = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not ckpt_dirs:
        raise FileNotFoundError(
            f'No token-fusion checkpoint directory found under '
            f'{TOKEN_FUSION_LOGS}/{dataset}/'
        )
    # Use the last (most recent) run directory if multiple exist
    return find_best_checkpoint(ckpt_dirs[-1])


def write_dsq_script(dsq_fn, log_dir, joblist_fn, n_jobs, time, dsq_name, gpu=True):
    """Write a dSQ SLURM array submission script."""
    gpu_flag = f'{GPU_INFO} ' if gpu else ''
    content = (
        f"#!/bin/bash\n"
        f"#SBATCH --output {log_dir}/{dsq_name}-%A_%1a-%N.txt\n"
        f"#SBATCH --array 0-{n_jobs - 1}\n"
        f"#SBATCH --job-name {dsq_name}\n"
        f"#SBATCH --partition={PARTITION} --time={time} --account={ACCOUNT} "
        f"--nodes={N_NODES} {gpu_flag}--ntasks-per-node={N_TASKS_PER_NODE} "
        f"--ntasks={N_TASKS} --exclude={EXCLUDE} "
        f"--cpus-per-task={CPUS_PER_TASK} --mem-per-cpu={MEM_PER_CPU}\n\n"
        f"# DO NOT EDIT LINE BELOW\n"
        f"/optnfs/common/dSQ/dSQ-1.05/dSQBatch.py "
        f"--job-file {joblist_fn} --status-dir {log_dir}\n"
    )
    with open(dsq_fn, 'w') as f:
        f.write(content)


# -----------------------------------------------------------------------
# Subcommand: results
# -----------------------------------------------------------------------

def run_results(p):
    """Generate SLURM inference jobs for all model variants × datasets."""

    analysis_dir = os.path.join(BASE_DIR, 'code/analysis')
    submit_dir   = os.path.join(analysis_dir, 'submit_scripts')
    dsq_dir      = os.path.join(submit_dir, 'dsq')
    joblist_dir  = os.path.join(submit_dir, 'joblists')
    logs_dir     = os.path.join(BASE_DIR, 'derivatives', 'logs', 'careful-whisper', 'results')

    for d in [dsq_dir, joblist_dir, logs_dir]:
        attempt_makedirs(d)

    run_script = os.path.join(analysis_dir, 'run_careful_whisper_results.py')
    all_cmds   = []

    for dataset in p.datasets:
        for variant_name, variant in MODEL_VARIANTS.items():

            try:
                ckpt_path = find_model_checkpoint(dataset, variant['run_dir'])
            except FileNotFoundError as e:
                print(f'WARNING: {e}', flush=True)
                continue

            model_name    = f"{dataset}_{variant['model_name_suffix']}"
            all_overrides = ['experiment=careful_whisper.yaml'] + variant['overrides']
            overrides_str = '-overrides ' + ' '.join(all_overrides)

            token_fusion_arg = ''
            if variant['audiovisual']:
                try:
                    tf_ckpt = find_token_fusion_checkpoint(dataset)
                    token_fusion_arg = f'-token_fusion_ckpt {tf_ckpt}'
                except FileNotFoundError as e:
                    print(f'WARNING: {e}', flush=True)
                    continue

            cmd = (
                f"{DSQ_MODULES} "
                f"srun python {run_script} results "
                f"-d {dataset} "
                f"-s {p.split} "
                f"-model_name {model_name} "
                f"-ckpt_path {ckpt_path} "
                f"{overrides_str} "
                f"-shuffle_context_dims {variant['shuffle']} "
                f"-context_type {variant['context_type']} "
                f"-av {variant['audiovisual']} "
                f"{token_fusion_arg}"
            ).strip()

            all_cmds.append(cmd)
            print(f'  [{dataset}/{variant_name}] {os.path.basename(ckpt_path)}')

    if not all_cmds:
        print('No jobs to submit.', flush=True)
        sys.exit(0)

    dsq_name   = 'dsq_careful_whisper_results'
    joblist_fn = os.path.join(joblist_dir, 'careful_whisper_results.txt')
    dsq_fn     = os.path.join(dsq_dir, f'{dsq_name}.sh')
    log_dir    = os.path.join(logs_dir, dsq_name)

    attempt_makedirs(log_dir)

    with open(joblist_fn, 'w') as f:
        for cmd in all_cmds:
            f.write(f"{cmd}\n")
    print(f'\nJoblist written: {joblist_fn} ({len(all_cmds)} jobs)')

    write_dsq_script(dsq_fn, log_dir, joblist_fn, len(all_cmds),
                     RESULTS_TIME, dsq_name, gpu=True)
    print(f'DSQ script written: {dsq_fn}')
    print(f'\nTo submit: sbatch {dsq_fn}')


# -----------------------------------------------------------------------
# Subcommand: results-subsets
# -----------------------------------------------------------------------

def run_results_subsets(p):
    """Generate SLURM inference jobs for all base modalities × subset percentages × datasets."""

    analysis_dir = os.path.join(BASE_DIR, 'code/analysis')
    submit_dir   = os.path.join(analysis_dir, 'submit_scripts')
    dsq_dir      = os.path.join(submit_dir, 'dsq')
    joblist_dir  = os.path.join(submit_dir, 'joblists')
    logs_dir     = os.path.join(BASE_DIR, 'derivatives', 'logs', 'careful-whisper', 'results')

    for d in [dsq_dir, joblist_dir, logs_dir]:
        attempt_makedirs(d)

    run_script = os.path.join(analysis_dir, 'run_careful_whisper_results.py')
    all_cmds   = []

    for dataset in p.datasets:
        for modality, cfg in _BASE_VARIANTS.items():
            for pct in SUBSET_PERCENTAGES:
                run_dir = f"{modality}_subset-{pct:03d}"

                try:
                    ckpt_path = find_model_checkpoint(dataset, run_dir)
                except FileNotFoundError as e:
                    print(f'WARNING: {e}', flush=True)
                    continue

                model_name    = f"{dataset}_{cfg['model_name_suffix']}_subset-{pct:03d}"
                all_overrides = ['experiment=careful_whisper.yaml'] + cfg['overrides']
                overrides_str = '-overrides ' + ' '.join(all_overrides)

                token_fusion_arg = ''
                if cfg['audiovisual']:
                    try:
                        tf_ckpt = find_token_fusion_checkpoint(dataset)
                        token_fusion_arg = f'-token_fusion_ckpt {tf_ckpt}'
                    except FileNotFoundError as e:
                        print(f'WARNING: {e}', flush=True)
                        continue

                cmd = (
                    f"{DSQ_MODULES} "
                    f"srun python {run_script} results "
                    f"-d {dataset} "
                    f"-s {p.split} "
                    f"-model_name {model_name} "
                    f"-ckpt_path {ckpt_path} "
                    f"{overrides_str} "
                    f"-shuffle_context_dims 0 "
                    f"-context_type {cfg['context_type']} "
                    f"-av {cfg['audiovisual']} "
                    f"{token_fusion_arg}"
                ).strip()

                all_cmds.append(cmd)
                print(f'  [{dataset}/{run_dir}] {os.path.basename(ckpt_path)}')

    if not all_cmds:
        print('No jobs to submit.', flush=True)
        sys.exit(0)

    dsq_name   = 'dsq_careful_whisper_results_subsets'
    joblist_fn = os.path.join(joblist_dir, 'careful_whisper_results_subsets.txt')
    dsq_fn     = os.path.join(dsq_dir, f'{dsq_name}.sh')
    log_dir    = os.path.join(logs_dir, dsq_name)

    attempt_makedirs(log_dir)

    with open(joblist_fn, 'w') as f:
        for cmd in all_cmds:
            f.write(f"{cmd}\n")
    print(f'\nJoblist written: {joblist_fn} ({len(all_cmds)} jobs)')

    write_dsq_script(dsq_fn, log_dir, joblist_fn, len(all_cmds),
                     RESULTS_TIME, dsq_name, gpu=True)
    print(f'DSQ script written: {dsq_fn}')
    print(f'\nTo submit: sbatch {dsq_fn}')


# -----------------------------------------------------------------------
# Subcommand: analysis
# -----------------------------------------------------------------------

def run_analysis(p):
    """Generate a lightweight SLURM analysis job (no GPU)."""

    analysis_dir = os.path.join(BASE_DIR, 'code/analysis')
    submit_dir   = os.path.join(analysis_dir, 'submit_scripts')
    dsq_dir      = os.path.join(submit_dir, 'dsq')
    joblist_dir  = os.path.join(submit_dir, 'joblists')
    logs_dir     = os.path.join(BASE_DIR, 'derivatives', 'logs', 'careful-whisper', 'analysis')

    for d in [dsq_dir, joblist_dir, logs_dir]:
        attempt_makedirs(d)

    run_script   = os.path.join(analysis_dir, 'run_careful_whisper_results.py')
    datasets_str = ' '.join(p.datasets)

    cmd = (
        f"{DSQ_MODULES} "
        f"srun python {run_script} analysis "
        f"-d {datasets_str} "
        f"-s {p.split}"
    ).strip()

    dsq_name   = 'dsq_careful_whisper_analysis'
    joblist_fn = os.path.join(joblist_dir, 'careful_whisper_analysis.txt')
    dsq_fn     = os.path.join(dsq_dir, f'{dsq_name}.sh')
    log_dir    = os.path.join(logs_dir, dsq_name)

    attempt_makedirs(log_dir)

    with open(joblist_fn, 'w') as f:
        f.write(f"{cmd}\n")
    print(f'Joblist written: {joblist_fn} (1 job)')

    write_dsq_script(dsq_fn, log_dir, joblist_fn, 1,
                     ANALYSIS_TIME, dsq_name, gpu=False)
    print(f'DSQ script written: {dsq_fn}')
    print(f'\nTo submit: sbatch {dsq_fn}')


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Generate SLURM jobs for CarefulWhisper results and analysis.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest='subcommand', required=True)

    # ---- results subcommand ----
    res_p = subparsers.add_parser(
        'results',
        help='Generate inference jobs (all model variants × datasets)',
    )
    res_p.add_argument('-d', '--datasets', type=str, nargs='+',
                       default=['lrs3', 'avspeech', 'voxceleb2'])
    res_p.add_argument('-s', '--split', type=str, default='test')

    # ---- results-subsets subcommand ----
    sub_p = subparsers.add_parser(
        'results-subsets',
        help='Generate inference jobs for subset models (all modalities × subset percentages × datasets)',
    )
    sub_p.add_argument('-d', '--datasets', type=str, nargs='+',
                       default=['lrs3', 'avspeech', 'voxceleb2'])
    sub_p.add_argument('-s', '--split', type=str, default='test')

    # ---- analysis subcommand ----
    an_p = subparsers.add_parser(
        'analysis',
        help='Generate analysis job (compute ΔI from saved CSVs; no subsets)',
    )
    an_p.add_argument('-d', '--datasets', type=str, nargs='+',
                      default=['lrs3', 'avspeech', 'voxceleb2'])
    an_p.add_argument('-s', '--split', type=str, default='test')

    p = parser.parse_args()

    if p.subcommand == 'results':
        run_results(p)
    elif p.subcommand == 'results-subsets':
        run_results_subsets(p)
    else:
        run_analysis(p)

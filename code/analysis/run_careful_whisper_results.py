"""
CarefulWhisper results script.

Subcommands:
  results   Run inference for one model/checkpoint and save per-sequence NLL CSV.
  analysis  Load saved NLL CSVs for all model variants and compute structured
            information gain (ΔI = CE_scr − CE_multi).

Usage:
  python run_careful_whisper_results.py results -d lrs3 -s test \\
      -model_name lrs3_audio-careful-whisper_causal-xattn \\
      -ckpt_path /path/to/epoch_005.ckpt \\
      -overrides model.config.cross_attention=True \\
      -context_type audio_features -shuffle_context_dims 0 -av 0

  python run_careful_whisper_results.py analysis -d lrs3 avspeech voxceleb2
"""

import warnings
warnings.filterwarnings("ignore")

import os, sys
import argparse

sys.path.append('../utils/')
sys.path.append('../modeling/careful-whisper/scripts/')

from config import *
from dataset_utils import attempt_makedirs

import numpy as np
import pandas as pd

import pyrootutils
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
import hydra
from hydra import initialize, compose
from lightning import LightningModule
import prosody_utils as prosody
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Must be called before src.* imports so modeling dir is on sys.path
_MODELING_DIR = os.path.join(BASE_DIR, 'code/modeling/careful-whisper/')
pyrootutils.setup_root(_MODELING_DIR, indicator=".project-root", pythonpath=True)

from src.data.components.audio_text_dataset import AudioTextDataset
from src.data.components.collators import audio_text_collator


# -----------------------------------------------------------------------
# Results subcommand helpers
# -----------------------------------------------------------------------

def load_model(config_path, ckpt_path, overrides):
    with initialize(version_base="1.3", config_path=config_path):
        cfg = compose(config_name="train.yaml", overrides=overrides)

    model: LightningModule = hydra.utils.instantiate(cfg.model)

    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    return cfg, model


def get_model_results(model, dataloader):
    df_results = []

    for i, batch in enumerate(dataloader):
        print(f'Batch {i+1}/{len(dataloader)}', flush=True)

        with torch.no_grad():
            outputs = model.step(batch=batch)

        outputs = model._calculate_metrics(outputs, batch, stage=None)

        metrics = {k: v.numpy() for k, v in outputs.items()}
        df_results.append(pd.DataFrame.from_dict(metrics, orient='index').T)

    return pd.concat(df_results).reset_index(drop=True)


def run_results(p):
    modeling_dir = _MODELING_DIR
    results_dir  = os.path.join(BASE_DIR, f'derivatives/results/careful-whisper/{p.dataset}/')

    if p.dataset == 'gigaspeech-m':
        dataset_dir = os.path.join(DATASETS_DIR, 'nlp-datasets', p.dataset.replace('-', '/'))
        cache_dir   = os.path.join(SCRATCH_DIR, 'nlp-datasets', p.dataset.replace('-', '/'))
    else:
        dataset_dir = os.path.join(DATASETS_DIR, 'nlp-datasets', p.dataset)
        cache_dir   = os.path.join(SCRATCH_DIR, 'nlp-datasets', p.dataset)

    print(f'{p.model_name}', flush=True)
    attempt_makedirs(results_dir)

    config_path = os.path.join(os.path.relpath(modeling_dir, os.getcwd()), 'configs')
    cfg, model  = load_model(config_path, p.ckpt_path, p.overrides)

    print(dataset_dir, flush=True)
    print(p.split, flush=True)

    dataset_path = os.path.join(dataset_dir, 'features', 'metadata', 'gpt2-wav2vec2-data2vec')

    metadata_fn = (
        f"metadata_subset-{str(p.subset_percentage).zfill(3)}.json"
        if p.subset_percentage else None
    )

    dataset = AudioTextDataset(
        data_dir=dataset_path,
        split=p.split,
        token_fusion_method='mlp' if p.token_fusion_ckpt else None,
        metadata_file=metadata_fn,
        ckpt_path=p.token_fusion_ckpt,
        shuffle_context_dims=bool(p.shuffle_context_dims),
        context_type=p.context_type,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        collate_fn=audio_text_collator,
        shuffle=False,
    )

    if p.dataset != 'pfka-moth-stories':
        df_results = get_model_results(model, dataloader)
        df_results['model_name'] = p.model_name
        df_results.to_csv(
            os.path.join(results_dir, f'{p.model_name}_test.csv'), index=False
        )
    else:
        dataset._initialize_models()

        TOP_N       = [1, 5]
        WINDOW_SIZE = 25

        out_dir    = os.path.join(BASE_DIR, 'derivatives/model-predictions', p.split,
                                  'careful-whisper', p.model_name,
                                  f'window-size-{str(WINDOW_SIZE).zfill(5)}')
        logits_dir = os.path.join(SCRATCH_DIR, 'derivatives/model-predictions', p.split,
                                  'careful-whisper', p.model_name,
                                  f'window-size-{str(WINDOW_SIZE).zfill(5)}', 'logits')

        attempt_makedirs(out_dir)
        attempt_makedirs(logits_dir)

        df_preproc = pd.read_csv(
            os.path.join(BASE_DIR, 'stimuli/preprocessed/', p.split,
                         f'{p.split}_transcript-preprocessed.csv')
        )
        df_preproc = df_preproc.rename(
            columns={'Word_Written': 'word', 'Punctuation': 'punctuation'}
        )

        print('Loading word models...', flush=True)
        word_models = {
            m: prosody.load_word_model(model_name=m, cache_dir=CACHE_DIR)
            for m in prosody.WORD_MODELS.keys()
        }

        df = prosody.create_results_dataframe()
        df.loc[len(df)] = {'ground_truth_word': df_preproc.iloc[0]['word'].lower()}

        df_stack   = {str(n): [df] for n in TOP_N}
        prev_probs = None

        for i, batch in enumerate(dataloader):
            ground_truth_index = i + 1
            ground_truth_word  = df_preproc.loc[ground_truth_index, 'word']

            with torch.no_grad():
                outputs = model.step(batch=batch)

            logits = outputs['logits'][:, -1, :]

            if df_preproc.loc[ground_truth_index, 'NWP_Candidate']:
                logits_fn = os.path.join(
                    logits_dir,
                    f'{p.split}_window-size-{str(WINDOW_SIZE).zfill(5)}'
                    f'_logits-{str(ground_truth_index).zfill(5)}.pt'
                )
                torch.save(logits, logits_fn)

            probs = F.softmax(logits, dim=-1)

            for n in TOP_N:
                segment_stats = prosody.get_model_statistics(
                    ground_truth_word, probs, dataset.text_tokenizer,
                    prev_probs=prev_probs, word_models=word_models, top_n=n
                )
                df_stack[str(n)].append(segment_stats)

            prev_probs = probs
            print(f'Processed segment {i+1}/{len(dataloader)}', flush=True)

        for n in TOP_N:
            df_results = pd.concat(df_stack[str(n)]).reset_index(drop=True)
            df_results.to_csv(
                os.path.join(
                    out_dir,
                    f'task-{p.split}_window-size-{str(WINDOW_SIZE).zfill(5)}_top-{n}.csv'
                ),
                index=False,
            )


# -----------------------------------------------------------------------
# Analysis subcommand helpers
# -----------------------------------------------------------------------

MODEL_FNAMES = {
    'text':                 '{dataset}_text-careful-whisper_no-xattn_test.csv',
    'audio':                '{dataset}_audio-careful-whisper_causal-xattn_test.csv',
    'audio-shuffled':       '{dataset}_audio-shuffled-careful-whisper_causal-xattn_test.csv',
    'prosody':              '{dataset}_prosody-careful-whisper_causal-xattn_test.csv',
    'prosody-shuffled':     '{dataset}_prosody-shuffled-careful-whisper_causal-xattn_test.csv',
    'audiovisual':          '{dataset}_audiovisual-careful-whisper_causal-xattn_token-fusion-mlp_test.csv',
    'audiovisual-shuffled': '{dataset}_audiovisual-shuffled-careful-whisper_causal-xattn_token-fusion-mlp_test.csv',
}

MODALITIES = ['audio', 'prosody', 'audiovisual']


def load_nll(results_dir, dataset, model_key):
    """Load per-sequence NLL (loss column) for one model variant."""
    fname = MODEL_FNAMES[model_key].format(dataset=dataset)
    fpath = os.path.join(results_dir, dataset, fname)
    if not os.path.exists(fpath):
        raise FileNotFoundError(f'Missing results file: {fpath}')
    return pd.read_csv(fpath)['loss'].values


def bootstrap_ci(x, n_boot=10000, ci=95, seed=42):
    """Return (mean, lower_ci, upper_ci) via bootstrap."""
    rng = np.random.default_rng(seed)
    boot_means = np.array([
        rng.choice(x, size=len(x), replace=True).mean() for _ in range(n_boot)
    ])
    alpha = (100 - ci) / 2
    lower, upper = np.percentile(boot_means, [alpha, 100 - alpha])
    return x.mean(), lower, upper


def compute_information_gain(text_nll, multi_nll, scr_nll):
    """Compute per-sequence information gain quantities.

    delta_I uses the more conservative null (min of text_nll, scr_nll) so it:
      - controls for dimensionality when scrambled helps (I_scr > 0)
      - is bounded by I_total when scrambled hurts (I_scr < 0)
    Equivalently: delta_I = min(I_total, scr_nll - multi_nll) = I_total - max(0, I_scr)

    delta_I_raw is the uncorrected version (scr_nll - multi_nll), which can exceed I_total
    when scrambled features actively hurt performance (I_scr < 0).
    """
    I_total     = text_nll - multi_nll                          # total gain: text → multimodal
    I_scr       = text_nll - scr_nll                            # dimensionality effect (can be negative)
    delta_I_raw = scr_nll  - multi_nll                          # uncorrected ΔI (can exceed I_total)
    delta_I     = np.minimum(text_nll, scr_nll) - multi_nll    # ΔI_controlled = min(I_total, ΔI_raw)
    return I_total, I_scr, delta_I, delta_I_raw


def run_analysis(p):
    results_dir = os.path.join(BASE_DIR, 'derivatives', 'results', 'careful-whisper')
    plots_dir   = os.path.join(BASE_DIR, 'derivatives', 'plots', 'final', 'careful-whisper')
    attempt_makedirs(plots_dir)

    all_rows = []

    for dataset in p.datasets:
        print(f'\n=== {dataset} ===')

        try:
            text_nll = load_nll(results_dir, dataset, 'text')
        except FileNotFoundError as e:
            print(f'  SKIP: {e}')
            continue

        for modality in MODALITIES:
            try:
                multi_nll = load_nll(results_dir, dataset, modality)
                scr_nll   = load_nll(results_dir, dataset, f'{modality}-shuffled')
            except FileNotFoundError as e:
                print(f'  SKIP {modality}: {e}')
                continue

            n = min(len(text_nll), len(multi_nll), len(scr_nll))
            I_total, I_scr, delta_I, delta_I_raw = compute_information_gain(
                text_nll[:n], multi_nll[:n], scr_nll[:n]
            )

            # Write IG columns directly into _test.csv for this modality
            test_fname = MODEL_FNAMES[modality].format(dataset=dataset)
            test_path  = os.path.join(results_dir, dataset, test_fname)
            df_test = pd.read_csv(test_path)
            for col, arr in [('I_total', I_total), ('I_scr', I_scr),
                              ('delta_I', delta_I), ('delta_I_raw', delta_I_raw)]:
                df_test[col] = np.nan
                df_test.loc[:n-1, col] = arr
            df_test.to_csv(test_path, index=False)
            print(f'  IG columns written to {test_path}')

            mean_I_total,  lo_I_total,  hi_I_total  = bootstrap_ci(I_total)
            mean_I_scr,    lo_I_scr,    hi_I_scr    = bootstrap_ci(I_scr)
            mean_delta,    lo_delta,    hi_delta     = bootstrap_ci(delta_I)
            mean_delta_raw,lo_delta_raw,hi_delta_raw = bootstrap_ci(delta_I_raw)

            t_stat, p_val = stats.ttest_1samp(delta_I, popmean=0, alternative='greater')

            print(
                f'  {modality:20s}  '
                f'I_total={mean_I_total:+.4f}  '
                f'I_scr={mean_I_scr:+.4f}  '
                f'ΔI={mean_delta:+.4f}  '
                f't={t_stat:.3f}  p={p_val:.3e}'
            )

            all_rows.append({
                'dataset':         dataset,
                'modality':        modality,
                'n_sequences':     n,
                'mean_I_total':    mean_I_total,
                'lo_I_total':      lo_I_total,
                'hi_I_total':      hi_I_total,
                'mean_I_scr':      mean_I_scr,
                'lo_I_scr':        lo_I_scr,
                'hi_I_scr':        hi_I_scr,
                'mean_delta_I':    mean_delta,
                'lo_delta_I':      lo_delta,
                'hi_delta_I':      hi_delta,
                'se_delta_I':      delta_I.std() / np.sqrt(n),
                'mean_delta_I_raw':mean_delta_raw,
                'lo_delta_I_raw':  lo_delta_raw,
                'hi_delta_I_raw':  hi_delta_raw,
                't_stat':          t_stat,
                'p_value':         p_val,
            })

    if not all_rows:
        print('No results to analyse.')
        sys.exit(0)

    df_summary = pd.DataFrame(all_rows)

    summary_path = os.path.join(results_dir, 'information_gain_summary.csv')
    df_summary.to_csv(summary_path, index=False)
    print(f'\nSummary saved to {summary_path}')

    # Bar chart: I_total, I_scr, ΔI_structured per modality per dataset
    fig, axes = plt.subplots(
        1, len(p.datasets), figsize=(5 * len(p.datasets), 5), sharey=False
    )
    if len(p.datasets) == 1:
        axes = [axes]

    _rocket = sns.color_palette('rocket', 3)
    palette = {'I_total': _rocket[2], 'I_scr': _rocket[1], 'delta_I': _rocket[0]}

    for ax, dataset in zip(axes, p.datasets):
        sub = df_summary[df_summary['dataset'] == dataset]
        if sub.empty:
            ax.set_visible(False)
            continue

        x     = np.arange(len(sub))
        width = 0.25

        for j, (key, label, lo_col, hi_col) in enumerate([
            ('mean_I_total', 'I_total',   'lo_I_total', 'hi_I_total'),
            ('mean_I_scr',   'I_scr',     'lo_I_scr',   'hi_I_scr'),
            ('mean_delta_I', 'ΔI_struct', 'lo_delta_I', 'hi_delta_I'),
        ]):
            vals = sub[key].values
            lo   = sub[lo_col].values
            hi   = sub[hi_col].values
            yerr = np.array([vals - lo, hi - vals])
            ax.bar(
                x + j * width, vals, width,
                label=label if ax == axes[0] else None,
                color=list(palette.values())[j],
                alpha=0.85, yerr=yerr, capsize=3,
                error_kw={'linewidth': 1},
            )

        # Dashed reference lines on ΔI bars showing uncorrected raw delta_I
        raw_vals = sub['mean_delta_I_raw'].values
        for xi, rv in zip(x, raw_vals):
            ax.hlines(rv, xi + 1.5 * width, xi + 2.5 * width,
                      colors='gray', linestyles='--', linewidth=1.5)

        ax.axhline(0, color='k', linewidth=0.8, linestyle='--')
        ax.set_xticks(x + width)
        ax.set_xticklabels(sub['modality'], rotation=20, ha='right')
        ax.set_title(dataset)
        ax.set_ylabel('Information gain (nats)')
        sns.despine(ax=ax)

    from matplotlib.lines import Line2D
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Line2D([0], [0], color='gray', linestyle='--', linewidth=1.5))
    labels.append('ΔI_raw (uncorrected)')
    fig.legend(handles, labels, loc='upper right', frameon=False)
    fig.suptitle('Structured information gain (ΔI = min(CE_text, CE_scr) − CE_multi)', y=1.02)
    plt.tight_layout()

    plot_path = os.path.join(plots_dir, 'information_gain_all-datasets.pdf')
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close('all')
    print(f'Plot saved to {plot_path}')


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='CarefulWhisper results and information gain analysis.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest='subcommand', required=True)

    # ---- results subcommand ----
    res_p = subparsers.add_parser('results', help='Run inference and save per-sequence NLL CSV')
    res_p.add_argument('-d',  '--dataset',             type=str)
    res_p.add_argument('-s',  '--split',               type=str)
    res_p.add_argument('-model_name',                  type=str)
    res_p.add_argument('-ckpt_path',                   type=str)
    res_p.add_argument('-overrides',                   type=str, nargs='+')
    res_p.add_argument('-subset_percentage',           type=int,   default=None)
    res_p.add_argument('-token_fusion_ckpt',           type=str,   default=None)
    res_p.add_argument('-av', '--audiovisual',         type=int,   default=1)
    res_p.add_argument('-o',  '--overwrite',           type=int,   default=0)
    res_p.add_argument('-shuffle_context_dims',        type=int,   default=0,
                       help='Shuffle feature dimensions per token (1=True, 0=False)')
    res_p.add_argument('-context_type',                type=str,   default='audio_features',
                       help='Feature type to use (audio_features, prominence, audiovisual_features)')

    # ---- analysis subcommand ----
    an_p = subparsers.add_parser('analysis', help='Compute information gain from saved NLL CSVs')
    an_p.add_argument('-d', '--datasets', type=str, nargs='+',
                      default=['lrs3', 'avspeech', 'voxceleb2'])
    an_p.add_argument('-s', '--split',    type=str, default='test')

    p = parser.parse_args()

    if p.subcommand == 'results':
        run_results(p)
    else:
        run_analysis(p)

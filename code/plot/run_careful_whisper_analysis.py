import os, sys
import glob
import numpy as np
import pandas as pd
import argparse
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm

sys.path.append('../utils/')

from config import *
from dataset_utils import attempt_makedirs
import careful_whisper_utils as utils

def batch_average(df, batch_size, columns):
    """
    Average a DataFrame into batches of specified size.

    Parameters:
    df (pandas.DataFrame): Input DataFrame to be batched
    batch_size (int): Size of each batch

    Returns:
    pandas.DataFrame: DataFrame with averaged batches
    """
    # Calculate number of complete batches
    n_batches = len(df) // batch_size

    # Handle case where DataFrame length isn't divisible by batch_size
    remainder = len(df) % batch_size

    # If there's no perfect division, we'll need one more batch
    if remainder > 0:
        n_batches += 1

    # Create list to store batch averages
    batch_averages = []

    # Process complete batches
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(df))  # Use min to handle last batch

        # Calculate average for current batch
        batch_avg = df.iloc[start_idx:end_idx]
        batch_avg = batch_avg[columns].mean()
        batch_avg['batch_number'] = i  # Add batch number for reference
        batch_averages.append(batch_avg)

    # Combine all batch averages into a new DataFrame
    return pd.DataFrame(batch_averages)

def change_width(ax, new_value) :

    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)


def get_dataset_models(model_names, datasets, subsets=None):
    """Create model config variations for different subset sizes."""
    variations = {}

    for dataset in datasets:
        for model_name, model_type in model_names.items():

            # Add full dataset version
            if subsets is not None:
                for subset in subsets:
                    # variations[f'{dataset}_{model_name}-subset-{subset:.2f}'] = f'{model_type}'
                    variations[f"{model_name}_subset-{str(subset).zfill(3)}"] = f'{model_type}'

                # Also add base model
                variations[f'{dataset}_{model_name}'] = model_type
            else:
                variations[f'{dataset}_{model_name}'] = model_type

    return variations

MODEL_GROUPS = {
    'audio-main': {
        # f'careful-whisper_causal-xattn': 'Auditory Context',
        # f'prosody-whisper_causal-xattn': 'Prosody Access',
        # f'careful-whisper_no-xattn': 'Sensory Deprived',
    },
    'av-main': {
        f'audiovisual-careful-whisper_causal-xattn_token-fusion-mlp': 'Audiovisual Context',
        f'audio-careful-whisper_causal-xattn': 'Auditory Context',
        f'prosody-careful-whisper_causal-xattn': 'Prosody Access',
        f'text-careful-whisper_no-xattn': 'Sensory Deprived',
    },
    'av-subsets': {
        f'audiovisual-careful-whisper_causal-xattn_token-fusion-mlp': 'Audiovisual Context',
        f'audio-careful-whisper_causal-xattn': 'Auditory Context',
        f'prosody-careful-whisper_causal-xattn': 'Prosody Access',
        f'text-careful-whisper_no-xattn': 'Sensory Deprived',
    }
}

# Maps shuffled model file-name patterns → the display name of their paired real model
SHUFFLED_MODEL_NAMES = {
    'av-main': {
        'audio-shuffled-careful-whisper_causal-xattn':                        'Auditory Context',
        'prosody-shuffled-careful-whisper_causal-xattn':                      'Prosody Access',
        'audiovisual-shuffled-careful-whisper_causal-xattn_token-fusion-mlp': 'Audiovisual Context',
    },
    'av-subsets': {
        'audio-shuffled-careful-whisper_causal-xattn':                        'Auditory Context',
        'prosody-shuffled-careful-whisper_causal-xattn':                      'Prosody Access',
        'audiovisual-shuffled-careful-whisper_causal-xattn_token-fusion-mlp': 'Audiovisual Context',
    },
}

# Short model name shorthands saved to CSV files (display names only used in plots)
DISPLAY_TO_SHORT = {
    'Audiovisual Context': 'AVC',
    'Auditory Context':    'AC',
    'Prosody Access':      'PA',
    'Sensory Deprived':    'SD',
}
SHORT_TO_DISPLAY = {v: k for k, v in DISPLAY_TO_SHORT.items()}

# Per-model result CSV filename patterns (mirrored from run_careful_whisper_results.py)
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

DATASET_HOURS = {
    'voxceleb2': 717,
    'av-combined': 1065,
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # type of analysis we're running --> linked to the name of the regressors
    parser.add_argument('-d', '--datasets', type=str, nargs='+')
    parser.add_argument('-group', '--group', type=str, help="Name of the model grouping")
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-o', '--overwrite', type=int, default=0)
    parser.add_argument('--with_controls', type=int, default=0,
                        help='Overlay gray control (shuffled) bars on accuracy/perplexity/I_total plots (1=True)')
    parser.add_argument('--bits', type=int, default=0,
                        help='Plot information gain in bits instead of nats (1=True)')
    p = parser.parse_args()

    output_dir = os.path.join(BASE_DIR, 'derivatives/results/careful-whisper/')
    plots_dir  = os.path.join(BASE_DIR, 'derivatives/plots/final/careful-whisper/')

    attempt_makedirs(plots_dir)
    attempt_makedirs(output_dir)

    if p.group == 'subsets':
        subsets = np.arange(0.1, 1, 0.1).tolist()
        subsets += (np.logspace(0.3, 1.4, 10) / 100).tolist()
    else:
        subsets = None

    model_names = MODEL_GROUPS[p.group]
    dataset_model_names = get_dataset_models(model_names, p.datasets, subsets)

    # Load results for all models except the yoked models
    # Glob per-dataset to avoid scanning archive/ dirs and unrelated dataset subdirs
    _results_dir = os.path.join(BASE_DIR, 'derivatives/results/careful-whisper')
    results_fns = []
    for _ds in p.datasets:
        results_fns += glob.glob(os.path.join(_results_dir, _ds, '*_test.csv'))

    results_fns = [fn for model_name in dataset_model_names.keys() for fn in results_fns if model_name in fn]

    if p.group != 'subsets':
        results_fns = [fn for fn in results_fns if 'subset' not in fn]

    results_fns = sorted(set(results_fns))

    # Strip dataset prefix so bare model names match MODEL_GROUPS keys
    dataset_model_names = {'_'.join(k.split('_')[1:]): v for k, v in dataset_model_names.items()}
    df_results = []

    for fn in tqdm(results_fns, desc='Loading results'):
        base_name    = os.path.splitext(os.path.basename(fn))[0].split('_')
        dataset_name = base_name[0]

        if 'subset' in p.group:
            subset_idx = next((i for i, s in enumerate(base_name) if s.startswith('subset-')), None)
            if subset_idx is not None:
                model_name = '_'.join(base_name[1:subset_idx])
            else:
                model_name = '_'.join(base_name[1:-1])  # full model: strip dataset + 'test'
        else:
            model_name = '_'.join(base_name[1:-1])  # strip leading dataset + trailing 'test'

        df = pd.read_csv(fn)
        ig_cols_present = [c for c in ['I_total', 'I_scr', 'delta_I', 'delta_I_raw'] if c in df.columns]
        df = batch_average(df, batch_size=p.batch_size, columns=['loss', 'accuracy'] + ig_cols_present)
        df['dataset']    = dataset_name
        df['model_name'] = model_name
        df['perplexity'] = np.exp(df['loss'])

        if 'subset' in p.group:
            subset_name    = base_name[-2]
            current_subset = 100 if 'subset' not in subset_name else int(subset_name.split('-')[-1])
            df['subset']   = current_subset

        df_results.append(df)

    df_results = pd.concat(df_results).reset_index(drop=True)

    # Load shuffled (control) results
    if p.with_controls and p.group in SHUFFLED_MODEL_NAMES:
        shuffled_names = SHUFFLED_MODEL_NAMES[p.group]
        _all_fns = []
        for _ds in p.datasets:
            _all_fns += glob.glob(os.path.join(_results_dir, _ds, '*_test.csv'))

        shuffled_fns = sorted({fn for sname in shuffled_names for fn in _all_fns
                               if sname in fn and os.path.isfile(fn)})
        _df_shuf_list = []
        for fn in shuffled_fns:
            _base = os.path.splitext(os.path.basename(fn))[0].split('_')
            _dataset = _base[0]
            _mname   = '_'.join(_base[1:-1])
            _disp    = shuffled_names.get(_mname)
            if _disp is None:
                continue
            _df = pd.read_csv(fn)
            _ig_cols = [c for c in ['I_total', 'I_scr', 'delta_I', 'delta_I_raw'] if c in _df.columns]
            _df = batch_average(_df, batch_size=p.batch_size, columns=['loss', 'accuracy'] + _ig_cols)
            _df['dataset']    = _dataset
            _df['model_name'] = DISPLAY_TO_SHORT[_disp]
            _df['perplexity'] = np.exp(_df['loss'])
            _df_shuf_list.append(_df)

        if _df_shuf_list:
            df_shuffled = pd.concat(_df_shuf_list).reset_index(drop=True)
        else:
            df_shuffled = pd.DataFrame(columns=['dataset', 'model_name', 'batch_number', 'accuracy', 'perplexity'])
    else:
        df_shuffled = pd.DataFrame(columns=['dataset', 'model_name', 'accuracy', 'perplexity'])

    # Per-(dataset, model) mean accuracy and perplexity
    ordered_accuracy = (
        df_results[['dataset', 'model_name', 'accuracy', 'perplexity']]
        .groupby(['dataset', 'model_name'])
        .mean()
        .reset_index()
    )

    # Shuffled / null models (may be absent in multi-dataset glob; fall back to df_shuffled)
    null_models = ordered_accuracy['model_name'].str.contains('shuffle')
    if null_models.any():
        accuracy_chance   = ordered_accuracy.loc[null_models, 'accuracy'].max()
        perplexity_chance = ordered_accuracy.loc[null_models, 'perplexity'].min()
    elif not df_shuffled.empty:
        accuracy_chance   = df_shuffled['accuracy'].max()
        perplexity_chance = df_shuffled['perplexity'].min()
    else:
        accuracy_chance   = np.nan
        perplexity_chance = np.nan

    # Order non-null models by mean accuracy across datasets (low → high)
    ordered_models_raw = (
        ordered_accuracy[~null_models]
        .groupby('model_name')[['accuracy']]
        .mean()
        .sort_values('accuracy')
        .index.tolist()
    )
    ordered_accuracy = ordered_accuracy[~null_models]

    # Map internal names → display names, then → shorthands for CSV storage
    ordered_models_display = [dataset_model_names[m] for m in ordered_models_raw]
    ordered_models_short   = [DISPLAY_TO_SHORT[d] for d in ordered_models_display]

    df_results = df_results[~df_results['model_name'].str.contains('shuffle')]
    df_results['model_name'] = df_results['model_name'].apply(
        lambda x: DISPLAY_TO_SHORT[dataset_model_names[x]]
    )
    df_results = df_results.sort_values(by=['dataset', 'accuracy'], ascending=[True, False])

    if 'main' in p.group:

        dataset_order = p.datasets if p.datasets else []

        if not df_results.empty:

            # Derive I_total for shuffled rows: text_loss - shuffled_loss
            # (IG columns for main models already come from _test.csv via batch_average above)
            if not df_shuffled.empty:
                _text_loss = (
                    df_results[df_results['model_name'] == 'SD'][['dataset', 'batch_number', 'loss']]
                    .rename(columns={'loss': '_text_loss'})
                )
                df_shuffled = df_shuffled.merge(_text_loss, on=['dataset', 'batch_number'], how='left')
                df_shuffled['I_total'] = df_shuffled['_text_loss'] - df_shuffled['loss']
                df_shuffled = df_shuffled.drop(columns=['_text_loss'])

            # Combine main and shuffled results into a single CSV with is_shuffled column
            df_results['is_shuffled'] = False
            if not df_shuffled.empty:
                df_shuffled['is_shuffled'] = True
                df_save = pd.concat([df_results, df_shuffled], ignore_index=True)
            else:
                df_save = df_results.copy()

            # Save to careful-whisper results directory
            if len(p.datasets) > 1:
                csv_out_fn = os.path.join(output_dir, f'all-dataset-{p.group}_careful-whisper_all-results_batch-size-{p.batch_size}.csv')
            else:
                csv_out_fn = os.path.join(output_dir, f'{p.datasets[0]}-{p.group}_careful-whisper_all-results_batch-size-{p.batch_size}.csv')
            out_fn = csv_out_fn

            df_save.to_csv(out_fn, index=False)

            # Reload and split for plotting (remap shorthands → display names)
            df_loaded = pd.read_csv(out_fn)
            df_results = df_loaded[~df_loaded['is_shuffled']].copy()
            df_results['model_name'] = df_results['model_name'].map(SHORT_TO_DISPLAY)

            df_shuffled_plot = df_loaded[df_loaded['is_shuffled']].copy()
            df_shuffled_plot['model_name'] = df_shuffled_plot['model_name'].map(SHORT_TO_DISPLAY)
            if not df_shuffled_plot.empty:
                df_shuffled_plot = (
                    df_shuffled_plot[['dataset', 'model_name', 'accuracy', 'perplexity']]
                    .groupby(['dataset', 'model_name']).mean().reset_index()
                )

            # Use display names for plot ordering
            ordered_models = ordered_models_display

            #################################################
            ########## Plot 1: Plot model accuracy  #########
            #################################################

            plt.figure(figsize=(6, 5))
            sns.set(style='white')

            ax = sns.barplot(data=df_results, x="dataset", y="accuracy", hue="model_name",
                palette="rocket", alpha=0.8, order=dataset_order, hue_order=ordered_models, legend=True)

            plt.xlabel('Model')
            plt.ylabel('Accuracy (Percent Correct)')
            plt.title(f'All models – test set accuracy')
            plt.xticks(rotation=45, ha='right')

            if p.group == 'av-main':
                plt.ylim([0.15, 0.425])
            elif p.group == 'audio-main':
                plt.ylim([0.1, 0.325])

            sns.despine()

            # Overlay gray control bars in front (accuracy: colored behind, gray in front)
            if p.with_controls and not df_shuffled_plot.empty:
                _n_models = len(ordered_models)
                _bar_width = list(ax.patches)[0].get_width()
                for _x_idx, _dataset in enumerate(dataset_order):
                    for _hue_idx, _mdisp in enumerate(ordered_models):
                        if _mdisp == 'Sensory Deprived':
                            continue
                        _bar_center = _x_idx + (_hue_idx - (_n_models - 1) / 2.0) * _bar_width
                        _row = df_shuffled_plot[
                            (df_shuffled_plot['model_name'] == _mdisp) &
                            (df_shuffled_plot['dataset']    == _dataset)
                        ]
                        if _row.empty:
                            continue
                        ax.bar(_bar_center, _row['accuracy'].values[0], _bar_width,
                               color='gray', alpha=1.0, zorder=2)

            plt.tight_layout()

            if len(p.datasets) > 1:
                out_fn = os.path.join(plots_dir, f"all-dataset-{p.group}_careful-whisper_accuracy.pdf")
            else:
                out_fn = os.path.join(plots_dir, f"{p.datasets[0]}-{p.group}_careful-whisper_accuracy.pdf")

            plt.savefig(out_fn, bbox_inches='tight', dpi=600)
            plt.close('all')

            ###################################################
            ########## Plot 2: Plot model perplexity  #########
            ###################################################

            plt.figure(figsize=(6, 5))
            sns.set(style='white')

            ax = sns.barplot(data=df_results, x="dataset", y="perplexity", hue="model_name",
                palette="rocket", alpha=0.8, order=dataset_order, hue_order=ordered_models, legend=True)

            plt.xlabel('Model')
            plt.ylabel('Perplexity')
            plt.title(f'All models – test set perplexity')
            plt.xticks(rotation=45, ha='right')

            if p.group == 'av-main':
                plt.ylim(0, 200)
            elif p.group == 'audio-main':
                plt.ylim(0, 350)

            sns.despine()

            # Overlay gray control bars behind (perplexity: colored in front, gray behind)
            if p.with_controls and not df_shuffled_plot.empty:
                _n_models = len(ordered_models)
                _bar_width = list(ax.patches)[0].get_width()
                for _x_idx, _dataset in enumerate(dataset_order):
                    for _hue_idx, _mdisp in enumerate(ordered_models):
                        if _mdisp == 'Sensory Deprived':
                            continue
                        _bar_center = _x_idx + (_hue_idx - (_n_models - 1) / 2.0) * _bar_width
                        _row = df_shuffled_plot[
                            (df_shuffled_plot['model_name'] == _mdisp) &
                            (df_shuffled_plot['dataset']    == _dataset)
                        ]
                        if _row.empty:
                            continue
                        ax.bar(_bar_center, _row['perplexity'].values[0], _bar_width,
                               color='gray', alpha=1.0, zorder=0)

            plt.tight_layout()

            if len(p.datasets) > 1:
                out_fn = os.path.join(plots_dir, f"all-dataset-{p.group}_careful-whisper_perplexity.pdf")
            else:
                out_fn = os.path.join(plots_dir, f"{p.datasets[0]}-{p.group}_careful-whisper_perplexity.pdf")

            plt.savefig(out_fn, bbox_inches='tight', dpi=600)
            plt.close('all')

        #####################################################
        ########## Plot 3: ΔI structured info gain ##########
        #####################################################

        # Build IG plot frame from the merged all-results CSV (non-shuffled, non-SD rows)
        df_loaded_main = pd.read_csv(csv_out_fn)
        df_ig_plot = df_loaded_main[
            ~df_loaded_main['is_shuffled'] &
            (df_loaded_main['model_name'] != 'SD') &
            df_loaded_main['I_total'].notna()
        ].copy()

        if not df_ig_plot.empty:
            _bits = bool(p.bits)
            _scale = utils.NATS_TO_BITS if _bits else 1.0
            _unit  = 'bits' if _bits else 'nats'

            # Map shorthands → display names for the hue axis
            ig_modality_order = [SHORT_TO_DISPLAY[s] for s in ['PA', 'AC', 'AVC'] if s in df_ig_plot['model_name'].unique()]
            df_ig_plot['modality'] = df_ig_plot['model_name'].map(SHORT_TO_DISPLAY)

            if len(p.datasets) > 1:
                ig_prefix = os.path.join(plots_dir, f"all-dataset-{p.group}_careful-whisper")
            else:
                ig_prefix = os.path.join(plots_dir, f"{p.datasets[0]}-{p.group}_careful-whisper")

            def _plot_ig(y_col, ylabel, title, null_overlay=False):
                """Bar plot for one IG quantity using per-batch data."""
                df_p = df_ig_plot.copy()
                df_p[y_col] = df_p[y_col] * _scale

                plt.figure(figsize=(6, 5))
                sns.set(style='white')
                ax = sns.barplot(
                    data=df_p, x='dataset', y=y_col, hue='modality',
                    palette='rocket', alpha=0.8,
                    order=dataset_order, hue_order=ig_modality_order,
                    errorbar='se',
                )

                # Gray null (shuffled) overlay for I_total plot
                if null_overlay and p.with_controls:
                    df_shuf_ig = df_loaded_main[
                        df_loaded_main['is_shuffled'] &
                        (df_loaded_main['model_name'] != 'SD') &
                        df_loaded_main['I_total'].notna()
                    ].copy()
                    if not df_shuf_ig.empty:
                        df_shuf_ig['modality'] = df_shuf_ig['model_name'].map(SHORT_TO_DISPLAY)
                        df_shuf_mean = (
                            df_shuf_ig.groupby(['dataset', 'modality'])[y_col]
                            .mean().reset_index()
                        )
                        _n = len(ig_modality_order)
                        _bw = list(ax.patches)[0].get_width()
                        for _xi, _ds in enumerate(dataset_order):
                            for _hi, _mod in enumerate(ig_modality_order):
                                _bc = _xi + (_hi - (_n - 1) / 2.0) * _bw
                                _row = df_shuf_mean[
                                    (df_shuf_mean['dataset'] == _ds) &
                                    (df_shuf_mean['modality'] == _mod)
                                ]
                                if _row.empty:
                                    continue
                                ax.bar(_bc, _row[y_col].values[0] * _scale, _bw,
                                       color='gray', alpha=1.0, zorder=2)

                if _bits:
                    ylabel = ylabel.replace('(nats)', '(bits)').replace('nats', 'bits')
                ax.set_xlabel('Dataset')
                ax.set_ylabel(ylabel)
                ax.set_title(title)
                ax.legend(frameon=False)
                plt.xticks(rotation=20, ha='right')
                sns.despine()
                plt.tight_layout()

            # ---- Plot 3a: ΔI controlled ----
            _plot_ig('delta_I',
                     ylabel=f'Structured information gain ΔI_controlled ({_unit})',
                     title='Structured information gain by modality')
            plt.savefig(f"{ig_prefix}_delta-I.pdf", bbox_inches='tight', dpi=600)
            plt.close('all')

            # ---- Plot 3b: I_scrambled ----
            _plot_ig('I_scr',
                     ylabel=f'I_scrambled ({_unit})',
                     title='Scrambled information by modality')
            plt.savefig(f"{ig_prefix}_I-scrambled.pdf", bbox_inches='tight', dpi=600)
            plt.close('all')

            # ---- Plot 3c: I_total (with optional gray shuffled control bar) ----
            _plot_ig('I_total',
                     ylabel=f'I_total ({_unit})',
                     title='Total information by modality',
                     null_overlay=True)
            plt.savefig(f"{ig_prefix}_I-total.pdf", bbox_inches='tight', dpi=600)
            plt.close('all')

    ###################################################
    ############ Subset comparison plots ##############
    ###################################################

    elif 'subsets' in p.group:

        total_hours = DATASET_HOURS[p.datasets[0]]

        # Map shorthands back to display names for utils functions
        df_results_disp = df_results.copy()
        df_results_disp['model_name'] = df_results_disp['model_name'].map(SHORT_TO_DISPLAY)

        # Find equivalent points and ratios
        df_comparisons, curves = utils.find_all_model_comparisons(
            df_results_disp,
            main_models=['Audiovisual Context', 'Auditory Context', 'Prosody Access'],
            comparison_model=['Sensory Deprived'],
            kind='power',
            group=True,
            stabilization_method='huber',
        )

        dfs = []

        for i, df in df_comparisons.groupby('true_subset'):
            df['hours'] = (i / 100) * total_hours
            dfs.append(df)

        df_comparisons = pd.concat(dfs).reset_index(drop=True)

        # Map display names → shorthands for CSV storage
        for col in ['main_model', 'comparison_model']:
            if col in df_comparisons.columns:
                df_comparisons[col] = df_comparisons[col].map(
                    lambda x: DISPLAY_TO_SHORT.get(x, x)
                )

        out_fn = os.path.join(output_dir, 'all-subsets_careful-whisper_model-comparisons.csv')
        df_comparisons.to_csv(out_fn, index=False)

        df_comparisons = pd.read_csv(out_fn)

        # Remap shorthands → display names for plotting
        for col in ['main_model', 'comparison_model']:
            if col in df_comparisons.columns:
                df_comparisons[col] = df_comparisons[col].map(
                    lambda x: SHORT_TO_DISPLAY.get(x, x)
                )

        df_visual = df_comparisons[df_comparisons['true_subset'] >= 5]
        ax = utils.plot_all_comparisons(df_visual, 'Sensory Deprived', x_axis='hours', palette='rocket', remove_outliers=False)

        out_fn = os.path.join(plots_dir, f"all-subsets_careful-whisper_joint-plot.pdf")
        plt.savefig(out_fn, bbox_inches='tight', dpi=600)
        plt.close('all')

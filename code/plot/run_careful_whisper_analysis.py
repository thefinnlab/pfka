import os, sys
import glob
import numpy as np
import pandas as pd
import argparse
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
        # f'careful-whisper_causal-xattn': 'AudioXAttn',
        # f'prosody-whisper_causal-xattn': 'ProsodyXAttn',
        # f'careful-whisper_no-xattn': 'GPT2',
    },
    'av-main': {
        f'audiovisual-careful-whisper_causal-xattn_token-fusion-mlp': 'AudioVisualXAttn',
        f'audio-careful-whisper_causal-xattn': 'AudioXAttn',
        f'prosody-careful-whisper_causal-xattn': 'ProsodyXAttn',
        f'text-careful-whisper_no-xattn': 'GPT2',
    },
    'av-subsets': {
        f'audiovisual-careful-whisper_causal-xattn': 'AudioVisualXAttn',
        f'audio-careful-whisper_causal-xattn': 'AudioXAttn',
        f'prosody-careful-whisper_causal-xattn': 'ProsodyXAttn',
        f'text-careful-whisper_no-xattn': 'GPT2',
    }
}

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
    p = parser.parse_args()

    results_dir = os.path.join(BASE_DIR, 'derivatives/results/behavioral/')
    output_dir = os.path.join(BASE_DIR, 'derivatives/results/careful-whisper/')
    plots_dir = os.path.join(BASE_DIR, 'derivatives/plots/final/careful-whisper/')

    attempt_makedirs(plots_dir)

    if p.group == 'subsets':
        subsets = np.arange(0.1, 1, 0.1).tolist()
        subsets += (np.logspace(0.3, 1.4, 10) / 100).tolist()
    else:
        subsets = None

    model_names = MODEL_GROUPS[p.group]
    dataset_model_names = get_dataset_models(model_names, p.datasets, subsets)

    # Load results for all models except the yoked models
    if len(p.datasets) > 1:
        results_fns = glob.glob(os.path.join(BASE_DIR, f'derivatives/results/careful-whisper/*/*'))
        results_fns = [fn for model_name in dataset_model_names.keys() for fn in results_fns if model_name in fn]
        
        if p.group != 'subsets':
            results_fns = [fn for fn in results_fns if 'subset' not in fn]
        
    else:
        results_fns = glob.glob(os.path.join(BASE_DIR, f'derivatives/results/careful-whisper/{p.datasets[0]}/*'))

    results_fns = sorted(results_fns)

    # # now remove dataset names for dataset model names
    # dataset_model_names = {'_'.join(k.split('_')[1:]): v for k, v in dataset_model_names.items()}
    # df_results = []

    # # Load the data and note the dataset name
    # for i, fn in tqdm(enumerate(results_fns), desc='Loading results'):

    #     base_name = os.path.splitext(os.path.basename(fn))[0]
    #     base_name = base_name.split('_')

    #     dataset_name = base_name[0]

    #     if 'subset' in p.group:
    #         model_name = '_'.join(base_name[1:3])
    #     else:
    #         model_name = '_'.join(base_name[1:-1])
        
    #     df = pd.read_csv(fn)
    #     df = batch_average(df, batch_size=p.batch_size, columns=['loss', 'accuracy'])

    #     df['dataset'] = dataset_name
    #     df['model_name'] = model_name
    #     df['perplexity'] = np.exp(df['loss'])

    #     if 'subset' in p.group:
    #         subset_name = base_name[-2]

    #         if 'subset' not in subset_name:
    #             # continue
    #             current_subset = 100
    #         else:
    #             current_subset = subset_name.split('-')[-1]
            
    #         df['subset'] = int(current_subset)

    #     df_results.append(df)
    
    # df_results = pd.concat(df_results).reset_index(drop=True)

    # # Get order of models by binary accuracy
    # ordered_accuracy = df_results.loc[:,['dataset', 'model_name', 'accuracy', 'perplexity']] \
    #     .groupby(['dataset', 'model_name']) \
    #     .mean() \
    #     .reset_index()

    # # get max chance of null models
    # null_models = ordered_accuracy['model_name'].str.contains('shuffle')
    # accuracy_chance = ordered_accuracy.loc[null_models, 'accuracy'].max()
    # perplexity_chance = ordered_accuracy.loc[null_models, 'perplexity'].min()

    # # Get overall model ordering (averaged across datasets)
    # ordered_models = ordered_accuracy[~null_models] \
    #     .groupby('model_name')[['accuracy']] \
    #     .mean() \
    #     .sort_values(by='accuracy') \
    #     .index.tolist()
        
    # # Filter out null models but keep dataset information for bar labels
    # ordered_accuracy = ordered_accuracy[~null_models]

    # # Sort by dataset and accuracy to match the bar order
    # ordered_accuracy = ordered_accuracy.sort_values(by=['dataset', 'accuracy']) #.reset_index(drop=True)

    # dataset_order = ordered_accuracy['dataset'].unique()
    # model_order = ordered_accuracy['model_name'].unique()

    # # Create a multi-index DataFrame to ensure values are in the right order
    # ordered_values = []
    # for model in model_order:
    #     for dataset in dataset_order:
    #         row = ordered_accuracy[
    #             (ordered_accuracy['dataset'] == dataset) & 
    #             (ordered_accuracy['model_name'] == model)
    #         ].iloc[0]

    #         acc = row['accuracy']
    #         ppl = row['perplexity']
    #         ordered_values.append((acc, ppl))

    # ordered_acc_values, ordered_ppl_values = zip(*ordered_values)

    # # remove the null models
    # df_results = df_results[~df_results['model_name'].str.contains('shuffle')]
    # df_results['model_name'] = df_results['model_name'].apply(lambda x: dataset_model_names[x])
    # ordered_models = [dataset_model_names[model] for model in ordered_models]

    # df_results = df_results.sort_values(by=['dataset', 'accuracy'], ascending=[True, False])

    if 'main' in p.group:

        # Save to a csv file
        if len(p.datasets) > 1:
            out_fn = os.path.join(results_dir, f'all-dataset-{p.group}_careful-whisper_all-results_batch-size-{p.batch_size}.csv')
        else:
            out_fn = os.path.join(results_dir, f'{p.datasets[0]}-{p.group}_careful-whisper_all-results_batch-size-{p.batch_size}.csv')

        df_results.to_csv(out_fn, index=False)
        df_results = pd.read_csv(out_fn)

        #################################################
        ########## Plot 1: Plot model accuracy  #########
        #################################################

        plt.figure(figsize=(6, 5))
        sns.set(style='white')
        
        ax = sns.barplot(data=df_results, x="dataset", y="accuracy", hue="model_name",  # Add hue parameter
            palette="rocket", alpha=0.8, hue_order=ordered_models, legend=True)

        # change_width(ax, 0.5 / len(p.datasets))

        plt.xlabel('Model')
        plt.ylabel('Accuracy (Percent Correct)')

        plt.title(f'All models – test set accuracy')
        plt.xticks(rotation=45, ha='right')

        if p.group == 'av-main':

            plt.ylim([0.15, 0.425])
        elif p.group == 'audio-main':
            plt.ylim([0.1, 0.325])
        # if p.dataset == 'helsinki':
        #     plt.ylim([0.18, 0.26])
        # elif p.dataset == 'gigaspeech':
        #     plt.ylim([0.18, 0.26])

        plt.axhline(y=accuracy_chance, color='k', linestyle='--')
        sns.despine()

        # Add text labels on top of each bar
        for patch, acc in zip(ax.patches, ordered_acc_values):  # Changed variable name from p to patch
            height = patch.get_height()
            ax.text(
                patch.get_x() + patch.get_width() / 2.,
                height + 0.008,
                f'{acc:.3f}',
                ha='center',
                va='bottom'
            )

        plt.tight_layout()
        # Save to a csv file

        if len(p.datasets) > 1:
            out_fn = os.path.join(plots_dir, f"all-dataset-{p.group}_careful-whisper_accuracy.pdf")
        else:
            out_fn = os.path.join(plots_dir, f"{p.datasets[0]}-{p.group}_careful-whisper_accuracy.pdf")
            
        plt.savefig(out_fn, bbox_inches='tight', dpi=600)
        plt.close('all')

        ###################################################
        ########## Plot 2: Plot model perplexity  #########
        ###################################################

        plt.figure(figsize=(6,5))
        sns.set(style='white')
        
        ax = sns.barplot(data=df_results, x="dataset", y="perplexity", hue="model_name",
            palette="rocket", alpha=0.8, hue_order=ordered_models, legend=True)

        # change_width(ax, 0.5 / len(p.datasets))

        plt.xlabel('Model')
        plt.ylabel('Perplexity')

        plt.title(f'All models – test set perplexity')
        plt.xticks(rotation=45, ha='right')

        if p.group == 'av-main':
            plt.ylim(0, 200)
        elif p.group == 'audio-main':
            plt.ylim(0, 350)

        plt.axhline(y=perplexity_chance, color='k', linestyle='--')
        sns.despine()

        # Add text labels on top of each bar
        for patch, acc in zip(ax.patches, ordered_ppl_values):  # Changed variable name from p to patch
            height = patch.get_height()
            ax.text(
                patch.get_x() + patch.get_width() / 2.,
                height + 15,
                f'{acc:.2f}',
                ha='center',
                va='bottom'
            )

        plt.tight_layout()


        if len(p.datasets) > 1:
            out_fn = os.path.join(plots_dir, f"all-dataset-{p.group}_careful-whisper_perplexity.pdf")
        else:
            out_fn = os.path.join(plots_dir, f"{p.datasets[0]}-{p.group}_careful-whisper_perplexity.pdf")

        plt.savefig(out_fn, bbox_inches='tight', dpi=600)
        plt.close('all')

    ###################################################
    ############ Subset comparison plots ##############
    ###################################################

    elif 'subsets' in p.group:

        # total_hours = DATASET_HOURS[p.datasets[0]]

        # # Find equivalent points and ratios
        # df_comparisons, curves = utils.find_all_model_comparisons(
        #     df_results, 
        #     main_models=['AudioVisualXAttn', 'AudioXAttn', 'ProsodyXAttn'],
        #     comparison_model=['GPT2'], 
        #     kind='power',
        #     group=True,
        #     stabilization_method='huber',
        # )

        # dfs = []

        # for i, df in df_comparisons.groupby('true_subset'):
        #     df['hours'] = (i / 100) * total_hours
        #     dfs.append(df)
            
        # df_comparisons = pd.concat(dfs).reset_index(drop=True)
            
        out_fn = os.path.join(results_dir, 'all-subsets_careful-whisper_model-comparisons.csv')
        # df_comparisons.to_csv(out_fn, index=False)

        df_comparisons = pd.read_csv(out_fn)

        df_visual = df_comparisons[df_comparisons['true_subset'] >= 5]
        ax = utils.plot_all_comparisons(df_visual, 'GPT2', x_axis='hours', palette='rocket', remove_outliers=False)
 
        out_fn = os.path.join(plots_dir, f"all-subsets_careful-whisper_joint-plot.pdf")
        plt.savefig(out_fn, bbox_inches='tight', dpi=600)
        plt.close('all')
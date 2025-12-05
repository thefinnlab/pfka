import os, sys
import glob
import argparse
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

sys.path.append('../utils/')

from config import *

from tommy_utils import nlp
from dataset_utils import attempt_makedirs
import plotting_utils as utils

FIGSIZE = (6,5)


def create_modality_contrasts(df, modality_pairs, variables):
    """
    Create contrasts for multiple variables across modality pairs.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with columns ['task', 'model_name', 'modality', *variables].
    modality_pairs : list of tuple
        List of modality pairs to compare, e.g. [('text', 'speech'), ('image', 'video')].
    variables : list of str
        Variables to compute contrasts for.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with contrasts for all variables and modality pairs.
    """
    results = []

    for (task, model_name), df_group in df.groupby(['task', 'model_name']):
        contrasts = {}

        for var in variables:
            for mod1, mod2 in modality_pairs:
                df_mod1 = df_group[df_group['modality'] == mod1].reset_index(drop=True)
                df_mod2 = df_group[df_group['modality'] == mod2].reset_index(drop=True)

                if len(df_mod1) == 0 or len(df_mod2) == 0:
                    continue  # skip if missing modality

                diff = (df_mod2[var] - df_mod1[var]).to_numpy()
                contrasts[f"{mod1}-{mod2}_{var}"] = diff

        if contrasts:
            df_diff = pd.DataFrame(contrasts)
            df_diff["task"] = task
            df_diff["model_name"] = model_name
            results.append(df_diff)

    return pd.concat(results).reset_index(drop=True)


def plot_contrasts(df_contrasts, modality_pairs, variables, plots_dir, models_order, N_MODELS):
    """
    Plot contrasts for multiple variables and modality pairs.
    """
    for var in variables:
        for mod1, mod2 in modality_pairs:
            contrast_col = f"{mod1}-{mod2}_{var}"

            if contrast_col not in df_contrasts.columns:
                continue

            cmap = utils.create_colormap(dtype=contrast_col.split('_')[0], N_MODELS=N_MODELS)

            fig, ax = plt.subplots(figsize=(6, 4.5))

            sns.pointplot(
                df_contrasts,
                x=contrast_col, y="model_name",
                color="black", markersize=4, linestyles="none",
                order=models_order, errorbar="se", ax=ax
            )

            if var == 'kl_divergence':
                plt.xlim([-1, 1])
            elif var == 'jensenshannon_dist':
                plt.xlim([-0.05, 0.05])
            elif var == 'earthmovers_dist':
                plt.xlim([-0.0005, 0.0005])

            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()

            # Gradient background
            gradient_rect = patches.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin,
                transform=ax.transData, color="white", alpha=0.2
            )
            ax.add_patch(gradient_rect)

            ax.imshow(
                np.linspace(0, 1, 256).reshape(1, -1),
                aspect="auto", extent=(xmin, xmax, ymin, ymax),
                origin="lower", cmap=cmap, alpha=0.75
            )

            ax.vlines(0, ymin=ymin, ymax=ymax, linestyle="--", color="0.5", linewidth=2)

            fname = f"contrast_{var}_{mod1}-{mod2}.pdf"
            plt.savefig(os.path.join(plots_dir, fname), bbox_inches="tight", dpi=600)
            plt.close(fig)

def split_model_name(df, input_col, dataset_col='dataset', model_col='model_name'):
    """
    Split a column's strings on the first underscore into two columns.
    
    Args:
        df: pandas DataFrame
        input_col: str, name of column to split
        dataset_col: str, name for the new dataset column (default: 'dataset')
        model_col: str, name for the remaining text column (default: 'model_name')
    
    Returns:
        DataFrame with new columns added
    """
    # Create a mask for rows that contain underscore
    has_underscore = df[input_col].str.contains('_', na=False)
    
    # Initialize new columns
    df[dataset_col] = None
    df[model_col] = df[input_col]
    
    # Only split for rows that have underscore
    split_data = df.loc[has_underscore, input_col].str.split('_', n=1, expand=True)
    
    # Update values only for rows with underscore
    df.loc[has_underscore, dataset_col] = split_data[0]
    df.loc[has_underscore, model_col] = split_data[1]
    
    # Drop input column if it's different from model_col
    if input_col != model_col:
        df = df.drop(input_col, axis=1)
    
    return df

MODEL_NAME_MAPPING = {
    'careful-whisper_audio-token-fusion': 'AudioFusion',
    'careful-whisper_causal-xattn': 'AudioXAttn',
    'prosody-whisper_token-fusion': 'ProsodyFusion',
    'prosody-whisper_causal-xattn': 'ProsodyXAttn',
    'careful-whisper_no-xattn': 'NoXAttn',
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # type of analysis we're running --> linked to the name of the regressors
    parser.add_argument('-task_list', '--task_list', type=str, nargs='+', default=['black', 'wheretheressmoke', 'howtodraw'])
    parser.add_argument('-word_model', '--word_model', type=str, default='fasttext')
    parser.add_argument('-m', '--model_name', type=str, default='gpt2-xl')
    parser.add_argument('-window', '--window_size', type=int, default=25)
    parser.add_argument('-modality_list', '--modality_list', type=str, nargs='+', default=['video', 'audio', 'text']) 
    parser.add_argument('-careful_whisper', '--careful_whisper', type=int, default=0)
    parser.add_argument('-drop_shortened', '--drop_shortened', type=int, default=0)
    parser.add_argument('-o', '--overwrite', type=int, default=0)
    p = parser.parse_args()

    if len(p.modality_list) == 3:
        CMAP_DTYPE = 'multimodal'
        FIG_SIZE = (3.5,5)
        modality_pairs = [('video', 'text'), ('audio', 'text'), ('video', 'audio')]
    else:
        CMAP_DTYPE = 'spoken-written'
        FIG_SIZE = (3,5)
        modality_pairs = [('audio', 'text')]

    preproc_dir = os.path.join(BASE_DIR, 'stimuli/preprocessed')

    ###################################
    ### Load results from task list ###
    ###################################

    # Load model names
    if p.careful_whisper:
        results_dir = os.path.join(BASE_DIR, 'derivatives/results/behavioral/')
        plots_dir = os.path.join(BASE_DIR, 'derivatives/plots/final/human-careful-whisper-comparison/')
        models_dir = os.path.join(BASE_DIR, 'derivatives/model-predictions/careful-whisper/')

        models = sorted(glob.glob(os.path.join(BASE_DIR, f'derivatives/model-predictions/{p.task_list[0]}/careful-whisper/*')))
        models = [os.path.basename(model) for model in models]
        model_names = ' '.join(models)

        print (f'Loading the following models')
        print (f'Careful Whisper models: {models}')
    else:

        if p.drop_shortened:
            results_dir = os.path.join(BASE_DIR, 'derivatives/results/behavioral-shortened-removed/')
            plots_dir = os.path.join(BASE_DIR, 'derivatives/plots/final-shortened-removed/human-llm-comparison/')
        else:
            results_dir = os.path.join(BASE_DIR, 'derivatives/results/behavioral/')
            plots_dir = os.path.join(BASE_DIR, 'derivatives/plots/final/human-llm-comparison/')
        
        models_dir = os.path.join(BASE_DIR, 'derivatives/model-predictions/')

        MLM_MODELS = list(nlp.MLM_MODELS_DICT.keys())[1:]
        CLM_MODELS = list(nlp.CLM_MODELS_DICT.keys()) 
        model_names = CLM_MODELS + MLM_MODELS
        
        remove_models = ['qwen3-8B', 'llama3.1-8B']
        model_names = [model for model in model_names if model not in remove_models]
        # model_names = ' '.join(model_names)

        N_MODELS = len(model_names)

        print (f'Loading the following models')
        print (f'MLM models: {MLM_MODELS}')
        print (f'CLM models: {CLM_MODELS}')

    attempt_makedirs(plots_dir)
    
    ###################################
    ### Load results from task list ###
    ###################################

    df_results = []

    for task in p.task_list:
        if p.careful_whisper:
            results_fn = os.path.join(results_dir, f'task-{task}_group-analyzed-behavior_window-size-{str(p.window_size).zfill(5)}_human-careful-whisper-lemmatized.csv')
        else:
            results_fn = os.path.join(results_dir, f'task-{task}_group-analyzed-behavior_window-size-{str(p.window_size).zfill(5)}_human-model-lemmatized.csv')
        df_task = pd.read_csv(results_fn)
        df_task['task'] = task
        df_results.append(df_task)

    # concatenate into one dataframe --> write to file for posterity 
    df_results = pd.concat(df_results).reset_index(drop=True)

    if p.careful_whisper:
        df_results = split_model_name(df_results, input_col='modality', dataset_col='dataset', model_col='modality')
        df_results['modality'] = df_results['modality'].map(MODEL_NAME_MAPPING).fillna(df_results['modality'])
        out_fn = os.path.join(results_dir, f'all-task_group-analyzed-behavior_window-size-{str(p.window_size).zfill(5)}_human-careful-whisper-lemmatized.csv')
    else:
        out_fn = os.path.join(results_dir, f'all-task_group-analyzed-behavior_window-size-{str(p.window_size).zfill(5)}_human-model-lemmatized.csv')
    
    df_results.to_csv(out_fn, index=False)

    # use the accuracy within these results to get the order to plot the models
    ordered_accuracy = utils.get_ordered_accuracy(df_results)

    # always put humans first (audio, text) then CLM models, then MLM models
    if p.careful_whisper:
        models_order = [item for item in ordered_accuracy if item not in p.modality_list]
    else:
        order = p.modality_list + MLM_MODELS
        models_order = [item for item in ordered_accuracy if item not in order]
        models_order = models_order + MLM_MODELS

    human_models_order = p.modality_list + models_order

    #################################################
    #### Plot 1a: binary accuracy for each task #####
    #################################################

    fig, axes = plt.subplots(1,3, figsize=(15, 5))
    axes = axes.flatten()

    cmap = utils.create_colormap(dtype=CMAP_DTYPE, continuous=False, N_MODELS=N_MODELS)
    variable = f'accuracy'

    for ax, (task, df) in zip(axes, df_results.groupby('task')):
        plt.sca(ax)
        ax = utils.plot_bar_results(df, x='modality', y=variable, hue=None, order=human_models_order, cmap=cmap, figsize=None, add_points=False)
        ax.set_ylim([0, 1])
        ax.set_title(f'Task - {task}')
        plt.xticks(rotation=45, ha='right')

        ax.set_xlabel('Modality/Model')
        ax.set_ylabel('Accuracy (Percent Correct)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "individual-task_human-llm-comparison_binary-accuracy.pdf"), bbox_inches='tight', dpi=600)
    plt.close('all')

    #######################################################
    ### Plot 1b: binary accuracy collapsed across tasks ###
    #######################################################

    # plot top word accuracy for humans
    cmap = utils.create_colormap(dtype=CMAP_DTYPE, continuous=False, N_MODELS=N_MODELS)
    ax = utils.plot_bar_results(df_results, x='modality', y='accuracy', hue=None, order=human_models_order, cmap=cmap, figsize=FIGSIZE, add_points=False)
    
    plt.xlabel('Modality/Model')
    plt.xticks(rotation=45, ha='right')
    
    plt.ylim([0, 1])
    plt.ylabel('Accuracy (Percent Correct)')
    plt.title(f'All task - binary accuracy')
    plt.tight_layout()

    plt.savefig(os.path.join(plots_dir, "all-task_human-llm-comparison_binary-accuracy.pdf"), bbox_inches='tight', dpi=600)
    plt.close('all')

    #################################################
    ### Plot 2a: continuous accuracy for each task ###
    #################################################

    fig, axes = plt.subplots(1,3, figsize=(15, 5))
    axes = axes.flatten()

    variable = f'{p.word_model}_top_word_accuracy'

    for ax, (task, df) in zip(axes, df_results.groupby('task')):
        plt.sca(ax)
        ax = utils.plot_bar_results(df, x='modality', y=variable, hue=None, order=human_models_order, cmap=cmap, figsize=None, add_points=False)
        ax.set_ylim([0, 1])
        ax.set_title(f'Task - {task}')
        plt.xticks(rotation=45, ha='right')

        ax.set_xlabel('Modality/Model')
        ax.set_ylabel('Cosine similarity')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "all-task_human-llm-comparison_continuous-accuracy.pdf"), bbox_inches='tight', dpi=600)
    plt.close('all')

    ###########################################################
    ### Plot 2b: continuous accuracy collapsed across tasks ###
    ###########################################################

    # plot top word accuracy for humans
    variable = f'{p.word_model}_top_word_accuracy'

    cmap = utils.create_colormap(dtype=CMAP_DTYPE, continuous=False, N_MODELS=N_MODELS)
    ax = utils.plot_bar_results(df_results, x='modality', y=variable, hue=None, order=human_models_order, cmap=cmap, figsize=FIGSIZE, add_points=False)

    plt.xlabel('Task')
    plt.xticks(rotation=45, ha='right')

    plt.ylim([0, 1.0])
    plt.ylabel('Cosine similarity')
    
    plt.title(f'All task - continuous accuracy')
    plt.tight_layout()

    plt.savefig(os.path.join(plots_dir, "individual-task_human-llm-comparison_continuous-accuracy.pdf"), bbox_inches='tight', dpi=600)
    plt.close('all')

    ###########################################################
    ######## Plot 2c: continuous accuracy each quadrant #######
    ###########################################################

    # This places the bars in the order of the quadrant plot
    accuracy_entropy_groups = [
        'high-low',
        'high-high',
        'low-low',
        'low-high'
    ]

    variable = f'{p.word_model}_top_word_accuracy'
    N_MODELS = len(models_order)

    cmap = utils.create_colormap(dtype=CMAP_DTYPE, continuous=False, N_MODELS=N_MODELS)

    fig, axes = plt.subplots(2,2, figsize=(FIGSIZE[0]*2, FIGSIZE[1]*2))
    axes = axes.flatten()

    for ax, group in zip(axes, accuracy_entropy_groups):

        accuracy_group, entropy_group = group.split('-')

        accuracy_mask = df_results['accuracy_group'] == accuracy_group
        entropy_mask = df_results['entropy_group'] == entropy_group
        df_group = df_results[accuracy_mask & entropy_mask]

        utils.plot_bar_results(df_group, x='modality', y=variable, hue=None, order=human_models_order, cmap=cmap, figsize=None, add_points=False, ax=ax)
        ax.set_ylim([0, 1])
        # ax.set_title(f'Task - {task}')
        # plt.xticks(rotation=45, ha='right')
        # Rotate x-axis labels for this axis
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
        # sns.barplot(df_group, x='modality', y='fasttext_top_word_accuracy',  ax=ax, order=human_models_order)
        ax.set_title(f'{accuracy_group} accuracy, {entropy_group} entropy')
        # ax.set_xlabel('Average Accuracy')
        ax.set_ylabel(variable)
        sns.despine()

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "all-task_human-llm-comparison_continuous-accuracy-quadrant-bars.pdf"), bbox_inches='tight', dpi=600)
    plt.close('all')

    #################################################
    #### Plot 3: KL Divergence of model & humans ####
    #################################################
    
    df_distributions = []

    for task in p.task_list:
        if p.careful_whisper:
            results_fn = os.path.join(results_dir,  f'task-{task}_group-analyzed-behavior_window-size-{str(p.window_size).zfill(5)}_human-careful-whisper-distributions-lemmatized.csv')
        else:
            results_fn = os.path.join(results_dir,  f'task-{task}_group-analyzed-behavior_window-size-{str(p.window_size).zfill(5)}_human-model-distributions-lemmatized.csv')
        
        df = pd.read_csv(results_fn)
        df['task'] = task
        df_distributions.append(df)

    df_distributions = pd.concat(df_distributions).reset_index(drop=True)

    if p.careful_whisper:
        df_distributions = split_model_name(df_distributions, input_col='model_name', dataset_col='dataset', model_col='model_name')

        # Map values using dictionary, keeping original value if no mapping exists
        df_distributions['model_name'] = df_distributions['model_name'].map(MODEL_NAME_MAPPING).fillna(df_distributions['model_name'])
        out_fn = os.path.join(results_dir, f'all-task_group-analyzed-behavior_window-size-{str(p.window_size).zfill(5)}_human-careful-whisper-distributions-lemmatized.csv')
    else:
        out_fn = os.path.join(results_dir, f'all-task_group-analyzed-behavior_window-size-{str(p.window_size).zfill(5)}_human-model-distributions-lemmatized.csv')

    df_distributions.to_csv(out_fn, index=False)

    # cmap = create_spoken_written_cmap(continuous=False)
    sns.set(style='white', rc={'figure.figsize':(8,5)})

    cmap = utils.create_colormap(dtype=CMAP_DTYPE, continuous=False, N_MODELS=N_MODELS)
    ax = utils.plot_bar_results(df_distributions, x='model_name', y="kl_divergence", hue="modality", order=models_order, cmap=cmap, figsize=(7,5), add_points=False)

    plt.xticks(rotation=45, ha='right')

    plt.ylim([0, 6.5])
    plt.ylabel('KL Divergence')
    
    plt.title(f'All task - model fit to human distribution')
    plt.gca().get_legend().remove()
    plt.tight_layout()

    plt.savefig(os.path.join(plots_dir, "all-task_human-llm-comparison_kl-divergence.pdf"), bbox_inches='tight', dpi=600)
    plt.close('all')

    # sns.barplot(data=df_kl_div, x='model_name', y=variable, hue='modality', palette=cmap, order=ordered_modalities)
    # plt.ylim([0, 4.5])

    #################################################
    ####### Plot 4: quadrant plot of accuracy #######
    #################################################

    sns.reset_defaults()

    # Load file with accuracy difference information
    if p.careful_whisper:
        results_fn = os.path.join(results_dir, f'all-task_group-analyzed-behavior_window-size-{str(p.window_size).zfill(5)}_human-careful-whisper-distributions-lemmatized.csv')
    else:
        results_fn = os.path.join(results_dir, f'all-task_group-analyzed-behavior_window-size-{str(p.window_size).zfill(5)}_human-model-distributions-lemmatized.csv')

    df_tasks = pd.read_csv(results_fn)

    # Load quadrants across tasks
    df_quadrants = utils.load_task_model_quadrants(preproc_dir, models_dir, p.task_list, model_names, p.word_model)

    # Create the density plot
    cmap = utils.create_colormap(dtype='human-model', continuous=True, N_MODELS=N_MODELS)
    g = utils.create_joint_density_plot(df_tasks, df_quadrants, cmap=cmap)

    plt.savefig(os.path.join(plots_dir, "all-task_human-llm-comparison_quadrant-plot.pdf"), bbox_inches='tight', dpi=600)
    plt.close('all')

    #################################################
    ###### Plot 5: Difference of KL Divergence ######
    #################################################

    variables = ["kl_divergence", "jensenshannon_dist", "earthmovers_dist"]

    df_contrasts = create_modality_contrasts(
        df_distributions, modality_pairs, variables
    )

    plot_contrasts(
        df_contrasts, 
        modality_pairs, 
        variables,
        plots_dir, 
        models_order, 
        N_MODELS, 
    )
    

    # modality_columns = ['-'.join(pair) for pair in modality_pairs]

    # df_kl_difference = []

    # for (task, model_name), df in df_distributions.groupby(['task', 'model_name']):

    #     all_contrasts = []

    #     for mod1, mod2 in modality_pairs:
    #         df_mod1 = df[df['modality'] == mod1].reset_index(drop=True)
    #         df_mod2 = df[df['modality'] == mod2].reset_index(drop=True)

    #         kl_diff = (df_mod2['kl_divergence'] - df_mod1['kl_divergence']).to_numpy()
    #         all_contrasts.append(kl_diff)
        
    #     all_contrasts = np.stack(all_contrasts)
    #     df_diff = pd.DataFrame(all_contrasts.T, columns=modality_columns)
    #     df_diff['task'] = task
    #     df_diff['model_name'] = model_name
    #     df_kl_difference.append(df_diff)

    # df_kl_difference = pd.concat(df_kl_difference).reset_index(drop=True)

    # for i, (mod1, mod2) in enumerate(modality_pairs):

    #     cmap = utils.create_colormap(dtype=modality_columns[i], N_MODELS=N_MODELS)

    #     ax = sns.pointplot(df_kl_difference, x=modality_columns[i], y="model_name",
    #         color="black", markersize=4, linestyles="none", order=models_order,
    #         errorbar='se',
    #     )

    #     plt.xlim([-1, 1])

    #     xmin, xmax = ax.get_xlim()
    #     ymin, ymax = ax.get_ylim()

    #     # Create gradient rectangle
    #     gradient_rect = patches.Rectangle(
    #         (xmin, ymin), xmax - xmin, ymax - ymin,
    #         transform=ax.transData,
    #         color="white", alpha=0.2
    #     )

    #     ax.add_patch(gradient_rect)

    #     # Use the gradient as an image in the background
    #     ax.imshow(
    #         np.linspace(0, 1, 256).reshape(1, -1),  # Create 1D gradient
    #         aspect="auto", extent=(xmin, xmax, ymin, ymax),
    #         origin="lower", cmap=cmap, alpha=0.75
    #     )

    #     ax.vlines(0, ymin=ymin, ymax=ymax, linestyle='--', color='0.5', linewidth=2)

    #     plt.savefig(os.path.join(plots_dir, f"all-task_human-llm-comparison_{mod1}-{mod2}-kl-divergence-difference.pdf"), bbox_inches='tight', dpi=600)
    #     plt.close('all')
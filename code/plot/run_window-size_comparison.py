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

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # type of analysis we're running --> linked to the name of the regressors
    parser.add_argument('-task_list', '--task_list', type=str, nargs='+', default=['black', 'wheretheressmoke', 'howtodraw'])
    parser.add_argument('-word_model', '--word_model', type=str, default='fasttext')
    parser.add_argument('-modality_list', '--modality_list', type=str, nargs='+', default=['video', 'audio', 'text']) 
    parser.add_argument('-o', '--overwrite', type=int, default=0)
    p = parser.parse_args()

    # Set window sizes
    WINDOW_SIZES = [
        2, 3, 4, 5, 10, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300
    ]

    if len(p.modality_list) == 3:
        CMAP_DTYPE = 'multimodal'
    else:
        CMAP_DTYPE = 'spoken-written'

    FIG_SIZE = (5, 4)

    preproc_dir = os.path.join(BASE_DIR, 'stimuli/preprocessed')
    results_dir = os.path.join(BASE_DIR, 'derivatives/results/behavioral/')

    ###################################
    ### Load results from task list ###
    ###################################

    models_dir = os.path.join(BASE_DIR, 'derivatives/model-predictions/')
    plots_dir = os.path.join(BASE_DIR, 'derivatives/plots/final/human-llm-comparison/')

    MLM_MODELS = list(nlp.MLM_MODELS_DICT.keys())[1:]
    CLM_MODELS = list(nlp.CLM_MODELS_DICT.keys()) 
    model_names = CLM_MODELS + MLM_MODELS

    remove_models = ['qwen3-8B', 'llama3.1-8B']
    model_names = [model for model in model_names if model not in remove_models]

    print (f'Loading the following models')
    print (f'MLM models: {MLM_MODELS}')
    print (f'CLM models: {CLM_MODELS}')

    attempt_makedirs(plots_dir)
    
    # ###################################
    # ### Load results from task list ###
    # ###################################

    df_results = []

    for task in p.task_list:
        for i, window_size in enumerate(WINDOW_SIZES):
            results_fn = os.path.join(results_dir, f'task-{task}_group-analyzed-behavior_window-size-{str(window_size).zfill(5)}_human-model-lemmatized.csv')
            df_task = pd.read_csv(results_fn)
            df_task['task'] = task
            df_task['window_size'] = window_size
            df_task['window_number'] = i
            df_results.append(df_task)

    # concatenate into one dataframe --> write to file for posterity 
    out_fn = os.path.join(results_dir, f'all-task_group-analyzed-behavior_all-window-sizes_human-model-lemmatized.csv')

    df_results = pd.concat(df_results).reset_index(drop=True)
    df_results.to_csv(out_fn, index=False)

    # use the accuracy within these results to get the order to plot the models
    ordered_accuracy = utils.get_ordered_accuracy(df_results)

    # always put humans first (audio, text) then CLM models, then MLM models
    order = p.modality_list + MLM_MODELS
    models_order = [item for item in ordered_accuracy if item not in order]
    models_order = models_order + MLM_MODELS
    

    human_models_order = p.modality_list + models_order

    ###################################
    ##### Make line plot of sizes #####
    ###################################
    cmap = utils.create_colormap(dtype=CMAP_DTYPE, continuous=False, N_MODELS=len(models_order))

    sns.set(style='ticks', rc={'figure.figsize': FIG_SIZE})

    variable = f'{p.word_model}_top_word_accuracy'
    sns.lineplot(df_results, x='window_number', y=variable, hue='modality', marker="o", markersize=4, ci=None, palette=cmap, linewidth=2, hue_order=human_models_order)

    # plt.xticks(WINDOW_SIZES)
    plt.xticks(range(len(WINDOW_SIZES)), WINDOW_SIZES, rotation=45)

    plt.gca().legend(bbox_to_anchor=(1.1, 1.05))
    plt.xlabel('Context Window (# of Words)')
    plt.ylabel('Accuracy (% Correct)')
    plt.ylim([0, 1.0])

    plt.title('Human predictions are more accurate than LLMs regardless of context window', y=1.05)
    sns.despine()

    plt.savefig(os.path.join(plots_dir, "all-task_human-llm-comparison_continuous-accuracy_all-window-sizes.pdf"), bbox_inches='tight', dpi=600)
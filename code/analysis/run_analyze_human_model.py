import os, sys
import argparse
import numpy as np
import pandas as pd

sys.path.append('../utils/')

from config import *
import dataset_utils as utils
import analysis_utils as analysis
from tommy_utils.nlp import load_word_model

WRONG_PHONE_COLS = [
    'wrong_resp_n_correct', 'wrong_resp_n_incorrect', 'wrong_resp_accuracy',
    'barnard_stat', 'barnard_pvalue', 'filter_for_leakage'
]

N_WORDS_PROSODY = [3, 5, 7, 9]

def add_wrong_phoneme_info(df_cleaned_results, df_results_analyzed, model_name=None):

    create_filter = lambda df, modality, word_index: (df['modality'] == modality) & (df['word_index'] == word_index)

    # If we're using a model name, we can ignore
    if model_name:
        wrong_phone_cols = ['filter_for_leakage']
    else:
        wrong_phone_cols = WRONG_PHONE_COLS

    for (modality, word_index), _ in df_results_analyzed.groupby(['modality', 'word_index']):

        if model_name:
            # Place into our analyzed data 
            modality = 'text'
            dest_filter = create_filter(df_results_analyzed, model_name, word_index)
        else:
            dest_filter = create_filter(df_results_analyzed, modality, word_index)
        
        # Get the phoneme information
        src_filter = create_filter(df_cleaned_results, modality, word_index)
        wrong_phoneme_info = df_cleaned_results.loc[src_filter, wrong_phone_cols].iloc[0]

        df_results_analyzed.loc[dest_filter, wrong_phone_cols] = wrong_phoneme_info.to_numpy()

    return df_results_analyzed

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # type of analysis we're running --> linked to the name of the regressors
    parser.add_argument('-t', '--task', type=str)
    parser.add_argument('-m', '--model_names', type=str, nargs='+')
    parser.add_argument('-word_model', '--word_model_name', type=str, default='fasttext')
    parser.add_argument('-window_size', '--window_size', type=int, default=25)
    parser.add_argument('-careful_whisper', '--careful_whisper', type=int, default=0)
    parser.add_argument('-drop_shortened', '--drop_shortened', type=int, default=0)
    parser.add_argument('-o', '--overwrite', type=int, default=0)
    p = parser.parse_args()

    print (f'Task: {p.task}', flush=True)
    print (f'Window Size: {p.window_size}', flush=True)

    ###############################################
    ####### Set paths and directories needed ######
    ###############################################

    # Sourced for aggregating data across subjects
    if p.drop_shortened:
        results_dir = os.path.join(BASE_DIR, 'derivatives/results/behavioral-shortened-removed/')
    else:
        results_dir = os.path.join(BASE_DIR, 'derivatives/results/behavioral/')

    utils.attempt_makedirs(results_dir)

    preproc_dir = os.path.join(BASE_DIR, 'stimuli/preprocessed')
    models_dir = os.path.join(BASE_DIR, 'derivatives/model-predictions')
    logits_dir = models_dir.replace(BASE_DIR, BASE_DIR)

    ###############################################
    ####### Load transcript with prosody info #####
    ###############################################

    # Load transcript --> use the version that we include prosody values within
    df_transcript = pd.read_csv(os.path.join(preproc_dir, p.task, f'{p.task}_transcript-selected_prosody-syntax.csv'))
    df_transcript = df_transcript.rename(columns={'Word_Written': 'word', 'Punctuation': 'punctuation'})

    # Need to load the preprocessed transcript that has all words to get the model quadrants
    df_preproc_transcript = pd.read_csv(os.path.join(preproc_dir, p.task, f'{p.task}_transcript-preprocessed.csv'))
    candidate_rows = np.where(df_preproc_transcript['NWP_Candidate'])[0]

    prosody_columns = [
        'prominence', 'boundary', #'prominence_mean', 'prominence_std', 
        #'relative_prominence', 'relative_prominence_norm',
         # 'boundary_mean', 'boundary_std', 
    ]

    # Add in the additional columns
    additional_prosody_columns = [f"{col}_mean_words{n_words}" for col in prosody_columns for n_words in N_WORDS_PROSODY]
    prosody_columns += additional_prosody_columns

    # Add in the past n word prosody and boundary columns
    prosody_columns += [f"prominence_{i+1}" for i in range(max(N_WORDS_PROSODY))]
    prosody_columns += [f"boundary_{i+1}" for i in range(max(N_WORDS_PROSODY))]

    # Add in part of speech + syntax info
    prosody_columns += ["POS", "POS_spaCy", "Dep", "Head", "Clause_ID", "Clause_Pos"]

    # select the columns that we want to save out for gross-comparison
    combined_columns = [
        'modality', 'word_index', 'top_pred', 'ground_truth', 'accuracy', 
        f'{p.word_model_name}_top_word_accuracy', 'top_prob', 'predictability', 
        'entropy', 'entropy_group', 'accuracy_group'
    ]

    combined_columns = combined_columns + prosody_columns + WRONG_PHONE_COLS

    ###########################################################
    ####### Load word model used for continuous accuracy ######
    ###########################################################

    word_model = load_word_model(model_name=p.word_model_name, cache_dir=CACHE_DIR)
    word_model_info = (p.word_model_name, word_model)

    ########################################################
    ############ Load human results and analyze ############
    ########################################################

    # Analyze results from cleaned data 
    df_results = pd.read_csv(os.path.join(results_dir, f'task-{p.task}_group-cleaned-behavior.csv'))
    out_fn = os.path.join(results_dir, f'task-{p.task}_group-analyzed-behavior_human.csv')

    if not os.path.exists(out_fn) or p.overwrite:
        df_analyzed_results = analysis.analyze_human_results(df_transcript, df_results, word_model_info, window_size=p.window_size, top_n=None, drop_rt=None, drop_shortened=p.drop_shortened)

        # Add wrong phoneme info
        df_analyzed_results = add_wrong_phoneme_info(df_results, df_analyzed_results)

        # Add in prosody columns and add to the list
        df_analyzed_results.loc[:, prosody_columns] = df_transcript.loc[df_analyzed_results['word_index'], prosody_columns].reset_index(drop=True)
        df_analyzed_results.to_csv(out_fn, index=False)
    else:
        df_analyzed_results = pd.read_csv(out_fn)
    
    ### Now do model data
    if p.careful_whisper:
        out_fn = os.path.join(results_dir, f'task-{p.task}_group-analyzed-behavior_window-size-{str(p.window_size).zfill(5)}_human-careful-whisper.csv')
    else:
        out_fn = os.path.join(results_dir, f'task-{p.task}_group-analyzed-behavior_window-size-{str(p.window_size).zfill(5)}_human-model.csv')

    if not os.path.exists(out_fn) or p.overwrite:
        df_all_models = []

        for model_name in p.model_names:
            # Load model data and add wrong phoneme information
            df_model = analysis.analyze_model_accuracy(df_transcript, word_model_info=word_model_info, models_dir=models_dir, model_name=model_name, task=p.task, window_size=p.window_size, candidate_rows=candidate_rows, lemmatize=False)
            df_model = add_wrong_phoneme_info(df_results, df_model, model_name=model_name)

            # Add in prosody columns and add to the list
            df_model.loc[:, prosody_columns] = df_transcript.loc[df_model['word_index'], prosody_columns].reset_index(drop=True)
            df_all_models.append(df_model)

        # Concatenate models and then add to the humans dataframe
        df_all_models = pd.concat(df_all_models).reset_index(drop=True)

        df_combined = pd.concat([
            df_analyzed_results.reindex(columns=combined_columns), 
            df_all_models.reindex(columns=combined_columns)
        ]).reset_index(drop=True)

        df_combined.to_csv(out_fn, index=False)

    ########################################################
    ########## Repeat process for lemmatized words #########
    ########################################################

    # Analyze human lemmatized data
    df_lemmatized_results = pd.read_csv(os.path.join(results_dir, f'task-{p.task}_group-cleaned-behavior_lemmatized.csv'))
    out_fn = os.path.join(results_dir, f'task-{p.task}_group-analyzed-behavior_human-lemmatized.csv')
    
    if not os.path.exists(out_fn) or p.overwrite:
        df_analyzed_lemmatized = analysis.analyze_human_results(df_transcript, df_lemmatized_results, word_model_info, window_size=p.window_size, top_n=None, drop_rt=None, drop_shortened=p.drop_shortened)

        # Add wrong phoneme info
        df_analyzed_results = add_wrong_phoneme_info(df_lemmatized_results, df_analyzed_lemmatized)

        # Add in prosody columns and add to the list
        df_analyzed_lemmatized.loc[:, prosody_columns] = df_transcript.loc[df_analyzed_lemmatized['word_index'], prosody_columns].reset_index(drop=True)
        df_analyzed_lemmatized.to_csv(out_fn, index=False)
    else:
        df_analyzed_lemmatized = pd.read_csv(out_fn)

    ### Now do model data
    if p.careful_whisper:
        out_fn = os.path.join(results_dir, f'task-{p.task}_group-analyzed-behavior_window-size-{str(p.window_size).zfill(5)}_human-careful-whisper-lemmatized.csv')
    else:
        out_fn = os.path.join(results_dir, f'task-{p.task}_group-analyzed-behavior_window-size-{str(p.window_size).zfill(5)}_human-model-lemmatized.csv')

    if not os.path.exists(out_fn) or p.overwrite:
        df_lemmatized_models = []

        for model_name in p.model_names:
            df_model = analysis.analyze_model_accuracy(df_transcript, word_model_info=word_model_info, models_dir=models_dir, model_name=model_name, task=p.task, window_size=p.window_size, candidate_rows=candidate_rows, lemmatize=True)
            df_model = add_wrong_phoneme_info(df_lemmatized_results, df_model, model_name=model_name)

            # Add in prosody columns and add to the list
            df_model.loc[:, prosody_columns] = df_transcript.loc[df_model['word_index'], prosody_columns].reset_index(drop=True)
            df_lemmatized_models.append(df_model)

        # Concatenate models and then add to the humans dataframe
        df_lemmatized_models = pd.concat(df_lemmatized_models).reset_index(drop=True)

        df_lemmatized_combined = pd.concat([
            df_analyzed_lemmatized.reindex(columns=combined_columns), 
            df_lemmatized_models.reindex(columns=combined_columns)
        ]).reset_index(drop=True)

        df_lemmatized_combined.to_csv(out_fn, index=False)

    ########################################################
    ########### Conduct distribution comparison ############
    ########################################################

    df_lemmatized_results = pd.read_csv(os.path.join(results_dir, f'task-{p.task}_group-cleaned-behavior_lemmatized.csv'))

    if p.careful_whisper:
        out_fn = os.path.join(results_dir, f'task-{p.task}_group-analyzed-behavior_window-size-{str(p.window_size).zfill(5)}_human-careful-whisper-distributions-lemmatized.csv')
    else:
        out_fn = os.path.join(results_dir, f'task-{p.task}_group-analyzed-behavior_window-size-{str(p.window_size).zfill(5)}_human-model-distributions-lemmatized.csv')
    
    if not os.path.exists(out_fn) or p.overwrite:
        df_all_comparisons = []

        for model_name in p.model_names:
            df_comparison = analysis.compare_human_model_distributions(df_lemmatized_results, word_model_info, models_dir=logits_dir, model_name=model_name, task=p.task, lemmatize=True)
            df_comparison = add_wrong_phoneme_info(df_lemmatized_results, df_comparison, model_name=model_name)
            df_comparison.loc[:, prosody_columns] = df_transcript.loc[df_comparison['word_index'], prosody_columns].reset_index(drop=True)
            df_all_comparisons.append(df_comparison)

        df_all_comparisons = pd.concat(df_all_comparisons).reset_index(drop=True)
        df_all_comparisons.to_csv(out_fn, index=False)
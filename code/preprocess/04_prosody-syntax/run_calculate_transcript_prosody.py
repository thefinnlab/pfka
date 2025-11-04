import os, sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append('../../utils/')

from config import *
from text_utils import strip_punctuation

######################################
########## Prosody metrics ###########
######################################

REMOVE_WORDS = ["sp", "br", "lg", "cg", "ls", "ns", "sl", "ig", "{sp}", "{br}", "{lg}", 
 "{cg}", "{ls}", "{ns}", "{sl}", "{ig}", "SP", "BR", "LG", "CG", "LS",
 "NS", "SL", "IG", "{SP}", "{BR}", "{LG}", "{CG}", "{LS}", "{NS}", "{SL}", "{IG}", "pause"]

def calculate_prosody_metrics(df_prosody, n_prev=3, zscore=False):
    """
    Calculate past-word prominence/boundary columns for the last n_prev words.
    Returns the dataframe with new columns: prominence_1..n, boundary_1..n
    """
    df = df_prosody.copy()
    prosody_raw = df['prominence'].to_numpy()
    boundary_raw = df['boundary'].to_numpy()

    if zscore:
        prosody_raw = stats.zscore(prosody_raw)

    past_prosody = []
    past_boundaries = []

    for idx in tqdm(range(len(df)), desc="Calculating past-word prosody"):
        # How many previous words are available
        n_available = min(idx, n_prev)
        # Take the last n_available words
        past_vals_pro = prosody_raw[max(0, idx - n_prev):idx][::-1]  # reverse: most recent first
        past_vals_bound = boundary_raw[max(0, idx - n_prev):idx][::-1]

        # Pad with NaN if less than n_prev
        past_vals_pro = np.pad(past_vals_pro, (0, n_prev - n_available), constant_values=np.nan)
        past_vals_bound = np.pad(past_vals_bound, (0, n_prev - n_available), constant_values=np.nan)

        past_prosody.append(past_vals_pro)
        past_boundaries.append(past_vals_bound)

    # Add columns to dataframe
    for i in range(n_prev):
        df[f'prominence_{i+1}'] = [p[i] for p in past_prosody]
        df[f'boundary_{i+1}'] = [b[i] for b in past_boundaries]

    return df

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # type of analysis we're running --> linked to the name of the regressors
    parser.add_argument('-t', '--task', type=str)
    parser.add_argument('-num_words', '--num_words', type=int, nargs='+', default=5)
    parser.add_argument('-o', '--overwrite', type=int, default=0)
    p = parser.parse_args()

    ###############################################
    ####### Set paths and directories needed ######
    ###############################################

    stim_dir = os.path.join(BASE_DIR, 'stimuli/')

    ###############################################
    ######## Load prosody data and process ########
    ###############################################

    # Load prosody
    prosody_columns = ['stim', 'start', 'end', 'word', 'prominence', 'boundary']
    df_prosody = pd.read_csv(os.path.join(stim_dir, 'prosody', f'{p.task}.prom'), sep='\t', names=prosody_columns)

    # Only keep one set of past-word columns for the largest n_words
    max_n = max(p.num_words)
    df_prosody = calculate_prosody_metrics(df_prosody, n_prev=max_n, zscore=False)

    # Calculate mean columns for each n_words
    for n in p.num_words:
        past_cols = [f'prominence_{i+1}' for i in range(n)]
        df_prosody[f'prominence_mean_words{n}'] = df_prosody[past_cols].apply(
            lambda row: row.mean() if row.notna().sum() == n else np.nan, axis=1
        )

        past_cols = [f'boundary_{i+1}' for i in range(n)]
        df_prosody[f'boundary_mean_words{n}'] = df_prosody[past_cols].apply(
            lambda row: row.mean() if row.notna().sum() == n else np.nan, axis=1
        )
    # Remove non-words
    df_prosody = df_prosody[~df_prosody['word'].isin(REMOVE_WORDS)].reset_index(drop=True)

    # Load transcript
    transcript_fn = os.path.join(stim_dir, 'preprocessed', p.task, f'{p.task}_transcript-selected.csv')
    df_transcript = pd.read_csv(transcript_fn)
    df_transcript = df_transcript.rename(columns={'Word_Written': 'word', 'Punctuation': 'punctuation'})

    # Ensure words match
    words_transcript = df_transcript['word'].str.lower().apply(strip_punctuation)
    words_prosody = df_prosody['word'].str.lower().apply(strip_punctuation)
    assert all(words_transcript == words_prosody), "Words do not match between transcript and prosody!"

    # Merge prosody into transcript
    # Select only the columns we need
    prosody_cols_to_merge = (
        [f'prominence'] +
        [f'prominence_{i+1}' for i in range(max_n)] +
        [f'boundary'] +
        [f'boundary_{i+1}' for i in range(max_n)] +
        [f'prominence_mean_words{n}' for n in p.num_words] +
        [f'boundary_mean_words{n}' for n in p.num_words]
    )

    df_transcript.loc[:, prosody_cols_to_merge] = df_prosody.loc[:, prosody_cols_to_merge]

    df_transcript.to_csv(transcript_fn.replace('.csv', '_prosody.csv'), index=False)

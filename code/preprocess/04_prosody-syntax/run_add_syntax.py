import os, sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append('../../utils/')

from config import *
from preproc_utils import add_syntax_info

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str)
    p = parser.parse_args()

    ###############################################
    ####### Set paths and directories needed ######
    ###############################################

    stim_dir = os.path.join(BASE_DIR, 'stimuli/')

    #################################################
    ######## Load transcript and add syntax #########
    #################################################

    # Load transcript -- add prosody information to the transcript
    transcript_fn = os.path.join(stim_dir, 'preprocessed', p.task, f'{p.task}_transcript-selected_prosody.csv')
    df_transcript = pd.read_csv(transcript_fn)

    # Grab the syntax info
    df_syntax = add_syntax_info(df_transcript, text_column='word', punctuation_column='punctuation')
    df_syntax.to_csv(transcript_fn.replace('.csv', '-syntax.csv'), index=False)
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import os, sys
import glob
import argparse
import shutil
import random
from tqdm import tqdm
import numpy as np

import json
from collections import Counter

sys.path.append('../../../utils/')

from config import *
from dataset_utils import attempt_makedirs

sys.path.append('../utils/')

import utils 

def count_unique_words(data):
    """
    Count the total number of words/tokens and unique words/tokens in a JSON file
    where each item has a 'text' element. This function simply splits on spaces.
    
    Args:
        json_file_path (str): Path to the JSON file
        
    Returns:
        tuple: (total_token_count, unique_token_count, token_frequency)
    """

    # Initialize a list to store all tokens (words or numbers)
    all_words = []
    
    # Process each item in the JSON
    for item in tqdm(data, desc="Counting number of words"):
        if 'text' in item and isinstance(item['text'], str):
            # Simply split by spaces
            words = item['text'].split()
            all_words.extend(words)
    
    # Count token frequencies
    word_counter = Counter(all_words)
    
    # Calculate totals
    total_word_count = len(all_words)
    unique_word_count = len(word_counter)

    # Create distribution dictionary with each token and its frequency
    distribution = {word: count for word, count in word_counter.items()}
    
    # Create output data structure
    output_data = {
        "total_words": total_word_count,
        "unique_words": unique_word_count,
        "distribution": distribution
    }
    
    return output_data

def get_total_time(data):

    total_time = []

    # Go through dataset and count amount of time
    for item in tqdm(data, desc="Aggregating total time"):
        total_time.append(item['duration'])

    total_time_ms = sum(total_time) * 1000
    total_time_str = convert_ms_to_hms(total_time_ms)

    output_data = {
        'average_sample_length': np.mean(total_time),
        'std_sample_length': np.std(total_time),
        'total_time': total_time_str
    }

    return output_data

def convert_ms_to_hms(milliseconds):
    seconds = (milliseconds // 1000) % 60
    minutes = (milliseconds // (1000 * 60)) % 60
    hours = (milliseconds // (1000 * 60 * 60))
    return f"{int(hours)}:{int(minutes)}:{seconds:02}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--base_dir')

    ### Dataset filtering setup
    parser.add_argument('-av', '--audiovisual', type=int, default=1)
    parser.add_argument('-o', '--overwrite', type=int, default=0)

    p = parser.parse_args()

    if p.audiovisual: 
        datasets = ['lrs3', 'avspeech', 'voxceleb2']
        model_combo = 'gpt2-wav2vec2-data2vec'
        out_fn = f"audiovisual-dataset-info.json"
    
    splits = ['train', 'val', 'test']

    nlp_datasets_dir = os.path.join(DATASETS_DIR, 'nlp-datasets')
    results_dir = os.path.join(BASE_DIR, 'derivatives', 'results', 'careful-whisper')

    all_dataset_stats = {}

    for dataset in datasets:

        dataset_stats = {}

        for split in splits:

            dataset_stats[split] = {}

            # Create path to metadata directory
            metadata_dir = os.path.join(nlp_datasets_dir, dataset, 'features', 'metadata', model_combo, split)
            metadata_fn = os.path.join(metadata_dir, f"metadata.json")

            # Load metadata
            metadata = utils.load_json(metadata_fn)

            # Get number of unique words
            words_info = count_unique_words(metadata)
            dataset_stats[split].update(words_info)

            # Get amount of hours
            time_info = get_total_time(metadata)
            dataset_stats[split].update(time_info)
        
        all_dataset_stats[dataset] = dataset_stats

    utils.save_json(os.path.join(results_dir, out_fn), all_dataset_stats)

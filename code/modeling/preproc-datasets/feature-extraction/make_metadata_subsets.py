import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import os, sys
import glob
import argparse
import shutil
import random
from tqdm import tqdm
import numpy as np

sys.path.append('../../../utils/')

from config import *
from dataset_utils import attempt_makedirs

sys.path.append('../utils/')

import utils 

def shuffle_metadata(data, seed=42):
    # Set the seed for random.shuffle
    random.seed(seed)

    # Create a list of all indices
    all_indices = list(range(len(data)))
    random.shuffle(all_indices)

    # reorder the data based on shuffle
    shuffled_data = [data[idx] for idx in all_indices]

    return shuffled_data

def get_durations(metadata):
    # Number of seconds in an hour
    n_secs = 3600
    durations = [x['duration'] / n_secs for x in metadata]

    total_duration = np.sum(durations)

    return total_duration, durations

def make_metadata_subset(data, durations, subset_percentage):

    # Grab the total duration
    total_duration = np.sum(durations)
    
    # Find the number of hours in the current subset --> mask to those number of hours
    subset_hours = total_duration * subset_percentage
    subset_mask = np.where(np.cumsum(durations) < subset_hours)[0]

    print (f"Making {subset_percentage*100}% subset: {subset_hours:.2f} hours", flush=True)

    # Grab the subset from the metadata
    subset_metadata = [shuffled_data[idx] for idx in subset_mask]

    return subset_metadata
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--base_dir')
    parser.add_argument('-d', '--datasets', type=str, nargs='+')
    parser.add_argument('--lang_id', type=str, default='eng',
                help='Language ID ISO-639 code for AVSpeech')

    ### Model names
    parser.add_argument('--text_model', type=str, default='gpt2', help='Text model to use')
    parser.add_argument('--audio_model', type=str, default='wav2vec2',
                    help='Audio model to use, or "None" to skip audio processing')
    parser.add_argument('--video_model', type=str, default='data2vec',
                    help='Video model to use, or None to skip video processing')

    ### Dataset filtering setup
    parser.add_argument('-o', '--overwrite', type=int, default=0)

    args = parser.parse_args()

    splits = ['train', 'val', 'test']

    nlp_datasets_dir = os.path.join(DATASETS_DIR, 'nlp-datasets')

    # Set name of the model combo for the metadata
    model_combo = args.text_model

    if args.audio_model:
        model_combo += f'-{args.audio_model}'

    if args.video_model:
        model_combo += f'-{args.video_model}'

    ################################################
    ########## Create subset percentages ###########
    ################################################

    subset_percentages = np.logspace(0.3, 1.4, 10) / 100

    subset_percentages = np.concatenate((
        subset_percentages,
        np.arange(0.3, 1, 0.1)
    ))

    subset_percentages = np.sort(np.round(subset_percentages, 2))

    for split in splits:

        # Create a list to collect all the metadata
        all_metadata = []

        # Load data for all datasets
        for dataset in args.datasets:

            # Create cache for our features and a temp directory for writing progress
            cache_dir = os.path.join(nlp_datasets_dir, dataset, 'features', 'metadata', model_combo)
            metadata_path = os.path.join(cache_dir, split, 'metadata.json')

            metadata = utils.load_json(metadata_path)
            all_metadata += metadata

        ################################################
        ############ Save combined metadata ############
        ################################################

        if len(args.datasets) > 1:

            av_datasets = ['lrs3', 'voxceleb2', 'avspeech']
            av_dataset = all([x in av_datasets for x in args.datasets])

            if av_dataset:
                output_dir = os.path.join(nlp_datasets_dir, 'av-combined')
            else:
                output_dir = os.path.join(nlp_datasets_dir, 'speech-combined')
        else:
            output_dir = os.path.join(nlp_datasets_dir, args.datasets[0])

        output_dir = os.path.join(output_dir, 'features', 'metadata', model_combo, split)
        metadata_fn = os.path.join(output_dir, f"metadata.json")
        attempt_makedirs(output_dir)

        # Save all the metadata
        if not os.path.exists(metadata_fn) or args.overwrite:
            utils.save_json(metadata_fn, all_metadata)

        ################################################
        ########### Make output for subsets ############
        ################################################

        # We don't subset splits besides the train split
        if split != 'train':
            continue

        # shuffle all training data --> get durations of shuffled data
        shuffled_data = shuffle_metadata(all_metadata)
        total_duration, durations = get_durations(shuffled_data)

        for subset in subset_percentages:

            percentage = int(subset * 100)
            subset_fn = os.path.join(output_dir, f"metadata_subset-{str(percentage).zfill(3)}.json")

            subset_data = make_metadata_subset(shuffled_data, durations, subset_percentage=subset)
            utils.save_json(subset_fn, subset_data)
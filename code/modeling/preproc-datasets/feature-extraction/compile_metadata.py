import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import os, sys
import glob
import argparse
import shutil
from tqdm import tqdm
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import torch
import numpy as np

sys.path.append('../utils/')

import utils

def logarithmic_decay(initial_value, decay_rate, time_steps, min_value=2, max_value=4):
    """
    Implements a logarithmic decay function with specified minimum and maximum bounds.
    
    Parameters:
    initial_value (float): The starting value (will be clamped to max_value).
    decay_rate (float): The rate of decay (positive value).
    time_steps (int or array): The number of time steps or an array of time points.
    min_value (float): The minimum value the function will decay to (default: 2).
    max_value (float): The maximum value the function can have (default: 4).
    
    Returns:
    numpy.ndarray: Array of values following logarithmic decay, bounded between min_value and max_value.
    """
    # Convert time_steps to numpy array if it's not already
    t = np.asarray(time_steps)
    
    # Ensure initial_value doesn't exceed max_value
    initial_value = min(initial_value, max_value)
    
    # Calculate the decay
    decay = initial_value * (1 - decay_rate * np.log1p(t))
    
    # Clip values to be between min_value and max_value
    return np.clip(decay, min_value, max_value)

def compile_features(args, path, check_keys=None, map_data=False, n_token_threshold=4):
    """Load data from temp JSON and replace file paths with actual data."""

    threshold_fx = lambda x: np.round(logarithmic_decay(initial_value=4, decay_rate=0.12, time_steps=x))

    try:
        # Load the metadata for the file
        data = utils.load_json(path)
        n_words = len(data['text'].split())
        n_tokens = torch.tensor(torch.load(data['text_tokens_path'])).shape[-1]

        if map_data:

            # There are sometimes errors in transcription --> a simple heuristic lets us filter tokens
            if (n_tokens // n_words) > threshold_fx(n_words): 
                print (f"Error for file {path}", flush=True)
                print (f"Sample words: {data['text']}", flush=True)
                msg = f"Number of tokens ({n_tokens}) greater than number of words ({n_words}) with token threshold {n_token_threshold}"
                print (msg, flush=True)
                return False, None, msg

            # Map of paths to load and their destination keys
            mapping = {
                'text_tokens_path': 'text_tokens',
                'attention_mask_path': 'attention_mask',
                'prominence_path': 'prominence',
                'boundary_path': 'boundary'
            }

            if check_keys:
                for x in check_keys:
                    if x not in data:
                        raise Exception(f'Missing {x} data: {path}')

                    # # Test loading the item
                    # _ = torch.load(data[x])
            
            # Load data from paths and replace paths with data
            for k, v in mapping.items():
                if k in data:
                    # Convert tensor to a native Python type that's JSON serializable
                    tensor_data = torch.load(data[k])
                    # Use .tolist() instead of list() for proper conversion
                    if isinstance(tensor_data, torch.Tensor) or isinstance(tensor_data, np.ndarray):
                        data[v] = tensor_data.tolist()
                    else:
                        data[v] = list(tensor_data)
                    # data[v] = tensor_data.tolist() if isinstance(tensor_data, torch.Tensor) else tensor_data
                    del data[k]
        return True, data, "Success"
    except Exception as e:
        return False, None, str(e)

def compile_metadata(args, metadata, temp_dir, num_jobs=None, check_keys=None, map_data=False):
    '''
    Path to the metadata directory
    '''
    parent_dir = os.path.dirname(temp_dir)
    errors_dir = os.path.join(parent_dir, 'errors')
    metadata_type = os.path.basename(parent_dir)

    if num_jobs is None:
        # Use 75% of available cores by default, but at least 1
        num_jobs = max(1, int(multiprocessing.cpu_count() * 0.9))

    # Find all files that have been preprocessed
    all_fns = sorted(os.listdir(temp_dir))
    to_process_files = [fn.replace('_processed.json', '') for fn in all_fns]

    # If we don't want to overwrite and metadata exists
    existing_count = 0
    
    if not args.overwrite and metadata:

        # Search for existing files
        existing_files = [item['base_name'] for item in metadata]
        existing_count = len(existing_files)

        # Find set difference between the basenames --> only files that need to be added
        to_process_files = set(to_process_files).difference(existing_files)
        to_process_files = sorted(to_process_files)

        print(f"Found {len(existing_files)} existing transcripts", flush=True)

    to_process_count = len(to_process_files)
    to_process_files = [os.path.join(temp_dir, f'{fn}_processed.json') for fn in to_process_files]
    
    print(f"Processing {to_process_count} files into metadata", flush=True)

    if to_process_count == 0:
        print(f"All audio files already exist for {split} split. Skipping extraction.", flush=True)
        return metadata

    print(f"Found {existing_count} existing audio files and {to_process_count} files to process", flush=True)
    print(f"Using {num_jobs} worker processes for parallel extraction", flush=True)
    
    # Process files in parallel
    completed = 0
    errors = 0

    with ProcessPoolExecutor(max_workers=num_jobs) as executor:
        # Submit all tasks
        future_to_metadata = {
            executor.submit(compile_features, args, fn, check_keys, map_data): fn
            for fn in to_process_files
        }
        
        # Process results as they complete
        pbar = tqdm(total=to_process_count, desc=f"Compiling metadata for {metadata_type}")
        
        for future in as_completed(future_to_metadata):
            fn = future_to_metadata[future]
            try:
                success, data, message = future.result()
                
                if success:
                    metadata.append(data)
                    completed += 1
                else:
                    # shutil.move(fn, fn.replace(temp_dir, errors_dir))
                    errors += 1
                    print(f"Error processing {fn}: {message}", flush=True)

            except Exception as e:
                errors += 1
                print(f"Exception processing {fn}: {str(e)}", flush=True)
            
            pbar.update(1)
        
        pbar.close()
    return metadata

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--base_dir')
    parser.add_argument('-d', '--dataset', type=str)
    parser.add_argument('--output_dir', type=str, required=True, 
                    help='Base directory for output (default: dataset_name_processing)')
    parser.add_argument('--split', type=str, default=None,
                    help='Which split to process')
    parser.add_argument('--lang_id', type=str, default='eng',
                help='Language ID ISO-639 code for AVSpeech')

    ### Model names
    parser.add_argument('--text_model', type=str, default='gpt2', help='Text model to use')
    parser.add_argument('--audio_model', type=str, default='wav2vec2',
                    help='Audio model to use, or "None" to skip audio processing')
    parser.add_argument('--video_model', type=str, default=None,
                    help='Video model to use, or None to skip video processing')

    ### Dataset filtering setup
    parser.add_argument('--min_words', type=int, default=4, help='Minimum number of words per sample')
    parser.add_argument('--max_words', type=int, default=128, help='Maximum number of words per sample')

    parser.add_argument('-o', '--overwrite', type=int, default=0)

    args = parser.parse_args()

    if args.split:
        splits = [args.split]
    else:
        splits = ['train', 'val', 'test']
    
    # Determine output directory
    output_dir = args.output_dir or f"{args.dataset}_processing"

    # Set name of the model combo for the metadata
    model_combo = args.text_model

    if args.audio_model:
        model_combo += f'-{args.audio_model}'

    if args.video_model:
        model_combo += f'-{args.video_model}'

    # Create cache for our features and a temp directory for writing progress
    cache_dir = os.path.join(args.output_dir, 'features', 'metadata', model_combo)

    for split in splits:

        temp_dir = os.path.join(cache_dir, split, 'temp')
        errors_dir = os.path.join(cache_dir, split, 'errors')

        if args.dataset == 'avspeech':
            temp_dir = os.path.join(cache_dir, split, args.lang_id, 'temp')
            errors_dir = os.path.join(cache_dir, split, args.lang_id, 'errors')
        elif args.dataset == 'voxceleb2':
        # if args.dataset in ['avspeech', 'voxceleb2']:
            if args.lang_id:
                temp_dir = os.path.join(temp_dir, args.lang_id)
                errors_dir = os.path.join(errors_dir, args.lang_id)
            else:
                continue
        
        # Metadata paths
        metadata_path = os.path.join(cache_dir, split, 'metadata.json')
        error_metadata_path = os.path.join(cache_dir, split, 'error_metadata.json')

        # Load or create metadata --> if doesn't exist, will return an empty list
        if args.overwrite:
            print (f"Overwriting existing metadata", flush=True)
            metadata = []
            error_metadata = []
        else:
            print (f"Updating metadata if exists", flush=True)
            metadata = utils.load_json(metadata_path)
            error_metadata = utils.load_json(error_metadata_path)

        metadata = compile_metadata(args, metadata, temp_dir, check_keys=['audio_features_path', 'video_features_path'], map_data=True)
        utils.save_json(metadata_path, metadata)

        # Repeat same process for errors information
        error_metadata = compile_metadata(args, error_metadata, errors_dir)
        utils.save_json(error_metadata_path, error_metadata)
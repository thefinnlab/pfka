import os
import sys
import glob
import math
import argparse
import random
import shutil
import tarfile
from tqdm import tqdm
import pandas as pd
import numpy as np
from natsort import natsorted
import concurrent.futures
from typing import Dict, List, Tuple

# Assuming these imports work in your environment
sys.path.append('../../../utils/')

from config import *
from dataset_utils import attempt_makedirs

sys.path.append('../utils/')

import utils

def untar_file(fn: str) -> bool:
    """
    Extract a tar file to a proper directory.
    
    Args:
        fn: Path to the tar file
        
    Returns:
        bool: True if extraction succeeded, False otherwise
    """
    print(f"Extracting {fn}...", flush=True)
    try:
        # Extract to parent directory instead of the tar file itself
        extract_dir = os.path.dirname(fn)

        # Create the extraction directory if it doesn't exist
        if not os.path.exists(extract_dir):
            os.makedirs(extract_dir)
                
        with tarfile.open(fn, 'r') as tar:
            tar.extractall(extract_dir)
        return True
    except Exception as e:
        print(f"Error extracting {fn}: {e}", flush=True)
        return False


def process_tar_file(fn, split_info, dirs, processor, model, processed_files=None):
    """
    Process a single tar file and move its contents to the appropriate split directories.
    
    Args:
        fn: Path to the tar file
        split_info: Dictionary mapping split names to lists of video IDs
        dirs: Directory structure for splits
        
    Returns:
        Dictionary mapping split names to lists of processed file IDs
    """

    # Initialize tracking for processed files
    processed_files = {split: {'lang_data': [], 'video_ids': []} for split in split_info.keys()}
    
    # Get file path without extension
    file_path = os.path.splitext(fn)[0]
    base_name = os.path.basename(file_path)
    
    # Check if it's a tar file that needs extraction
    if fn.endswith('.tar') and not os.path.isdir(file_path):
        if not untar_file(fn):
            return processed_files
    
    # Skip if directory doesn't exist
    if not os.path.isdir(file_path):
        print(f"Warning: Directory {file_path} does not exist or is not a directory", flush=True)
        return processed_files
        
    # Get video IDs from the extracted directory
    try:
        video_ids = os.listdir(file_path)
    except Exception as e:
        print(f"Error listing directory {file_path}: {e}", flush=True)
        return processed_files

    # Process each split
    for split, split_df in split_info.items():

        # Filter videos that belong to this split
        split_ids = split_df['video_id'].tolist()
        split_vids = [vid for vid in video_ids if vid in split_ids]

        # Move each video to its destination
        for vid in tqdm(split_vids, desc=f"Processing {base_name}, {split} videos"):

            # Directory of the current video
            vid_dir = os.path.join(file_path, vid)

            # Grab all the videos and load as a batch
            vid_fns = natsorted(glob.glob(os.path.join(vid_dir, '*')))
            clip_ids = [os.path.basename(fn) for fn in vid_fns] # Get clip IDs from filenames

            # Get segment information for this video if available
            segment_info = {}
            if split_df is not None:
                video_rows = split_df[split_df['video_id'] == vid]
                if not video_rows.empty:
                    # Get the first row for this video (assuming one row per video)
                    row = video_rows.iloc[0]
                    segment_info = {
                        'start_segment': row.get('start_segment'),
                        'end_segment': row.get('end_segment'),
                        'x_coord': row.get('x_coord'),
                        'y_coord': row.get('y_coord')
                    }
            try:
                # Prepare the batch and detect languages
                batch = prepare_audio_batch(vid_fns)
                detected_langs, probs =  utils.classify_language(batch, processor, model, return_probs=True)

                # Get counts of each language
                unique_langs, counts = np.unique(detected_langs, return_counts=True)

                # Find the most common language
                max_count = np.argmax(counts)
                lang_ratio = np.max(counts) / len(detected_langs)
                most_common_lang = unique_langs[max_count]

                # Create destination directory with language subfolder
                dest_path = os.path.join(dirs['src'], split, most_common_lang, vid)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)

                # Log language data for each clip
                for i, (clip_id, lang, prob) in enumerate(zip(clip_ids, detected_langs, probs)):
                    # Add to language log
                    lang_info = {
                        'video_id': vid,
                        'clip_id': clip_id,
                        'lang_id': lang,
                        'prob': prob.item(),
                        'split': split,
                        'most_common_lang': most_common_lang,
                        'percent_clips_lang': lang_ratio,
                    }

                    # Add in segment information
                    lang_info.update(segment_info)

                    # Append to processed files information
                    processed_files[split]['lang_data'].append(lang_info)
            
                # Move the video directory to the destination
                shutil.move(vid_dir, dest_path)
                processed_files[split]['video_ids'].append(vid)
            except Exception as e:
                print(f"Error moving {vid}: {e}", flush=True)
    return processed_files

def prepare_audio_batch(fns, sr=16000):

    batch = []

    for fn in fns:
        waveform, audio_sr = utils.load_audio(fn)
        waveform, audio_sr = utils.resample_audio(waveform, orig_sr=audio_sr, target_sr=16000)
        batch.append(waveform.squeeze())

    return batch

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Preprocess audio/video-text dataset')

    ### Sharding for more efficient processing
    parser.add_argument('--num_shards', type=int, default=1,
                        help='Number of shards to divide the dataset into')
    parser.add_argument('--current_shard', type=int, default=0,
                        help='Current shard to process (0-based indexing)')
    ### Video
    parser.add_argument('--overwrite', type=int, default=0,
                        help='Force extraction even if files exist')

    args = parser.parse_args()

    # Set up dataset paths
    dataset = 'avspeech'
    data_dir = os.path.join(DATASETS_DIR, 'nlp-datasets', dataset)
    dataset_dir = os.path.join(DATASETS_DIR, 'nlp-datasets', dataset)
    
    # Create directories if needed
    attempt_makedirs(data_dir)

    # Get dataset configuration
    dataset_config = utils.DATASET_CONFIGS[dataset]
    splits = dataset_config['splits']
    
    # Add validation split if needed
    if 'val' not in splits:
        splits.append('val')

    # Prepare directory structure
    dirs, splits = utils.prepare_directory_structure(data_dir, splits, dir_names=['src'])

    # Define CSV configuration
    csv_splits = ['train', 'test']
    split_columns = ['video_id', 'start_segment', 'end_segment', 'x_coord', 'y_coord']

    # Load video IDs for each split
    split_info = {}
    for split in csv_splits:
        try:
            df = pd.read_csv(os.path.join(dataset_dir, f'avspeech_{split}.csv'), names=split_columns)
            split_info[split] = df
        except Exception as e:
            print(f"Error loading CSV for {split}: {e}", flush=True)
            split_info[split] = []

    # Get all tar files
    tar_fns = sorted(glob.glob(os.path.join(dataset_dir, 'clips', '*.tar')))

    if not tar_fns:
        print(f"Warning: No tar files found in {os.path.join(dataset_dir, 'clips')}")

    # Apply sharding logic --> divide dataset into number of shards 
    if args.num_shards > 1:
        # Calculate shard size and starting/ending indices
        shard_size = math.ceil(len(tar_fns) / args.num_shards)
        start_idx = args.current_shard * shard_size
        end_idx = min(start_idx + shard_size, len(tar_fns))
        
        # Get only the files for the current shard
        tar_fns = tar_fns[start_idx:end_idx]
        
        print(f"Processing shard {args.current_shard+1}/{args.num_shards} with {len(tar_fns)} files", flush=True)

    # tar_fns = sorted(glob.glob(os.path.join(dataset_dir, 'clips/xa[o-z].tar')))
    # tar_fns.append(os.path.join(dataset_dir, 'clips/xba.tar'))

    # print (f"Total files: {tar_fns}", flush=True)

    # Load models for language classification
    processor, model = utils.load_language_classifier()

    # Initialize language log dataframe
    all_language_data = []

    # Process tar files (can be parallelized later)
    print("Starting processing of tar files...", flush=True)
    all_processed_files = {split: {'lang_data': [], 'video_ids': []} for split in split_info.keys()}
    
    # For testing, process just a few tar files sequentially
    for fn in tar_fns:  # Limit for testing, remove slice for full processing
        try:
            print(f"Processing {fn}...", flush=True)
            processed_info = process_tar_file(fn, split_info, dirs, processor, model)

            total_files = 0
            
            # Track processed files and language data
            for split, info in processed_info.items():

                video_ids = info['video_ids']
                lang_info = info['lang_data']

                all_processed_files[splxit]['video_ids'].extend(video_ids)
                all_processed_files[split]['lang_data'].extend(lang_info)

                total_files += len(video_ids)

            print(f"Processed {fn}: {total_files} files", flush=True)
        except Exception as e:
            print(f"Error processing {fn}: {e}", flush=True)

    # Create language log dataframes for each split
    for split, data in all_processed_files.items():
        if data:  # Only create CSV if we have data for this split
            split_lang_df = pd.DataFrame(data['lang_data'])
            lang_log_path = os.path.join(dataset_dir, 'src', split, f'{split}_metadata-{str(args.current_shard + 1).zfill(5)}.csv')

            if os.path.exists(lang_log_path) and not args.overwrite:
                # Load existing dataframe
                existing_df = pd.read_csv(lang_log_path)
                
                # Get unique clip_ids from existing dataframe
                existing_clip_ids = set(existing_df['clip_id'])
                
                # Filter out rows from the new dataframe that have clip_ids already in the existing dataframe
                split_lang_df = split_lang_df[~split_lang_df['clip_id'].isin(existing_clip_ids)]
                
                # Concatenate the existing and filtered new dataframes
                split_lang_df = pd.concat([existing_df, split_lang_df], ignore_index=True)
            
            split_lang_df.to_csv(lang_log_path, index=False)
            print(f"Created {split} language classification log at {lang_log_path} with {len(split_lang_df)} entries", flush=True)

    print("Processing complete!", flush=True)
import os
import sys
import glob
import argparse
import tqdm
import random

import shutil
import pandas as pd
import numpy as np

from pathlib import Path

import multiprocessing
import concurrent.futures

# Assuming these imports work in your environment
sys.path.append('../../../../utils/')

from config import *
from dataset_utils import attempt_makedirs

sys.path.append('../../utils/')

import utils

def filter_speaker_languages(df):
    '''
    Filter speaker videos by their most common language 
    '''

    # Group by speaker_id and count occurrences of each most_common_lang
    speaker_language_counts = df.drop_duplicates(subset=['speaker_id', 'video_id', 'most_common_lang']) \
        .groupby(['speaker_id', 'most_common_lang']) \
        .size() \
        .reset_index(name='count')

    # For each speaker_id, find the most common language (highest count)
    most_common_langs = speaker_language_counts.sort_values(['speaker_id', 'count'], ascending=[True, False]) \
        .drop_duplicates('speaker_id') \
        .rename(columns={'most_common_lang': 'primary_language'})

    # Keep only the rows with the most common language for each speaker
    result = df.merge(most_common_langs[['speaker_id', 'primary_language']], 
                        on='speaker_id')

    # If you want to drop the temporary 'primary_language' column
    filtered_df = result[result['most_common_lang'] == result['primary_language']]
    filtered_df = filtered_df.drop(columns=['primary_language']).sort_values(by=['speaker_id', 'video_id', 'clip_id'])

    return filtered_df

def filter_confidence(df):
    """
    Filter dataframe and return two dataframes:
    1. high_confidence: video_ids where percent_clips_lang == 1 and prob > 0.90 for each lang_id
    2. low_confidence: video_ids that don't meet the above criteria
    """
    high_confidence_df = pd.DataFrame()
    low_confidence_df = pd.DataFrame()
    
    # Get all unique language IDs
    languages = df['lang_id'].unique()
    languages = sorted([str(lang) for lang in languages])
    
    for lang_id in languages:
        # Get videos for this language
        lang_df = df[df['most_common_lang'] == lang_id]
        
        # Get video IDs that meet high confidence criteria
        valid_videos = lang_df[(lang_df['percent_clips_lang'] > 0.75) & 
                              (lang_df['prob'] > 0.95)]['video_id'].unique()
        
        # Create high confidence dataframe for this language
        high_conf_df = lang_df[lang_df['video_id'].isin(valid_videos)]
        low_conf_df = lang_df[~lang_df['video_id'].isin(valid_videos)]
        
        # Append to combined dataframes
        high_confidence_df = pd.concat([high_confidence_df, high_conf_df])
        low_confidence_df = pd.concat([low_confidence_df, low_conf_df])
        
        print(f"Language {lang_id.upper()}: {len(high_conf_df)} high confidence clips, {len(low_conf_df)} low confidence clips", flush=True)
        
    high_confidence_df, low_confidence_df = [df.sort_values(by=['speaker_id', 'video_id', 'clip_id']) for df in [high_confidence_df, low_confidence_df]]
    
    return high_confidence_df, low_confidence_df
    
def split_train_val(df, val_ratio=0.1, random_seed=42):
    """
    Split dataframe into train and validation sets based on video_ids
    """
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Get all unique language IDs
    languages = df['lang_id'].unique()
    
    train_df = pd.DataFrame()
    val_df = pd.DataFrame()
    
    for lang_id in languages:
        # Get data for this language
        lang_df = df[df['most_common_lang'] == lang_id]
        
        # Get unique speaker_ids for this language
        speaker_ids = lang_df['speaker_id'].unique()
        
        if len(speaker_ids) == 0:
            continue
        
        # Randomly select speakers for validation
        num_val_speakers = max(1, int(len(speaker_ids) * val_ratio))
        val_speaker_ids = random.sample(list(speaker_ids), num_val_speakers)
        
        # Create train and validation dataframes
        lang_val_df = lang_df[lang_df['speaker_id'].isin(val_speaker_ids)]
        lang_train_df = lang_df[~lang_df['speaker_id'].isin(val_speaker_ids)]
        
        # Append to combined dataframes
        train_df = pd.concat([train_df, lang_train_df])
        val_df = pd.concat([val_df, lang_val_df])
        
        print(f"Language {lang_id.upper()}: {len(lang_train_df)} train clips, {len(lang_val_df)} validation clips", flush=True)
    
    train_df, val_df = [df.sort_values(by=['speaker_id', 'video_id', 'clip_id']) for df in [train_df, val_df]]
    
    return train_df, val_df

def move_video_directory(params):
    """
    Helper function to move a single video directory. 
    Used for parallel processing.
    """
    lang_id, speaker_id, video_id, src_dir, dst_dir = params
    
    # Source and destination paths
    src_path = os.path.join(src_dir, lang_id, speaker_id, video_id)
    dst_path = os.path.join(dst_dir, lang_id, speaker_id, video_id)
    
    # Create destination directory if it doesn't exist
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    
    # Move directory if it exists
    if os.path.exists(src_path):
        try:
            shutil.move(src_path, dst_path)
            return True, f"Moved {src_path} to {dst_path}"
        except Exception as e:
            return False, f"Error moving {src_path}: {str(e)}"
    elif os.path.exists(dst_path):
        return True, f"Moved clips from {src_path} to {dst_path}"
    else:
        return False, f"Source directory not found: {src_path}"

def move_files_parallel(df, src_dir, dst_dir):
    """
    Move files in parallel from source directory to destination directory by video_id
    """

    # Use 75% of available cores by default, but at least 1
    num_jobs = max(1, int(multiprocessing.cpu_count() * 0.75))

    # Get unique combinations of lang_id and video_id
    video_groups = df[['lang_id', 'speaker_id', 'video_id']].drop_duplicates()
    
    # Create a list of parameters for each video directory to move
    params_list = [(row['lang_id'], row['speaker_id'], row['video_id'], src_dir, dst_dir) 
                   for _, row in video_groups.iterrows()]
    
    success_count = 0
    error_count = 0

    # Use ProcessPoolExecutor for CPU-bound operations like file moving
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_jobs) as executor:
        # Use tqdm to show progress
        results = list(tqdm.tqdm(
            executor.map(move_video_directory, params_list), 
            total=len(params_list),
            desc=f"Moving files from {os.path.basename(src_dir)} to {os.path.basename(dst_dir)}"
        ))
        
        # Count successes and errors
        for success, message in results:
            if success:
                success_count += 1
            else:
                print (message, flush=True)
                error_count += 1
    
    print(f"Successfully moved {success_count} video directories. Encountered {error_count} errors.", flush=True)

def copy_clip(params):
    """
    Helper function to copy a single clip to the new directory structure.
    Used for parallel processing.
    """
    dirs, split, clip_info = params

    lang_id, speaker_id, video_id, clip_id, composite_key = clip_info[['lang_id', 'speaker_id', 'video_id', 'clip_id', 'composite_key']]
    
    # Source path: src_base_dir/split/lang_id/video_id/clip_id.*
    src_dir = os.path.join(src_base_dir, split, lang_id, speaker_id, video_id)
    
    # Skip if source directory doesn't exist
    if not os.path.exists(src_dir):
        return False, f"Source directory not found: {src_dir}"
    
    # Find all files that start with the clip_id
    try:
        clip_files = [f for f in os.listdir(src_dir) if f.startswith(clip_id)]
    except Exception as e:
        return False, f"Error listing directory {src_dir}: {str(e)}"
    
    if not clip_files:
        return False, f"No files found for clip_id {clip_id} in {src_dir}"
    
    # Destination path: dst_base_dir/video/split/lang_id/
    dst_dir = os.path.join(dst_base_dir, split, lang_id)
    
    # Create destination directory if it doesn't exist
    os.makedirs(dst_dir, exist_ok=True)
    
    success = False

    assert (len(clip_files) == 1)
    
    # Copy each matching file
    # for clip_file in clip_files:
    src_path = os.path.join(src_dir, clip_file)
    dst_path = os.path.join(dst_dir, composite_key)
        
    if os.path.isfile(src_path) and not os.path.exists(dst_path):
        try:
            shutil.copy2(src_path, dst_path)
            success = True
        except Exception as e:
            return False, f"Error copying {src_path} to {dst_path}: {str(e)}"
    if success:
        return True, f"Copied clip {clip_id} to {dst_dir}"
    else:
        return False, f"Failed to copy any files for clip {clip_id}"

def organize_clip_files(dirs, splits, filtered_split_dfs):
    """
    Organize clips into a new directory structure based on clip_id
    """
    print("\nOrganizing clips into video directory...", flush=True)
    
    # # Create video directory with split subdirectories
    # video_dir = os.path.join(dst_base_dir, 'video')
    # os.makedirs(video_dir, exist_ok=True)
    
    # for split in splits:
    #     os.makedirs(os.path.join(video_dir, split), exist_ok=True)
    
    total_success = 0
    total_errors = 0
    
    # Use 75% of available cores by default, but at least 1
    num_jobs = max(1, int(multiprocessing.cpu_count() * 0.75))
    
    # Process each split
    for i, split in enumerate(splits):
        if i >= len(filtered_split_dfs):
            continue
            
        split_df = filtered_split_dfs[i]
        split_df_fn = os.path.join(dirs['video'], split, f"{split}_metadata-filtered.csv")

        # Save the split information to the directory
        if not os.path.exists(split_df_fn):
            split_df.to_csv(split_df_fn, index=False)
        
        # Get unique clip information
        clip_info = split_df[['lang_id', 'speaker_id', 'video_id', 'clip_id', 'composite_key']].drop_duplicates().reset_index(drop=True)
        
        if len(clip_info) == 0:
            print(f"No clips found for {split} split", flush=True)
            continue
            
        # Create a list of parameters for each clip to copy
        params_list = [(dirs, split, row) for _, row in clip_info.iterrows()]
        
        print(f"Copying {len(params_list)} clips for {split} split...", flush=True)
        
        # Use ProcessPoolExecutor for CPU-bound operations like file copying
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_jobs) as executor:
            # Use tqdm to show progress
            results = list(tqdm.tqdm(
                executor.map(copy_clip, params_list),
                total=len(params_list),
                desc=f"Copying clips for {split} split"
            ))
            
            # Count successes and errors
            success_count = sum(1 for success, _ in results if success)
            error_count = sum(1 for success, _ in results if not success)
            
            total_success += success_count
            total_errors += error_count
        
        print(f"Successfully copied {success_count} clips for {split} split. Encountered {error_count} errors.", flush=True)
    
    # Print summary
    print("\nClip organization complete!")
    print(f"Total clips copied: {total_success}")
    print(f"Total errors: {total_errors}")
    
    # Check if any languages were created
    for split in splits:
        split_dir = os.path.join(video_dir, split)
        if os.path.exists(split_dir):
            languages = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
            print(f"{split} split: Created directories for {len(languages)} languages: {', '.join(languages)}")

def main():
    parser = argparse.ArgumentParser(description='Split and filter language data for all languages.')
    parser.add_argument('--base_dir', type=str, help='Base directory containing train and test folders')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation set ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--move_files', type=int, default=0, help='Move files to validation and low-confidence directories')
    parser.add_argument('--organize_clips', type=int, default=0, help='Organize clips into a new directory structure')
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    random.seed(args.seed)

    src_splits = ['dev', 'test']
    dest_splits = ['train', 'val', 'test']

    # Grab the source directories + make the new splits
    src_dirs, _ = utils.prepare_directory_structure(
        args.base_dir, 
        splits=dest_splits,
        dir_names=['src']
    )

    # Make the eventual destination directories
    dirs, _ = utils.prepare_directory_structure(
        args.base_dir, 
        splits=dest_splits,
        dir_names=['video']
    )
    
    dirs.update(src_dirs)

    #####################################
    ###### Load data for each split #####
    #####################################

    split_metadata = {}

    for split in src_splits:
        split_metadata_fn = os.path.join(dirs['src'], split, f'{split}_metadata.csv')

        if not os.path.exists(split_metadata_fn):
            # Grab split filenames
            split_pattern = os.path.join(dirs['src'], split, f'{split}_metadata-*.csv')
            split_fns = sorted(glob.glob(split_pattern))

            # Make into a single file
            df_metadata = pd.concat([pd.read_csv(fn) for fn in split_fns]).reset_index(drop=True)
            df_metadata.to_csv(split_metadata_fn, index=False)
        else:
            df_metadata = pd.read_csv(split_metadata_fn)

        # Add to dictionary 
        split_metadata[split] = df_metadata
    
    # Process train data
    print("Processing dev data...", flush=True)
    train_df, low_confidence_train = filter_confidence(split_metadata['dev']) # First filter based on confidence of language classification
    train_df = filter_speaker_languages(train_df) # Then filter to the most common language for the speaker
    
    # Split high confidence train data into train and validation sets
    train_df, val_df = split_train_val(train_df, args.val_ratio, args.seed)
    
    # Process test data
    print("Processing test data...", flush=True)
    test_df, low_confidence_test = filter_confidence(split_metadata['test']) # First filter based on confidence of language classification
    test_df = filter_confidence(test_df) # Then filter to the most common language for the speaker
    
    # Double check no overlap
    train_set = set(train_df['speaker_id'].unique())
    val_set = set(val_df['speaker_id'].unique())
    test_set = set(test_df['speaker_id'].unique())

    train_val_intersection = train_set.intersection(val_set)
    test_train_intersection = test_set.intersection(train_set)
    test_val_intersection = test_set.intersection(val_set)

    assert (not train_val_intersection and not test_train_intersection and not test_val_intersection)

    print (f"Number of train speakers: {len(train_set)}")
    print (f"Number of val speakers: {len(val_set)}")
    print (f"Number of test speakers: {len(test_set)}")

    # Save filtered metadata to CSVs
    print("Saving filtered metadata...", flush=True)
    
    # splits.append('val')
    dest_splits = ['train', 'val', 'test']
    filtered_split_dfs = [train_df, val_df, test_df]

    # Make the split dataframes
    for split, split_df in zip(dest_splits, filtered_split_dfs):
        fn = os.path.join(dirs['src'], split, f"{split}_metadata-filtered.csv") 

        print (f"{split} has {len(split_df)} rows", flush=True)

        if not os.path.exists(fn):
            split_df.to_csv(fn, index=False)

    # Move files if requested
    if args.move_files:
        print (f"Moving files...", flush=True)

        # Move train files (sourcing from dev to train)
        move_files_parallel(train_df, 
                           os.path.join(dirs['src'], 'dev'), 
                           os.path.join(dirs['src'], 'train'),
       )
        
        # Move validation files (sourcing from dev to val)
        move_files_parallel(val_df, 
                           os.path.join(dirs['src'], 'dev'), 
                           os.path.join(dirs['src'], 'val'),
       )
        
        # Move low confidence files from train
        move_files_parallel(low_confidence_train, 
                           os.path.join(dirs['src'], 'dev'), 
                           os.path.join(dirs['src'], 'low-confidence'),
        )
        
        # Move low confidence files from test
        move_files_parallel(low_confidence_test, 
                           os.path.join(dirs['src'], 'test'), 
                           os.path.join(dirs['src'], 'low-confidence'),
        )

        # Combine all low confidence data
        all_low_conf_df = pd.concat([low_confidence_train, low_confidence_test])
        all_low_conf_df.to_csv(os.path.join(dirs['src'], 'low-confidence', 'low_confidence_metadata.csv'), index=False)

        print("File moving complete!", flush=True)

    # Organize clips if requested
    if args.organize_clips:
        # Identify the parent directory (where src/ would be)
        parent_dir = os.path.dirname(args.base_dir)

        # Organize clips into the new directory structure
        organize_clip_files(filtered_split_dfs, splits, args.base_dir, parent_dir)

if __name__ == "__main__":
    main()
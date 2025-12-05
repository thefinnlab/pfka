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

def filter_data(df):
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
        
        # Create low confidence dataframe for this language
        low_conf_df = lang_df[~lang_df['video_id'].isin(valid_videos)]
        
        # Append to combined dataframes
        high_confidence_df = pd.concat([high_confidence_df, high_conf_df])
        low_confidence_df = pd.concat([low_confidence_df, low_conf_df])
        
        print(f"Language {lang_id.upper()}: {len(high_conf_df)} high confidence clips, {len(low_conf_df)} low confidence clips", flush=True)
    
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
        
        # Get unique video_ids for this language
        video_ids = lang_df['video_id'].unique()
        
        if len(video_ids) == 0:
            continue
            
        # Randomly select videos for validation
        num_val_videos = max(1, int(len(video_ids) * val_ratio))
        val_video_ids = random.sample(list(video_ids), num_val_videos)
        
        # Create train and validation dataframes
        lang_val_df = lang_df[lang_df['video_id'].isin(val_video_ids)]
        lang_train_df = lang_df[~lang_df['video_id'].isin(val_video_ids)]
        
        # Append to combined dataframes
        train_df = pd.concat([train_df, lang_train_df])
        val_df = pd.concat([val_df, lang_val_df])
        
        print(f"Language {lang_id.upper()}: {len(lang_train_df)} train clips, {len(lang_val_df)} validation clips", flush=True)
    
    return train_df, val_df

def move_video_directory(params):
    """
    Helper function to move a single video directory. 
    Used for parallel processing.
    """
    lang_id, video_id, src_dir, dst_dir = params
    
    # Source and destination paths
    src_path = os.path.join(src_dir, lang_id, video_id)
    dst_path = os.path.join(dst_dir, lang_id, video_id)
    
    # Create destination directory if it doesn't exist
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    
    # Move directory if it exists
    if os.path.exists(src_path):
        try:
            shutil.move(src_path, dst_path)
            return True, f"Moved {src_path} to {dst_path}"
        except Exception as e:
            return False, f"Error moving {src_path}: {str(e)}"
        # else:
        #     # If destination already exists, create it if needed
        #     os.makedirs(dst_path, exist_ok=True)
            
        #     # Get all clip files in the source directory
        #     try:
        #         clip_files = os.listdir(src_path)
        #     for clip_file in clip_files:
        #         clip_src = os.path.join(src_path, clip_file)
        #         clip_dst = os.path.join(dst_path, clip_file)
        #         if os.path.isfile(clip_src) and not os.path.exists(clip_dst):
        #             shutil.move(clip_src, clip_dst)
            
        #     # Remove source directory if empty
        #     if len(os.listdir(src_path)) == 0:
        #         os.rmdir(src_path)
    elif os.path.exists(dst_path):
        return True, f"Moved clips from {src_path} to {dst_path}"
            # except Exception as e:
            #     return False, f"Error moving clips from {src_path}: {str(e)}"
    else:
        return False, f"Source directory not found: {src_path}"

def move_files_parallel(df, src_dir, dst_dir):
    """
    Move files in parallel from source directory to destination directory by video_id
    """

    # Use 75% of available cores by default, but at least 1
    num_jobs = max(1, int(multiprocessing.cpu_count() * 0.75))

    # Get unique combinations of lang_id and video_id
    video_groups = df[['lang_id', 'video_id']].drop_duplicates()
    
    # Create a list of parameters for each video directory to move
    params_list = [(row['lang_id'], row['video_id'], src_dir, dst_dir) 
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
                error_count += 1
    
    print(f"Successfully moved {success_count} video directories. Encountered {error_count} errors.", flush=True)

def copy_clip(params):
    """
    Helper function to copy a single clip to the new directory structure.
    Used for parallel processing.
    """
    split, lang_id, video_id, clip_id, src_base_dir, dst_base_dir = params
    
    # Source path: src_base_dir/split/lang_id/video_id/clip_id.*
    src_dir = os.path.join(src_base_dir, split, lang_id, video_id)
    
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
    dst_dir = os.path.join(dst_base_dir, 'video', split, lang_id)
    
    # Create destination directory if it doesn't exist
    os.makedirs(dst_dir, exist_ok=True)
    
    success = False
    
    # Copy each matching file
    for clip_file in clip_files:
        src_path = os.path.join(src_dir, clip_file)
        dst_path = os.path.join(dst_dir, clip_file)
        
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

def organize_clip_files(filtered_split_dfs, splits, src_base_dir, dst_base_dir):
    """
    Organize clips into a new directory structure based on clip_id
    """
    print("\nOrganizing clips into video directory...", flush=True)
    
    # Create video directory with split subdirectories
    video_dir = os.path.join(dst_base_dir, 'video')
    os.makedirs(video_dir, exist_ok=True)
    
    for split in splits:
        os.makedirs(os.path.join(video_dir, split), exist_ok=True)
    
    total_success = 0
    total_errors = 0
    
    # Use 75% of available cores by default, but at least 1
    num_jobs = max(1, int(multiprocessing.cpu_count() * 0.75))
    
    # Process each split
    for i, split in enumerate(splits):
        if i >= len(filtered_split_dfs):
            continue
            
        split_df = filtered_split_dfs[i]
        split_df_fn = os.path.join(video_dir, split, f"{split}_metadata-filtered.csv")

        # Save the split information to the directory
        if not os.path.exists(split_df_fn):
            split_df.to_csv(split_df_fn, index=False)
        
        # Get unique clip information
        clip_info = split_df[['lang_id', 'video_id', 'clip_id']].drop_duplicates().reset_index(drop=True)
        
        if len(clip_info) == 0:
            print(f"No clips found for {split} split", flush=True)
            continue
            
        # Create a list of parameters for each clip to copy
        params_list = [(split, row['lang_id'], row['video_id'], row['clip_id'], src_base_dir, dst_base_dir)
                      for _, row in clip_info.iterrows()]
        
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
    
    # Create val and low-confidence directories if they don't exist
    os.makedirs(os.path.join(args.base_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(args.base_dir, 'low-confidence'), exist_ok=True)
    
    # Load metadata
    splits = ['train', 'test']
    split_metadata = {}
    
    for split in splits:
        split_metadata_fn = os.path.join(args.base_dir, split, f'{split}_metadata.csv')

        # if not os.path.exists(split_metadata_fn):
        split_fns = sorted(glob.glob(os.path.join(args.base_dir, split, f'{split}_metadata-*.csv')))
        df_metadata = pd.concat([pd.read_csv(fn) for fn in split_fns]).reset_index(drop=True)
        # df_metadata['composite_key'] = df_metadata['clip_id']
        df_metadata.to_csv(split_metadata_fn, index=False)
        # else:
        #     df_metadata = pd.read_csv(split_metadata_fn)

        split_metadata[split] = df_metadata
    
    # Process train data
    print("Processing train data...", flush=True)
    train_df, low_confidence_train = filter_data(split_metadata['train'])
    
    # Split high confidence train data into train and validation sets
    train_df, val_df = split_train_val(train_df, args.val_ratio, args.seed)
    
    # Process test data
    print("Processing test data...", flush=True)
    test_df, low_confidence_test = filter_data(split_metadata['test'])
    
    # Save filtered metadata to CSVs
    print("Saving filtered metadata...", flush=True)
    
    # splits.append('val')
    splits = ['train', 'val', 'test']
    filtered_split_dfs = [train_df, val_df, test_df]

    # Make the split dataframes
    for split, split_df in zip(splits, filtered_split_dfs):
        fn = os.path.join(args.base_dir, split, f"{split}_metadata-filtered.csv") 

        # if not os.path.exists(fn):
        split_df.to_csv(fn, index=False)

    # Move files if requested
    if args.move_files:
        print (f"Moving files...", flush=True)
        # Move validation files
        move_files_parallel(val_df, 
                           os.path.join(args.base_dir, 'train'), 
                           os.path.join(args.base_dir, 'val'),
       )
        
        # Move low confidence files from train
        move_files_parallel(low_confidence_train, 
                           os.path.join(args.base_dir, 'train'), 
                           os.path.join(args.base_dir, 'low-confidence'),
        )
        
        # Move low confidence files from test
        move_files_parallel(low_confidence_test, 
                           os.path.join(args.base_dir, 'test'), 
                           os.path.join(args.base_dir, 'low-confidence'),
        )

        # Combine all low confidence data
        all_low_conf_df = pd.concat([low_confidence_train, low_confidence_test])
        all_low_conf_df.to_csv(os.path.join(args.base_dir, 'low-confidence', 'low_confidence_metadata.csv'), index=False)

        print("File moving complete!", flush=True)

    # Organize clips if requested
    if args.organize_clips:
        # Identify the parent directory (where src/ would be)
        parent_dir = os.path.dirname(args.base_dir)

        # Organize clips into the new directory structure
        organize_clip_files(filtered_split_dfs, splits, args.base_dir, parent_dir)

if __name__ == "__main__":
    main()
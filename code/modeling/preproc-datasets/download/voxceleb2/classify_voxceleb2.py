import os
import sys
import glob
import argparse
from tqdm import tqdm
import random
from natsort import natsorted

import shutil
import pandas as pd
import numpy as np

from pathlib import Path

# Assuming these imports work in your environment
sys.path.append('../../../../utils/')

from config import *
from dataset_utils import attempt_makedirs

sys.path.append('../../utils/')

import utils

def process_speaker(dirs, split, speaker_id, processor, model, existing_df=None, batch_size=8):

    speaker_dir = os.path.join(dirs['src'], speaker_id)
    video_ids = os.listdir(speaker_dir)

    processed_files = []
    
    for vid in tqdm(video_ids, desc=f"Processing {speaker_id} videos"):
        try:
            # Directory of the current video
            vid_dir = os.path.join(speaker_dir, vid)

            # Grab all the videos and load as a batch
            vid_fns = natsorted(glob.glob(os.path.join(vid_dir, '*')))
            clip_ids = [os.path.basename(fn) for fn in vid_fns] # Get clip IDs from filenames

            # Filter out clips that already exist in the dataframe
            existing_clip_ids = set()

            if existing_df is not None:
                speaker_video_mask = (existing_df['speaker_id'] == speaker_id) & (existing_df['video_id'] == vid)
                existing_clip_ids = set(existing_df.loc[speaker_video_mask, 'clip_id'].tolist())

            # Get only new clip IDs and their corresponding file paths
            unprocessed_ids = [(fn, clip_id) for fn, clip_id in zip(vid_fns, clip_ids) if clip_id not in existing_clip_ids]

            if not unprocessed_ids:
                print(f"All clips for speaker {speaker_id}, video {vid} already processed. Skipping.")
                continue  # Skip to the next speaker/video
            
            # Unzip and pack back into lists
            vid_fns, clip_ids = zip(*unprocessed_ids)

            # Process batches
            results = {
                'detected_langs': [],
                'probs': []
            }

            for i in range(0, len(vid_fns), batch_size):
                batch = utils.prepare_audio_batch(vid_fns)
                detected_langs, probs =  utils.classify_language(batch, processor, model, return_probs=True)

                results['detected_langs'].extend(detected_langs)
                results['probs'].extend(probs)

            # Unpack results after batching
            detected_langs = results['detected_langs']
            probs = results['probs']

            # Get counts of each language
            unique_langs, counts = np.unique(detected_langs, return_counts=True)

            # Find the most common language
            max_count = np.argmax(counts)
            lang_ratio = np.max(counts) / len(detected_langs)
            most_common_lang = unique_langs[max_count]

            # Create destination directory with language subfolder
            dest_path = os.path.join(dirs['dest'], most_common_lang, speaker_id, vid)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            
            # Log language data for each clip
            for i, (clip_id, lang, prob) in enumerate(zip(clip_ids, detected_langs, probs)):
                # Add to language log
                lang_info = {
                    'speaker_id': speaker_id,
                    'video_id': vid,
                    'clip_id': clip_id,
                    'lang_id': lang,
                    'prob': prob.item(),
                    'split': split,
                    'most_common_lang': most_common_lang,
                    'percent_clips_lang': lang_ratio,
                }

                processed_files.append(lang_info)

            # Move the video directory to the destination
            shutil.copytree(vid_dir, dest_path, dirs_exist_ok=True)
        # processed_files[split]['video_ids'].append(vid)
        except Exception as e:
            print(f"Error moving {vid}: {e}", flush=True)

    return processed_files

def find_processed_speakers(speaker_ids, split_dir, existing_df):
    """Identify speakers whose data has already been fully processed."""
    
    processed_speakers = set()

    if existing_df is None:
        return processed_speakers
    
    # Group by speaker for more efficient lookup
    speaker_group = existing_df.groupby('speaker_id')
    
    for speaker in speaker_ids:
        speaker_dir = os.path.join(split_dir, speaker)
        
        # Skip if directory doesn't exist
        if not os.path.exists(speaker_dir):
            processed_speakers.add(speaker)
            continue
        
        # If speaker not in dataframe, it's not processed
        if speaker not in speaker_group.groups:
            continue
        
        # Get speaker data once
        speaker_data = existing_df[existing_df['speaker_id'] == speaker]
        existing_videos = set(speaker_data['video_id'].unique())
        
        # Check all videos and clips
        all_processed = True
        for vid in os.listdir(speaker_dir):
            if vid not in existing_videos:
                all_processed = False
                break
            
            # Check if all clips in this video are processed
            vid_dir = os.path.join(speaker_dir, vid)
            clip_files = glob.glob(os.path.join(vid_dir, '*'))
            clip_ids = [os.path.basename(fn) for fn in clip_files]
            
            # Get processed clips for this speaker+video
            mask = (speaker_data['video_id'] == vid)
            processed_clips = set(speaker_data.loc[mask, 'clip_id'].tolist())
            
            # If any clip is missing, speaker is not fully processed
            if not all(clip_id in processed_clips for clip_id in clip_ids):
                all_processed = False
                break
        
        if all_processed:
            processed_speakers.add(speaker)
    
    return processed_speakers

def save_progress(processed_data, lang_log_path, existing_df=None, overwrite=False):
    """Save current progress to CSV file."""

    # Create a temporary dataframe with the current batch
    temp_df = pd.DataFrame(processed_data)
    
    # If the file exists, load and append to it
    if os.path.exists(lang_log_path) and not overwrite:
        # If we have an existing dataframe already loaded, use it
        if existing_df is not None:
            existing_data = existing_df
        else:
            # Otherwise load from file
            existing_data = pd.read_csv(lang_log_path)

        # Make a composite key
        temp_df['composite_key'] = temp_df['speaker_id'] + '_' + temp_df['video_id'] + '_' + temp_df['clip_id']
        
        # Create composite keys for the existing dataframe
        if (existing_data is not None) and (not existing_data.empty):
            # Make the composite key if it doesn't exist
            if 'composite_key' not in existing_data.columns:
                existing_data['composite_key'] = existing_data['speaker_id'] + '_' + existing_data['video_id'] + '_' + existing_data['clip_id']

            # Filter out rows that have composite keys already in the existing dataframe
            existing_keys = set(existing_data['composite_key'])
            temp_df = temp_df[~temp_df['composite_key'].isin(existing_keys)]
        
        # Concatenate the dataframes if there's new data
        if not temp_df.empty:
            combined_df = pd.concat([existing_data, temp_df], ignore_index=True)
            combined_df.to_csv(lang_log_path, index=False)
            print(f"Updated progress at {lang_log_path} with {len(temp_df)} new entries", flush=True)
            return combined_df
        return existing_data
    else:
        # Write new file if it doesn't exist
        # Make sure to remove the composite_key column if it exists
        temp_df.to_csv(lang_log_path, index=False)
        print(f"Created progress file at {lang_log_path} with {len(temp_df)} entries", flush=True)
        return temp_df

def main():
    parser = argparse.ArgumentParser(description='Split and filter language data for all languages.')
    parser.add_argument('--base_dir', type=str, help='Base directory containing train and test folders')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation set ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--num_shards', type=int, default=1, help='Number of shards to divide the dataset into')
    parser.add_argument('--current_shard', type=int, default=0, help='Current shard to process (0-based indexing)')
    parser.add_argument('--overwrite', type=int, default=0, help='Current shard to process (0-based indexing)')
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    random.seed(args.seed)

    splits = ['dev', 'test']

    # Grab the source directories
    dirs, _ = utils.prepare_directory_structure(
        args.base_dir, 
        dir_names=['src']
    )

    #####################################
    ######## Language classifier ########
    #####################################

    # Load models for language classification
    processor, model = utils.load_language_classifier()

    # Initialize language log dataframe
    all_language_data = []

    #####################################
    ######## Process each video #########
    #####################################

    speaker_data = {}

    for split in splits:

        # Grab the language data if it exists
        lang_log_path = os.path.join(dirs['src'], split, f'{split}_metadata-{str(args.current_shard + 1).zfill(5)}.csv')

        # Load dataframe if it exists
        if os.path.exists(lang_log_path) and not args.overwrite:
            existing_df = pd.read_csv(lang_log_path)
        else:
            existing_df = None

        # Grab unique speaker IDs from the split directory
        split_dirs = {k: os.path.join(v, 'video', split) for k, v in dirs.items()}
        split_dirs['dest'] = os.path.join(dirs['src'], split)
        speaker_ids = os.listdir(split_dirs['src'])
        
        # Get shard ids
        speaker_ids = utils.get_shard_data(speaker_ids, num_shards=args.num_shards, current_shard=args.current_shard)

        # Identify fully processed speakers
        fully_processed_speakers = find_processed_speakers(speaker_ids, split_dirs['src'], existing_df)
 
        print(f"Skipping {len(fully_processed_speakers)} fully processed speakers out of {len(speaker_ids)} total")
               
        # Filter out speakers that are completely processed
        speaker_ids = [s for s in speaker_ids if s not in fully_processed_speakers]

        # Go through each remaining speaker and process their language data 
        for i, speaker in enumerate(speaker_ids):
            print (f"Processing speaker {speaker} // {i+1}/{len(speaker_ids)}", flush=True)

            # Add the processed data to the list
            processed_data = process_speaker(split_dirs, split, speaker, processor, model, existing_df=existing_df, batch_size=4)

            # Save progress after each speaker
            if processed_data:
                existing_df = save_progress(processed_data, lang_log_path, existing_df, overwrite=args.overwrite)
                
            print(f"Completed processing speaker {speaker}", flush=True)
    
    print("Processing completed for all splits and speakers.", flush=True)
    
if __name__ == "__main__":
    main()
# process_speech.py
import os, sys
import glob
import argparse
import math
from tqdm import tqdm
from torchvision.io import read_video
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

sys.path.append('../utils/')

import utils

def process_video_file(video_path, video_dir, audio_dir, target_sr=16000):
    """
    Process a single video file to extract audio
    
    Args:
        video_path (str): Path to video file
        video_dir (str): Base video directory
        audio_dir (str): Base audio directory
        target_sr (int): Target sampling rate
        
    Returns:
        tuple: (success, video_path, message)
    """
    try:
        # Determine relative path to maintain directory structure
        rel_path = os.path.relpath(video_path, video_dir)
        audio_path = os.path.join(audio_dir, os.path.splitext(rel_path)[0] + '.wav')
        
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(audio_path), exist_ok=True)
        
        # Skip if audio file already exists
        if os.path.exists(audio_path):
            return True, video_path, "Already exists"
            
        # Extract audio using torchvision's read_video
        video_data, audio_data, info = read_video(video_path, pts_unit="sec")
        
        # Get the sampling rate
        original_sr = info["audio_fps"]
        
        # Convert audio to numpy array
        audio_array = audio_data.numpy()[0]  # Mono channel
        
        # Resample if needed
        audio_array, _ = utils.resample_audio(audio_array, original_sr, target_sr)
        
        # Save audio
        utils.save_audio_file(audio_array, audio_path, target_sr)
        
        return True, video_path, "Success"
        
    except Exception as e:
        return False, video_path, str(e)

def extract_audio_from_video(dirs, split, target_sr=16000, force=False, num_jobs=None, num_shards=1, current_shard=0):
    """
    Extract audio from video files and resample to target sampling rate (parallel version)
    
    Args:
        dirs (dict): Dictionary of directory paths
        split (str): Dataset split
        target_sr (int): Target sampling rate
        force (bool): Force extraction even if audio files exist
        num_workers (int): Number of worker processes to use
        
    Returns:
        str: Path to audio directory
    """
    if num_jobs is None:
        # Use 75% of available cores by default, but at least 1
        num_jobs = max(1, int(multiprocessing.cpu_count() * 0.75))

    # video_dir = os.path.join(dirs['video'], split)
    # audio_dir = os.path.join(dirs['audio'], split)
    
    # Get all video files
    video_files = sorted(glob.glob(os.path.join(dirs['video'], "*.mp4"), recursive=True))

    utils.get

    video_files = utils.get_shard_data(video_files, num_shards=num_shards, current_shard=current_shard)
    
    # Count existing audio files and find files to process
    existing_count = 0
    to_process = []
    
    for video_path in video_files:
        rel_path = os.path.relpath(video_path, dirs['video'])
        audio_path = os.path.join(dirs['audio'], os.path.splitext(rel_path)[0] + '.wav')
        if os.path.exists(audio_path) and not force:
            existing_count += 1
        else:
            to_process.append(video_path)
    
    to_process_count = len(to_process)
    
    if to_process_count == 0:
        print(f"All audio files already exist for {split} split. Skipping extraction.", flush=True)
        return dirs['audio']
    
    print(f"Found {existing_count} existing audio files and {to_process_count} files to process", flush=True)
    print(f"Using {num_jobs} worker processes for parallel extraction", flush=True)
    
    # Create output directories
    os.makedirs(dirs['audio'], exist_ok=True)
    
    # Process files in parallel
    completed = 0
    errors = 0
    
    with ProcessPoolExecutor(max_workers=num_jobs) as executor:
        # Submit all tasks
        future_to_video = {
            executor.submit(process_video_file, video_path, dirs['video'], dirs['audio'], target_sr): video_path 
            for video_path in to_process
        }
        
        # Process results as they complete
        pbar = tqdm(total=to_process_count, desc=f"Extracting audio from {split} videos")
        
        for future in as_completed(future_to_video):
            video_path = future_to_video[future]
            try:
                success, _, message = future.result()
                if success:
                    completed += 1
                else:
                    errors += 1
                    print(f"Error processing {video_path}: {message}", flush=True)
            except Exception as e:
                errors += 1
                print(f"Exception processing {video_path}: {str(e)}", flush=True)
            
            pbar.update(1)
        
        pbar.close()
    
    print(f"Audio extraction complete: {completed} successful, {errors} failed", flush=True)
    return dirs['audio']

def main():
    parser = argparse.ArgumentParser(description='Process speech datasets to Praat TextGrids')
    parser.add_argument('-d','--dataset', type=str, choices=['lrs3', 'avspeech', 'voxceleb2'], required=True, 
                    help='Which dataset to process')
    parser.add_argument('--output_dir', type=str, default=None,
                    help='Base directory for output (default: dataset_name_processing)')
    parser.add_argument('--target_sr', type=int, default=16000,
                    help='Target sampling rate for audio (default: 16000 Hz)')
    parser.add_argument('--split', type=str, default=None,
                    help='Which split to process')
    parser.add_argument('--lang_id', type=str, default='eng',
                    help='Language ID ISO-639 code for AVSpeech')
    parser.add_argument('--num_jobs', type=int, default=None,
                    help='Number of worker processes for parallel extraction (default: 75% of CPU cores)')
    parser.add_argument('--num_shards', type=int, default=1,
                    help='Number of shards to divide the dataset into')
    parser.add_argument('--current_shard', type=int, default=0,
                    help='Current shard to process (0-based indexing)')
    parser.add_argument('--overwrite', type=int, default=0,
                    help='Force extraction even if files exist')
    
    args = parser.parse_args()

    if args.split:
        splits = [args.split]
    else:
        splits = ['train', 'val', 'test']

    # Determine output directory
    output_dir = args.output_dir or f"{args.dataset}_processing"

    # Prepare directory structure --> only this script is needed for video
    dirs, splits = utils.prepare_directory_structure(
        output_dir, 
        splits, 
        dir_names=['audio', 'video'],
    )

    # Process each split
    for split in splits:
        print(f"\nProcessing {args.dataset} {split} split...", flush=True)
        print(f"\nCurrent shard: {args.current_shard+1}/{args.num_shards}", flush=True)

        split_dirs = {k: os.path.join(v, split) for k, v in dirs.items()}

        if args.dataset == 'avspeech':
            
            if args.lang_id:
                lang_dirs = {k: os.path.join(v, args.lang_id) for k, v in split_dirs.items()}
            else:
                continue

            extract_audio_from_video(lang_dirs, split, args.target_sr, force=args.overwrite, num_jobs=args.num_jobs, num_shards=args.num_shards, current_shard=args.current_shard)
        else:
            # Extract audio from videos
            print(f"Extracting {args.dataset} {split} data...", flush=True)
            extract_audio_from_video(split_dirs, split, args.target_sr, force=args.overwrite, num_jobs=args.num_jobs, num_shards=args.num_shards, current_shard=args.current_shard)
        
    print("\nProcessing complete!", flush=True)

if __name__ == "__main__":
    main()
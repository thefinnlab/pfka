# process_speech.py
import os, sys
import glob
import argparse
from tqdm import tqdm
import shutil
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

sys.path.append('../utils/')

import utils

def process_corpus_file(dirs, split, audio_path):
    '''
    audio_path is the path to the existing audio file
    '''

    try:
        # Determine relative path
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        transcript_path = os.path.join(dirs['transcripts'], base_name + '.txt')

        # Skip if transcript doesn't exist
        if not os.path.exists(transcript_path):
            return False, transcript_path, "Missing transcript file"
        
        # Define corpus paths
        corpus_audio_path = os.path.join(dirs['corpus'], f"{base_name}.wav")
        corpus_transcript_path = os.path.join(dirs['corpus'], f"{base_name}.txt")
                
        # Copy files to corpus
        os.makedirs(os.path.dirname(corpus_audio_path), exist_ok=True)
        os.makedirs(os.path.dirname(corpus_transcript_path), exist_ok=True)
        
        shutil.copy2(audio_path, corpus_audio_path)
        shutil.copy2(transcript_path, corpus_transcript_path)
            
        return True, audio_path, "Success"    
    except Exception as e:
        return False, audio_path, str(e)

def prepare_corpus(dirs, split, num_jobs=None, num_shards=1, current_shard=0, force=False):
    """
    Prepare corpus directory with paired .wav and .txt files for MFA
    
    Args:
        dirs (dict): Dictionary of directory paths
        split (str): Dataset split
        force (bool): Force corpus preparation even if files exist
        
    Returns:
        str: Path to corpus directory
    """

    if num_jobs is None:
        # Use 75% of available cores by default, but at least 1
        num_jobs = max(1, int(multiprocessing.cpu_count() * 0.75))
    
    # Get all audio files
    audio_files = sorted(glob.glob(os.path.join(dirs['audio'], "*.wav"), recursive=True))

    # Apply sharding logic
    if num_shards > 1:
        # Calculate shard size and starting/ending indices
        shard_size = math.ceil(len(audio_files) / num_shards)
        start_idx = current_shard * shard_size
        end_idx = min(start_idx + shard_size, len(audio_files))
        
        # Get only the files for the current shard
        audio_files = audio_files[start_idx:end_idx]
        
        print(f"Processing shard {current_shard+1}/{num_shards} with {len(audio_files)} files", flush=True)

    # Count existing audio files and find files to process
    existing_count = 0
    to_process = []
    
    for audio_path in audio_files:
        rel_path = os.path.relpath(audio_path, dirs['audio'])
        corpus_audio_path = os.path.join(dirs['corpus'], os.path.splitext(rel_path)[0] + '.wav')
        corpus_transcript_path = os.path.join(dirs['corpus'], os.path.splitext(rel_path)[0] + '.txt')

        if os.path.exists(corpus_audio_path) and os.path.exists(corpus_transcript_path) and not force:
            existing_count += 1
        else:
            to_process.append(audio_path)
    
    to_process_count = len(to_process)
    
    if to_process_count == 0:
        print(f"All audio files already exist for {split} split. Skipping extraction.", flush=True)
        return dirs['audio']
    
    print(f"Found {existing_count} existing audio files and {to_process_count} files to process", flush=True)
    print(f"Using {num_jobs} worker processes for parallel extraction", flush=True)

    # Process files in parallel
    completed = 0
    errors = 0

    with ProcessPoolExecutor(max_workers=num_jobs) as executor:
        # Submit all tasks
        future_to_audio = {
            executor.submit(process_corpus_file, dirs, split, audio_path): audio_path 
            for audio_path in to_process
        }
        
        # Process results as they complete
        pbar = tqdm(total=to_process_count, desc=f"Preparing corpus for {split}")
        
        for future in as_completed(future_to_audio):
            audio_path = future_to_audio[future]
            try:
                success, _, message = future.result()
                if success:
                    completed += 1
                else:
                    errors += 1
                    print(f"Error processing {audio_path}: {message}", flush=True)
            except Exception as e:
                errors += 1
                print(f"Exception processing {audio_path}: {str(e)}", flush=True)
            
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
                    help='Force corpus copying even if files exist')
    
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
        dir_names=['audio','transcripts', 'corpus'],
    )

    # Process each split
    for split in splits:
        print(f"\nProcessing {args.dataset} {split} split...", flush=True)
        print(f"\nCurrent shard: {args.current_shard+1}/{args.num_shards}", flush=True)

        split_dirs = {k: os.path.join(v, split) for k, v in dirs.items()}

        print(f"\nPreparing corpus for {split} split...", flush=True)

        if args.dataset == 'avspeech':
            
            if args.lang_id:
                lang_dirs = {k: os.path.join(v, args.lang_id) for k, v in split_dirs.items()}
            else:
                continue

            prepare_corpus(lang_dirs, split, num_jobs=args.num_jobs, num_shards=args.num_shards, current_shard=args.current_shard, force=args.overwrite)
        else:
            #Prepare corpus
            prepare_corpus(split_dirs, split, num_jobs=args.num_jobs, num_shards=args.num_shards, current_shard=args.current_shard, force=args.overwrite)

    print("\nProcessing complete!", flush=True)

if __name__ == "__main__":
    main()
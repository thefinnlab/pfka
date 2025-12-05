# process_speech.py
import os
import argparse
import subprocess
from datasets import load_dataset
from praatio import textgrid
import soundfile as sf
from tqdm import tqdm
import shutil
import librosa

sys.path.append('../')

import utils

N_FILES = 3

def extract_dataset_data(dirs, dataset_name, split, subset_size=None, num_jobs=None):
    """Download and extract dataset for a specific split"""

    # Normalize split name
    norm_split = split.replace('.', '-')
    
    config = utils.DATASET_CONFIGS[dataset_name]
    corpus_dir = os.path.join(dirs['corpus'], norm_split)
    audio_dir = os.path.join(dirs['audio'], norm_split)
    transcript_dir = os.path.join(dirs['transcripts'], norm_split)
    
    try:
        # Load dataset split
        dataset = load_dataset(
            config['name'],
            config['config'],
            split=split,
            num_proc=num_jobs,
            trust_remote_code=True,
        )

        # Apply subset if specified
        if subset_size is not None and dataset_name != 'emilia':
            dataset = dataset.select(range(min(subset_size, len(dataset))))
        
        # Process each audio file and its transcript
        for idx, item in enumerate(tqdm(dataset, desc=f"Processing {split} split")):
            try:                
                # Handle different audio formats
                if isinstance(item[config['audio_key']], dict):
                    audio_data = item[config['audio_key']]['array']
                    sampling_rate = item[config['audio_key']]['sampling_rate']
                    segment_id = item[config['segment_key']]
                else:
                    audio_data = item[config['audio_key']]
                    sampling_rate = 16000  # Default for datasets that don't specify
                    segment_id = item[config['segment_key']]

                # Clean up segment ID for peoples-speech
                if dataset_name == 'peoples-speech':
                    segment_id = os.path.basename(segment_id)

                # Resample audio if needed
                audio_data, sampling_rate = resample_audio(audio_data, sampling_rate, config['target_sr'])

                # Define paths
                audio_path = os.path.join(audio_dir, f"{segment_id}.wav")
                transcript_path = os.path.join(transcript_dir, f"{segment_id}.txt")
                corpus_audio_path = os.path.join(corpus_dir, f"{segment_id}.wav")
                corpus_transcript_path = os.path.join(corpus_dir, f"{segment_id}.txt")
                
                # Save audio files
                utils.save_audio_file(audio_data, audio_path, sampling_rate)
                utils.save_audio_file(audio_data, corpus_audio_path, sampling_rate)
                
                # Save transcript files
                utils.save_transcript(item[config['text_key']], transcript_path)
                utils.save_transcript(item[config['text_key']], corpus_transcript_path)
                    
            except Exception as e:
                print(f"Error processing item {idx} in {split} split: {str(e)}", flush=True)
                continue
                
        return True
                
    except Exception as e:
        print(f"Error loading {split} split: {str(e)}", flush=True)
        return False

def main():
    parser = argparse.ArgumentParser(description='Process speech datasets to Praat TextGrids')
    parser.add_argument('--dataset', type=str, choices=['gigaspeech', 'peoples-speech', 'libritts-r', 'tedlium'],
                      required=True, help='Which dataset to process')
    parser.add_argument('--subset_size', type=int, default=None,
                      help='Number of samples to process per split (default: all)')
    parser.add_argument('--output_dir', type=str, default=None,
                      help='Base directory for output (default: dataset_name_processing)')
    parser.add_argument('--num_jobs', type=int, default=1,
                      help='Number of parallel jobs for MFA alignment')
    
    args = parser.parse_args()

    # Find dataset splits
    # dataset_splits = utils.DATASET_CONFIGS[args.dataset]['splits']
    config = utils.DATASET_CONFIGS[args.dataset]

    # Determine output directory
    output_dir = args.output_dir or f"{args.dataset}_processing"

    # Prepare directory structure
    dirs, normalized_splits = utils.prepare_directory_structure(output_dir, config['splits'])

    # Process each split
    for split in config['splits']:
        print(f"\nProcessing {args.dataset} {split} split...", flush=True)
        
        # Step 1: Extract dataset data
        print(f"Extracting {args.dataset} {split} data...", flush=True)
        extract_dataset_data(dirs, args.dataset, split, args.subset_size, args.num_jobs)

    print("\nProcessing complete!", flush=True)

if __name__ == "__main__":
    main()
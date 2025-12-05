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

N_FILES = 3

DATASET_CONFIGS = {
    'gigaspeech': {
        'name': 'speechcolab/gigaspeech',
        'config': 'm',
        'splits': ['train', 'validation', 'test'],
        'audio_key': 'audio',
        'text_key': 'text',
        'segment_key': 'segment_id',
        'target_sr': 16000  # Use original sampling rate
    },
    'peoples-speech': {
        'name': 'MLCommons/peoples_speech',
        'config': 'clean_sa',
        'splits': ['train'], #'validation', 'test'], # 
        'audio_key': 'audio',
        'text_key': 'text',
        'segment_key': 'id',
        'target_sr': 16000  # People's Speech needs consistent sampling rate
    },
    'emilia': {
        'name': "amphion/Emilia-Dataset",
        'data_files': {"en": [f"EN/EN_B0000{i}.tar" for i in range(N_FILES)]}, #For all files: "EN/*.tar"
        'splits': ['train', 'validation', 'test'],
        'audio_key': 'mp3',
        'text_key': 'json',
        'segment_key': '__key__',
        'target_sr': 16000
    },
    'libritts-r': {
        'name': 'mythicinfinity/libritts_r',
        'config': 'clean',
        'splits': ['train.clean.360', 'dev.clean', 'test.clean'],
        'audio_key': 'audio',
        'text_key': 'text_normalized',
        'segment_key': 'id',
        'target_sr': 16000  # Use original sampling rate
    },
    'tedlium': {
        'name': 'LIUM/tedlium',
        'config': 'release3',
        'splits': ['train', 'validation', 'test'],
        'audio_key': 'audio',
        'text_key': 'text',
        'segment_key': 'id',
        'target_sr': 16000  # Use original sampling rate
    }
}

def prepare_data_directory(base_dir, splits):
    """Create necessary directories for MFA processing"""
    dirs = {
        'audio': os.path.join(base_dir, 'audio'),
        'transcripts': os.path.join(base_dir, 'transcripts'),
        'aligned': os.path.join(base_dir, 'aligned'),
        'textgrids': os.path.join(base_dir, 'textgrids'),
        'corpus': os.path.join(base_dir, 'corpus')  # Temporary directory for MFA
    }
    
    # Create main directories
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    splits = [split.replace('.', '-') for split in splits]
        
    # Create split-specific subdirectories
    for split in splits:
        for key in dirs:
            split_dir = os.path.join(dirs[key], split)
            os.makedirs(split_dir, exist_ok=True)
    
    return dirs

def extract_dataset_data(dirs, dataset_name, split, subset_size=None, num_jobs=None):
    """Download and extract dataset for a specific split"""

    # Rename libritts-r weird naming convention
    _split = split.replace('.', '-')

    config = DATASET_CONFIGS[dataset_name]
    corpus_dir = os.path.join(dirs['corpus'], _split)
    
    try:
        # Load dataset split
        dataset = load_dataset(
            config['name'],
            config['config'],
            split=split,
            num_proc=num_jobs,
            trust_remote_code=True,
        )

        # # Write all audio to the same sampling rate
        # sampling_rate = config['target_sr']

        # Apply subset if specified
        if subset_size is not None:
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
                    segment_id = item[config['segment_key']]

                # Resample audio if target_sr different from sampling rate
                if sampling_rate != config['target_sr']:
                    audio_data = librosa.resample(audio_data, orig_sr=sampling_rate, target_sr=config['target_sr'])
                    sampling_rate = config['target_sr']

                # Extract audio
                if dataset_name == 'peoples-speech':
                    segment_id = os.path.basename(segment_id)

                audio_path = os.path.join(corpus_dir, f"{segment_id}.wav")
                
                # Save audio to corpus directory
                sf.write(audio_path, audio_data, sampling_rate)
                
                # Save transcript to corpus directory
                transcript_path = os.path.join(corpus_dir, f"{segment_id}.txt")
                with open(transcript_path, 'w', encoding='utf-8') as f:
                    f.write(item[config['text_key']])
                    
                # Also save copies to original directories
                shutil.copy2(audio_path, os.path.join(dirs['audio'], _split, f"{segment_id}.wav"))
                shutil.copy2(transcript_path, os.path.join(dirs['transcripts'], _split, f"{segment_id}.txt"))
                    
            except Exception as e:
                print(f"Error processing item {idx} in {split} split: {str(e)}")
                continue
                
    except Exception as e:
        print(f"Error loading {split} split: {str(e)}")

def run_mfa_alignment(dirs, split, num_jobs=1):
    """Run Montreal Forced Aligner on the prepared data using command line interface"""

    _split = split.replace('.', '-')
    
    try:
        corpus_dir = os.path.join(dirs['corpus'], _split)
        aligned_dir = os.path.join(dirs['textgrids'], _split)
        
        cmd = ' '.join([
            'mfa',
            'align',
            corpus_dir,
            'english_us_arpa',
            'english_us_arpa',
            aligned_dir,
            '--clean',
            '--overwrite',
            '--verbose'
            '--output_format', 'long_textgrid',
            '--speaker_characters', str(13),
            '--temp_directory', os.path.join(aligned_dir, 'temp'),
            '--num_jobs', str(num_jobs)
        ])
        
        # Run MFA command
        process = subprocess.run(
            cmd,
            shell=True
            # check=True,
            # text=True,
            # capture_output=True
        )
        
        if process.returncode != 0:
            print(f"MFA alignment failed for {split} split")
            print("Error output:", process.stderr)
        else:
            print(f"MFA alignment completed successfully for {split} split")
            
    except subprocess.CalledProcessError as e:
        print(f"Error running MFA for {split} split: {str(e)}")
        print("Error output:", e.stderr)
    except Exception as e:
        print(f"Unexpected error running MFA for {split} split: {str(e)}")
    finally:
        # Clean up corpus directory after alignment
        try:
            shutil.rmtree(os.path.join(dirs['corpus'], split))
            print(f"Cleaned up temporary corpus directory for {split} split")
        except Exception as e:
            print(f"Error cleaning up corpus directory: {str(e)}")

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
    dataset_splits = DATASET_CONFIGS[args.dataset]['splits']
    
    # Set up base directory
    base_dir = args.output_dir or f"{args.dataset}_processing"
    dirs = prepare_data_directory(base_dir, dataset_splits)
    
    # Process each split
    for split in dataset_splits:
        print(f"\nProcessing {split} split...")
        
        # Extract dataset data
        print(f"Extracting {args.dataset} {split} data...")
        extract_dataset_data(dirs, args.dataset, split, args.subset_size, args.num_jobs)
        
        # Run MFA alignment
        print(f"Running Montreal Forced Aligner for {split}...")

        run_mfa_alignment(dirs, split, args.num_jobs)

    print("\nProcessing complete!")
    print(f"Results are available in:")
    print(f"- Aligned TextGrids: {dirs['aligned']}")
    print(f"- Praat TextGrids: {dirs['textgrids']}")

if __name__ == "__main__":
    main()
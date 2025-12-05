# process_speech.py
import os, sys
import argparse
import subprocess
from tqdm import tqdm

sys.path.append('../utils/')

import utils

def run_mfa_alignment(dirs, split, force_alignment=False, speaker_characters=13, num_jobs=1):
    """
    Run Montreal Forced Aligner on the prepared data
    
    Args:
        dirs (dict): Dictionary of directory paths
        split (str): Dataset split
        force_alignment (bool): Force alignment even if TextGrids exist
        num_jobs (int): Number of jobs for parallel processing
        
    Returns:
        bool: True if alignment was successful, False otherwise
    """    
    corpus_dir = os.path.join(dirs['corpus'], split)
    textgrids_dir = os.path.join(dirs['textgrids'], split)
    
    # Check if alignment is already done
    if not force_alignment and utils.check_textgrids_exist(dirs, split):
        return True
    
    # Check if corpus directory has files
    corpus_files = []
    for root, _, files in os.walk(corpus_dir):
        for file in files:
            if file.endswith('.wav'):
                corpus_files.append(os.path.join(root, file))
                
    if not corpus_files:
        print(f"No wav files found in corpus for {split} split. Skipping alignment.", flush=True)
        return False
    
    try:
        # Create MFA command
        cmd = [
            'mfa',
            'align',
            corpus_dir,
            'english_us_arpa',
            'english_us_arpa',
            textgrids_dir,
            '--clean',
            '--overwrite',
            '--verbose',
            '--output_format', 'long_textgrid',
            '--speaker_characters', str(speaker_characters),
            '--temp_directory', os.path.join(textgrids_dir, 'temp'),
            '--num_jobs', str(num_jobs)
        ]
        
        # Run MFA command
        print(f"Running MFA command: {' '.join(cmd)}", flush=True)
        process = subprocess.run(
            ' '.join(cmd),
            shell=True,
        )
        
        if process.returncode != 0:
            print(f"MFA alignment failed for {split} split", flush=True)
            print("Error output:", process.stderr, flush=True)
            return False
        else:
            print(f"MFA alignment completed successfully for {split} split", flush=True)
            return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error running MFA for {split} split: {str(e)}", flush=True)
        print("Error output:", e.stderr, flush=True)
        return False
    except Exception as e:
        print(f"Unexpected error running MFA for {split} split: {str(e)}", flush=True)
        return False


def main():
    parser = argparse.ArgumentParser(description='Process speech datasets to Praat TextGrids')
    parser.add_argument('--dataset', type=str,
                      required=True, help='Which dataset to process')
    parser.add_argument('--output_dir', type=str, default=None,
                      help='Base directory for output (default: dataset_name_processing)')
    parser.add_argument('--num_jobs', type=int, default=1,
                      help='Number of parallel jobs for MFA alignment')
    parser.add_argument('--split', type=str, default=None,
                      help='Current split for processing')
    parser.add_argument('--overwrite', type=int, default=0,
              help='Force MFA alignment even if files exist')
    
    args = parser.parse_args()

    # Find dataset splits
    # dataset_splits = DATASET_CONFIGS[args.dataset]['splits']
    video = True if args.dataset in ['lrs3', 'avspeech', 'voxceleb2'] else False

    if args.split:
        splits = [args.split]
    else:
        splits = utils.DATASET_CONFIGS[args.dataset]['splits']

    # Determine output directory
    output_dir = args.output_dir or f"{args.dataset}_processing"

    # Prepare directory structure
    dirs, splits = utils.prepare_directory_structure(
        output_dir, 
        splits, 
        dir_names=['aligned', 'corpus', 'textgrids'],
        video=video
    )
    
    # Process each split
    for split in splits:
        print(f"\nProcessing {split} split...", flush=True)
        
        # Run MFA alignment
        print(f"Running Montreal Forced Aligner for {split}...", flush=True)

        run_mfa_alignment(dirs, split, force_alignment=args.overwrite, num_jobs=args.num_jobs)

    print("\nProcessing complete!", flush=True)
    print(f"Results are available in:", flush=True)
    print(f"- Aligned TextGrids: {dirs['aligned']}", flush=True)
    print(f"- Praat TextGrids: {dirs['textgrids']}", flush=True)

if __name__ == "__main__":
    main()
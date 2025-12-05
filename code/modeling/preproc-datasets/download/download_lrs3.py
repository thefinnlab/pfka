#!/usr/bin/env python3
"""
Script to download and reorganize the LRS3 dataset
Usage: python download_lrs3.py [output_directory]
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import glob
import re

def run_command(command, verbose=True):
    """Run a shell command and print output"""
    if verbose:
        print(f"Running: {command}", flush=True)
    process = subprocess.run(command, shell=True, check=True, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if verbose and process.stdout:
        print(process.stdout, flush=True)
    if process.stderr:
        print(f"STDERR: {process.stderr}", file=sys.stderr, flush=True)
    return process.returncode

def check_dependencies():
    """Check if required programs are installed"""
    for cmd in ["wget", "unzip", "cat"]:
        try:
            subprocess.run(["which", cmd], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError:
            print(f"Error: {cmd} is not installed. Please install it first.", flush=True)
            sys.exit(1)

def download_and_extract(name, url, target_dir):
    """Download and extract a zip file"""
    # Skip if directory already exists
    if os.path.exists(target_dir) and os.listdir(target_dir):
        print(f"Directory {target_dir} already exists and is not empty. Skipping download.", flush=True)
        return
    
    print(f"Downloading {name}...", flush=True)
    run_command(f"wget -c '{url}' -O '{name}.zip'")
    
    print(f"Extracting {name}...", flush=True)
    os.makedirs(target_dir, exist_ok=True)
    run_command(f"unzip -q '{name}.zip' -d '{target_dir}'")
    os.remove(f"{name}.zip")

def download_multipart(prefix, parts, target_dir):
    """Download and extract a multipart zip file"""
    # Skip if directory already exists
    if os.path.exists(target_dir) and os.listdir(target_dir):
        print(f"Directory {target_dir} already exists and is not empty. Skipping download.", flush=True)
        return
    
    print(f"Downloading {prefix} (multipart files)...")
    temp_dir = Path("temp_parts")
    temp_dir.mkdir(exist_ok=True)
    
    # Change to temp directory
    original_dir = os.getcwd()
    os.chdir(temp_dir)
    
    # Download all parts
    for part in parts:
        print(f"Downloading part {part}...", flush=True)
        url = f"https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data3/{prefix}_parta{part}"
        run_command(f"wget -c '{url}'")
    
    # Combine the parts
    print("Combining parts into a single file...", flush=True)
    run_command(f"cat {prefix}_part* > {prefix}.zip")
    
    # Extract and remove the zip file
    print(f"Extracting {prefix}...", flush=True)
    target_path = Path(original_dir) / target_dir
    target_path.mkdir(exist_ok=True, parents=True)
    run_command(f"unzip -q '{prefix}.zip' -d '{target_path}'")
    
    # Return to original directory and clean up
    os.chdir(original_dir)
    shutil.rmtree(temp_dir)

def clean_transcript(file_path):
    """Extract only the text line from transcript and convert to lowercase"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Extract text after "Text:" using regex
        match = re.search(r'Text:\s+(.*)', content)
        if match:
            cleaned_text = match.group(1).strip().lower()
            return cleaned_text
        else:
            print(f"Warning: Could not find Text line in {file_path}", file=sys.stderr)
            return content.strip().lower()  # Fallback to entire content
    except Exception as e:
        print(f"Error cleaning transcript {file_path}: {e}", file=sys.stderr)
        return ""

def reorganize(src_dir, video_dest, transcript_dest):
    """Reorganize files from source directory to video and transcript destinations"""
    print(f"Processing directory: {src_dir} to {video_dest} and {transcript_dest}", flush=True)
    
    # Create destination directories if they don't exist
    os.makedirs(video_dest, exist_ok=True)
    os.makedirs(transcript_dest, exist_ok=True)
    
    # Find all directories (speakers) in source
    src_path = Path(src_dir)
    for speaker_dir in [d for d in src_path.iterdir() if d.is_dir()]:
        speaker = speaker_dir.name
        print(f"  Processing speaker: {speaker}", flush=True)
        
        # Process each file
        for file in speaker_dir.iterdir():
            if file.is_file():
                filename = file.name
                ext = file.suffix[1:]  # Remove the dot
                
                # Determine destination based on file extension
                if ext == "mp4":
                    dest = Path(video_dest) / f"{speaker}_{filename}"
                    if not dest.exists():
                        shutil.copy(file, dest)
                elif ext == "txt":
                    dest = Path(transcript_dest) / f"{speaker}_{filename}"
                    if not dest.exists():
                        # Clean transcript and save
                        cleaned_text = clean_transcript(file)
                        with open(dest, 'w') as f:
                            f.write(cleaned_text)

def should_reorganize(src_dir, video_dest, transcript_dest):
    """Check if reorganization is needed (if destinations are empty)"""
    if not os.path.exists(src_dir):
        print(f"Source directory {src_dir} does not exist. Skipping reorganization.", flush=True)
        return False
        
    video_path = Path(video_dest)
    transcript_path = Path(transcript_dest)
    
    if os.path.exists(video_dest) and os.listdir(video_dest) and \
       os.path.exists(transcript_dest) and os.listdir(transcript_dest):
        print(f"Destination directories {video_dest} and {transcript_dest} already have files. Skipping reorganization.", flush=True)
        return False
    
    return True

def main():
    """Main function to download and reorganize the LRS3 dataset"""
    # Set output directory (default: current directory)
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "./"
    os.makedirs(base_dir, exist_ok=True)
    os.chdir(base_dir)
    
    print("=== LRS3 Dataset Downloader and Reorganizer ===", flush=True)
    print(f"Working in: {os.getcwd()}", flush=True)
    
    # Check dependencies
    check_dependencies()
    
    # Create directory structure
    for directory in [
        "src",
        "video/train", "video/val", "video/test",
        "transcripts/train", "transcripts/val", "transcripts/test"
    ]:
        os.makedirs(directory, exist_ok=True)
    
    # Step 1: Download all parts to src/
    print("Step 1: Downloading dataset to src/ directory...", flush=True)
    
    # # Download pretrain (multipart)
    # download_multipart("lrs3_pretrain", "abcdefg", "src/pretrain")
    
    # # Download trainval
    # download_and_extract(
    #     "lrs3_trainval",
    #     "https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data3/lrs3_trainval.zip",
    #     "src/"
    # )
    
    # # Download test
    # download_and_extract(
    #     "lrs3_test",
    #     "https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data3/lrs3_test_v0.4.zip",
    #     "src/"
    # )
    
    # Step 2: Reorganize files
    print("Step 2: Reorganizing files...", flush=True)
    
    # Map source directories to destination directories
    # pretrain → train, trainval → val, test → test
    if should_reorganize("src/pretrain", "video/train", "transcripts/train"):
        reorganize("src/pretrain", "video/train", "transcripts/train")
    
    if should_reorganize("src/trainval", "video/val", "transcripts/val"):
        reorganize("src/trainval", "video/val", "transcripts/val")
    
    if should_reorganize("src/test", "video/test", "transcripts/test"):
        reorganize("src/test", "video/test", "transcripts/test")
    
    # Print summary
    print("=== Processing Complete ===", flush=True)
    print(f"LRS3 dataset has been downloaded and reorganized in: {os.getcwd()}", flush=True)
    print("\nDirectory structure:", flush=True)
    print("- src/: Original downloaded dataset", flush=True)
    print("- video/: Contains train, val, and test directories with renamed video files", flush=True)
    print("- transcripts/: Contains train, val, and test directories with cleaned and renamed transcript files", flush=True)
    print("\nFiles have been renamed to follow the format: SPEAKERNAME_FILENAME\n", flush=True)
    print("Transcripts have been cleaned to only contain the text in lowercase format.\n", flush=True)
    
    # Count files
    def count_files(directory):
        return len(glob.glob(f"{directory}/*"))
    
    print("Summary:", flush=True)
    print(f"Total video in train: {count_files('video/train')}", flush=True)
    print(f"Total video in val: {count_files('video/val')}", flush=True)
    print(f"Total video in test: {count_files('video/test')}", flush=True)
    print(f"Total transcripts in train: {count_files('transcripts/train')}", flush=True)
    print(f"Total transcripts in val: {count_files('transcripts/val')}", flush=True)
    print(f"Total transcripts in test: {count_files('transcripts/test')}", flush=True)

if __name__ == "__main__":
    main()
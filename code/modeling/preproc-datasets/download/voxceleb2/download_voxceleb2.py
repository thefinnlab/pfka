#!/usr/bin/env python3

import os
import sys
import argparse
import random
import urllib.request
import zipfile
import shutil
import subprocess
import time
from pathlib import Path
from tqdm import tqdm
import ssl
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import warnings

# Assuming these imports work in your environment
sys.path.append('../../../../utils/')

from config import *
from dataset_utils import attempt_makedirs

sys.path.append('../../utils/')

import utils

# Disable SSL warnings if needed
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

def parse_args():
    parser = argparse.ArgumentParser(description="Download and organize VoxCeleb2 dataset")
    parser.add_argument("--output-dir", type=str, default="./", 
                        help="Output directory for the dataset")
    parser.add_argument("--key", type=str, 
                        default="21b4db4134efd930a4410dd51cb6f263e87328757b63577f6982f9413eb706af34113c21c49367aa54e9e27e604c9766c8a3ed3c7256b6c019a9d8bcd482787976fba19047f9d1bdc406c76aa378948d17bccd890d6c7952078794739cb5fe84791272419e3972c15a0b44d1c2189ac4e8af42bb45fc662b7275f494073d5d6d",
                        help="Download key for VoxCeleb2")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for train/val split")
    parser.add_argument("--val-ratio", type=float, default=0.1, 
                        help="Ratio of validation set (default: 0.1)")
    parser.add_argument("--chunk-size", type=int, default=8192,
                        help="Chunk size for downloads in bytes (default: 8192)")
    parser.add_argument("--max-retries", type=int, default=5,
                        help="Maximum number of retries for downloads (default: 5)")
    parser.add_argument("--retry-delay", type=int, default=5,
                        help="Delay between retries in seconds (default: 5)")
    parser.add_argument("--timeout", type=int, default=30,
                        help="Connection timeout in seconds (default: 30)")
    parser.add_argument("--verify-ssl", action="store_true",
                        help="Verify SSL certificates (default: False)")
    return parser.parse_args()

class ProgressBar(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_block = 0
        
    def update_to(self, current, total=None):
        if total is not None:
            self.total = total
        block_count = current - self.last_block
        self.update(block_count)
        self.last_block = current

def create_session_with_retries(max_retries=5, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504]):
    """Create a session with retry logic"""
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def download_url_with_requests(url, output_path, chunk_size=8192, max_retries=5, retry_delay=5, timeout=30, verify_ssl=False):
    """Download file from URL with requests and chunking for large files"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    session = create_session_with_retries(max_retries)
    
    for attempt in range(max_retries):
        try:
            # Set verify to False to skip SSL certificate verification
            response = session.get(url, stream=True, timeout=timeout, verify=verify_ssl)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            # Check if file already exists and has the correct size
            if os.path.exists(output_path) and os.path.getsize(output_path) == total_size:
                print(f"File {output_path} already exists with correct size, skipping download.", flush=True)
                return
            
            # Use temporary file for downloading
            temp_path = f"{output_path}.part"
            
            with ProgressBar(unit='B', unit_scale=True, miniters=1, desc=os.path.basename(output_path)) as t:
                with open(temp_path, 'wb') as f:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:  # filter out keep-alive new chunks
                            f.write(chunk)
                            downloaded += len(chunk)
                            t.update_to(downloaded, total_size)
            
            # Rename to final filename only if download completes
            if os.path.exists(temp_path):
                # Verify file size
                if total_size > 0 and os.path.getsize(temp_path) != total_size:
                    raise Exception(f"Downloaded file size ({os.path.getsize(temp_path)}) doesn't match expected size ({total_size})")
                
                shutil.move(temp_path, output_path)
            return
            
        except Exception as e:
            print(f"Download attempt {attempt+1}/{max_retries} failed: {str(e)}", flush=True)
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...", flush=True)
                time.sleep(retry_delay)
            else:
                raise Exception(f"Failed to download after {max_retries} attempts: {str(e)}")

def download_voxceleb_part(key, file, output_path, chunk_size=8192, max_retries=5, retry_delay=5, timeout=30, verify_ssl=False):
    """Download a VoxCeleb2 file part"""
    url = f"https://cn01.mmai.io/download/voxceleb?key={key}&file={file}"
    download_url_with_requests(url, output_path, chunk_size, max_retries, retry_delay, timeout, verify_ssl)

def download_metadata(url, output_path, chunk_size=8192, max_retries=5, retry_delay=5, timeout=30, verify_ssl=False):
    """Download metadata file"""
    download_url_with_requests(url, output_path, chunk_size, max_retries, retry_delay, timeout, verify_ssl)

def extract_zip(zip_path, extract_to, replace_string=None, check_directory=None):
    """Extract a zip file"""
    os.makedirs(extract_to, exist_ok=True)
    
    print(f"Extracting {zip_path} to {extract_to}...", flush=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:

        # Iterate through each file in the ZIP archive
        for file_info in tqdm(zip_ref.infolist(), desc=f"Extracting files..."):

            if replace_string and check_directory:
                check_path = os.path.join(check_directory, file_info.filename.replace(replace_string, ''))

                if os.path.exists(check_path):
                    continue

            # Otherwise extract
            zip_ref.extract(file_info, extract_to)

        # zip_ref.extractall(extract_to)
    print(f"Extraction complete", flush=True)

def cat_files(parts, output_file):
    """Concatenate multiple files into one"""
    print(f"Concatenating parts to {output_file}...", flush=True)
    
    # Check if the output file already exists
    if os.path.exists(output_file):
        # Calculate expected size (sum of all parts)
        expected_size = sum(os.path.getsize(part) for part in parts)
        
        # If file exists with correct size, skip concatenation
        if os.path.getsize(output_file) == expected_size:
            print(f"File {output_file} already exists with correct size, skipping concatenation.", flush=True)
            return
    
    with open(output_file, 'wb') as outfile:
        for part in parts:
            print(f"  Adding part: {os.path.basename(part)}")
            with open(part, 'rb') as infile:
                shutil.copyfileobj(infile, outfile)
    print(f"Concatenation complete", flush=True)

# def find_speaker_ids(directory):
#     """Find all speaker IDs in a directory"""
#     return [d.name for d in Path(directory).iterdir() if d.is_dir()]

# def create_train_val_split(speaker_ids, val_ratio=0.1, seed=42):
#     """Split speaker IDs into train and validation sets"""
#     random.seed(seed)
#     random.shuffle(speaker_ids)
    
#     val_count = int(len(speaker_ids) * val_ratio)
#     train_speakers = speaker_ids[val_count:]
#     val_speakers = speaker_ids[:val_count]
    
#     return train_speakers, val_speakers

def move_files(source_dir, destination_dir):
    # List all files and directories in the source directory
    for item in os.listdir(source_dir):
        source_path = os.path.join(source_dir, item)
        destination_path = os.path.join(destination_dir, item)
        
        # Move the item
        shutil.move(source_path, destination_path)

    # Remove the now-empty source directory
    shutil.rmtree(source_dir)

# def organize_files(src_dir, dest_dir, split, speakers=None, file_ext="mp4"):
#     """
#     Organize files for a specific split (train, val, or test).
    
#     Parameters:
#         src_dir (str): Source directory containing the dataset
#         dest_dir (str): Destination directory for this specific split
#         split (str): The split type ('train', 'val', or 'test')
#         speakers (list): List of speaker IDs to include (for train/val only)
#         file_ext (str): File extension to process (default: 'mp4')
#     """
#     print(f"Organizing {split} {file_ext} files...", flush=True)
#     os.makedirs(dest_dir, exist_ok=True)
    
#     # Determine source subdirectory and filtering approach based on split
#     if split in ['train', 'val']:
#         source_subdir = os.path.join(src_dir, "dev")
#         # For train/val, we only process files from the specified speakers
#         def should_process_speaker(speaker_id):
#             return speaker_id in speakers
#     elif split == 'test':
#         source_subdir = os.path.join(src_dir, "test")
#         # For test, we process all speakers
#         def should_process_speaker(speaker_id):
#             return True
#     else:
#         print(f"Error: Unknown split type '{split}'", flush=True)
#         return
    
#     if not os.path.isdir(source_subdir):
#         print(f"Warning: Directory {source_subdir} not found", flush=True)
#         return
    
#     # Process all eligible speakers
#     file_count = 0
#     for speaker in os.listdir(source_subdir):
#         speaker_dir = os.path.join(source_subdir, speaker)
#         if not os.path.isdir(speaker_dir) or not should_process_speaker(speaker):
#             continue
            
#         print(f"Processing {speaker} for {split} set...", flush=True)
        
#         # Walk through all files for this speaker
#         for root, _, files in os.walk(speaker_dir):
#             for file in files:
#                 if file.endswith(f".{file_ext}"):
#                     src_file = os.path.join(root, file)
                    
#                     # Extract components from the file path
#                     rel_path = os.path.relpath(src_file, speaker_dir)
#                     path_parts = rel_path.split(os.sep)
                    
#                     # First part is video_id, last part is the filename
#                     video_id = path_parts[0]
#                     clip_number = os.path.splitext(path_parts[-1])[0]
                    
#                     # Create new filename with pattern: speaker_id_video_id_clip_number.file_ext
#                     new_filename = f"{speaker}_{video_id}_{clip_number}.{file_ext}"
#                     dest_file = os.path.join(dest_dir, new_filename)
                    
#                     shutil.copy2(src_file, dest_file)
#                     file_count += 1
    
#     print(f"Copied {file_count} {file_ext} files to {split} set", flush=True)

def main():

    parser = argparse.ArgumentParser(description='Preprocess audio/video-text dataset')
    args = parse_args()
    
    # Create directory structure
    data_dir = os.path.join(DATASETS_DIR, 'nlp-datasets', 'voxceleb2')

    splits = ['dev','test']
    # Create source directories
    dirs, splits = utils.prepare_directory_structure(
        data_dir, 
        splits=splits, 
        dir_names=['src/audio', 'src/video']
    )

    # print("=== VoxCeleb2 Dataset Downloader and Organizer ===")
    # print(f"Downloading to: {output_dir}")
    # print(f"Using key: {args.key[:10]}...{args.key[-10:]}")  # Show only part of the key for security
    # print(f"Chunk size: {args.chunk_size} bytes")
    # print(f"Max retries: {args.max_retries}")
    # print(f"Retry delay: {args.retry_delay} seconds")
    # print(f"Connection timeout: {args.timeout} seconds")
    # print(f"Verify SSL: {args.verify_ssl}")
    # print()
    
    # Download and extract audio files (dev)
    print("=== Downloading Audio Files (Dev) ===")
    # audio_parts = "abcdefgh"
    # audio_dev_parts_dir = os.path.join(src_dir, "audio", "parts")
    # os.makedirs(audio_dev_parts_dir, exist_ok=True)
    
    # audio_dev_parts_files = []
    # for part in audio_parts:
    #     file_name = f"vox2_dev_aac_parta{part}"
    #     output_path = os.path.join(audio_dev_parts_dir, file_name)
    #     audio_dev_parts_files.append(output_path)
    #     download_voxceleb_part(
    #         args.key, file_name, output_path, 
    #         chunk_size=args.chunk_size, 
    #         max_retries=args.max_retries, 
    #         retry_delay=args.retry_delay, 
    #         timeout=args.timeout,
    #         verify_ssl=args.verify_ssl
    #     )
    
    # # Concatenate audio parts
    # audio_dev_zip = os.path.join(src_dir, "audio", "vox2_dev_aac.zip")
    # cat_files(audio_dev_parts_files, audio_dev_zip)
    
    # # Extract audio dev zip
    # extract_zip(audio_dev_zip, os.path.join(src_dir, "audio"))

    # move_files(os.path.join(src_dir, "audio/dev/aac"), os.path.join(src_dir, "audio/dev/"))
    
    # # Download and extract audio files (test)
    # print("=== Downloading Audio Files (Test) ===")
    # audio_test_zip = os.path.join(src_dir, "audio", "vox2_test_aac.zip")

    # download_voxceleb_part(
    #     args.key, "vox2_test_aac.zip", audio_test_zip,
    #     chunk_size=args.chunk_size, 
    #     max_retries=args.max_retries, 
    #     retry_delay=args.retry_delay, 
    #     timeout=args.timeout,
    #     verify_ssl=args.verify_ssl
    # )
    # extract_zip(audio_test_zip, os.path.join(src_dir, "audio", "test"))

    # move_files(os.path.join(src_dir, "audio/test/aac"), os.path.join(src_dir, "audio/test/"))
    
    # # Download and extract video files (dev)
    # print("=== Downloading Video Files (Dev) ===")
    # video_parts = "abcdefghi"
    # video_dev_parts_dir = os.path.join(src_dir, "video", "parts")
    # os.makedirs(video_dev_parts_dir, exist_ok=True)
    
    # video_dev_parts_files = []
    # for part in video_parts:
    #     file_name = f"vox2_dev_mp4_parta{part}"
    #     output_path = os.path.join(video_dev_parts_dir, file_name)
    #     video_dev_parts_files.append(output_path)
    #     download_voxceleb_part(
    #         args.key, file_name, output_path,
    #         chunk_size=args.chunk_size, 
    #         max_retries=args.max_retries, 
    #         retry_delay=args.retry_delay, 
    #         timeout=args.timeout,
    #         verify_ssl=args.verify_ssl
    #     )
    
    # # # Concatenate video parts

    # # Download and extract video files (test)
    # print("=== Downloading Video Files (Test) ===")
    # video_test_zip = os.path.join(src_dir, "video", "vox2_test_mp4.zip")
    # download_voxceleb_part(
    #     args.key, "vox2_test_mp4.zip", video_test_zip,
    #     chunk_size=args.chunk_size, 
    #     max_retries=args.max_retries, 
    #     retry_delay=args.retry_delay, 
    #     timeout=args.timeout,
    #     verify_ssl=args.verify_ssl
    # )



    #####################################
    ########### Process splits ##########
    #####################################

    for split in splits:

        print (f"===== Extracting mp4s for {split} =======", flush=True)

        video_zip = os.path.join(dirs['src/video'], f"vox2_{split}_mp4.zip")

        # cat_files(video_dev_parts_files, video_dev_zip)

        # Extract video dev zip --> extracts to the source direcotry
        if split == 'test':
            split_dir = os.path.join(dirs['src/video'], split)
            replace_string = 'mp4/'
        else:
            split_dir = os.path.join(dirs['src/video'], split)
            replace_string = 'mp4/'
        
        extract_zip(video_zip, split_dir, replace_string=replace_string, check_directory=split_dir)

        move_files(
            os.path.join(split_dir, f"mp4"), 
            split_dir
        )

    
    # # Create train/val split
    # print("=== Creating Train/Val Split ===", flush=True)
    # speaker_ids = find_speaker_ids(os.path.join(src_dir, "audio", "dev"))
    # train_speakers, val_speakers = create_train_val_split(
    #     speaker_ids, val_ratio=args.val_ratio, seed=args.seed
    # )

    # speakers = {
    #     'train': train_speakers,
    #     'val': val_speakers,
    # }
    
    # print(f"Found {len(speaker_ids)} speakers", flush=True)
    # print(f"Assigned {len(train_speakers)} speakers to train set", flush=True)
    # print(f"Assigned {len(val_speakers)} speakers to validation set", flush=True)
    
    # # Save speaker lists for reference
    # with open(os.path.join(output_dir, "train_speakers.txt"), "w") as f:
    #     f.write("\n".join(train_speakers))
    
    # with open(os.path.join(output_dir, "val_speakers.txt"), "w") as f:
    #     f.write("\n".join(val_speakers))
    
    # # # Organize files
    # # organize_files(
    # #     os.path.join(src_dir, "audio"),
    # #     os.path.join(audio_dir, "train"),
    # #     os.path.join(audio_dir, "val"),
    # #     os.path.join(audio_dir, "test"),
    # #     train_speakers,
    # #     val_speakers,
    # #     "m4a"
    # # )
    
    # # Organize files for each split
    # for split in ['train', 'val', 'test']:
    #     organize_files(
    #         src_dir=os.path.join(src_dir, "video"),
    #         dest_dir=os.path.join(video_dir, split),
    #         split=split,
    #         speakers=speakers[split] if split != 'test' else None,
    #         file_ext="mp4"
    #     )
    
    # # Print summary
    # print("\n=== Download and Organization Complete ===", flush=True)
    # print(f"VoxCeleb2 dataset has been downloaded and organized in: {output_dir}", flush=True)
    # print("\nDirectory structure:", flush=True)
    # print("- src/: Original downloaded and extracted files", flush=True)
    # # print("- audio/: Organized audio files with train/val/test splits", flush=True)
    # print("- video/: Organized video files with train/val/test splits", flush=True)
    
    # # # Count files
    # # train_audio_count = sum([len(files) for _, _, files in os.walk(os.path.join(audio_dir, "train"))])
    # # val_audio_count = sum([len(files) for _, _, files in os.walk(os.path.join(audio_dir, "val"))])
    # # test_audio_count = sum([len(files) for _, _, files in os.walk(os.path.join(audio_dir, "test"))])
    
    # train_video_count = sum([len(files) for _, _, files in os.walk(os.path.join(video_dir, "train"))])
    # val_video_count = sum([len(files) for _, _, files in os.walk(os.path.join(video_dir, "val"))])
    # test_video_count = sum([len(files) for _, _, files in os.walk(os.path.join(video_dir, "test"))])
    
    # print("\nFile counts:", flush=True)
    # # print(f"Audio - Train: {train_audio_count}, Val: {val_audio_count}, Test: {test_audio_count}", flush=True)
    # print(f"Video - Train: {train_video_count}, Val: {val_video_count}, Test: {test_video_count}", flush=True)
    
    # # Get total size
    # def get_dir_size(path):
    #     total = 0
    #     with os.scandir(path) as it:
    #         for entry in it:
    #             if entry.is_file():
    #                 total += entry.stat().st_size
    #             elif entry.is_dir():
    #                 total += get_dir_size(entry.path)
    #     return total
    
    # total_size_gb = get_dir_size(output_dir) / (1024 * 1024 * 1024)
    # print(f"\nTotal dataset size: {total_size_gb:.2f} GB", flush=True)

if __name__ == "__main__":
    main()
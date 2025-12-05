#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import argparse
import math
from tqdm import tqdm

import torch
from torchvision.io import read_video, write_video

sys.path.append('../utils/')

import utils
from video_process import VideoProcessor
from detector import LandmarksDetector

def process_video_file(landmarks_detector, processor, video_path, output_dir):
    """
    Process a single video file
    
    Args:
        landmarks_detector (LandmarksDetector): Detector for facial landmarks
        processor (VideoProcessor): Video processing object
        video_path (str): Path to video file
        output_dir (str): Output directory for processed video
        target_size (tuple): Target size for video frames
        convert_gray (bool): Convert video to grayscale
        crop_mouth_roi (bool): Crop mouth region of interest
        
    Returns:
        tuple: (success, video_path, message)
    """
    try:
        # Create the directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine output path
        base_name = os.path.basename(video_path)
        output_path = os.path.join(output_dir, base_name)
        
        # Skip if processed file already exists
        if os.path.exists(output_path):
            return True, video_path, "Already processed"

        # Load video 
        video, audio, info = read_video(video_path, pts_unit="sec")
        video_numpy = video.numpy()

        # Grab the face landmarks
        landmarks = landmarks_detector(video_numpy)

        # Process video
        processed_video = processor(video_numpy, landmarks)
        
        if processed_video is None:
            return False, video_path, "Failed to process video"
        
        # Convert processed video to tensor for saving
        processed_tensor = torch.from_numpy(processed_video)

        # Write processed video as MP4 with original audio
        write_video(output_path, processed_tensor, fps=info['video_fps'], 
            audio_array=audio, audio_fps=info['audio_fps'], audio_codec='aac')

        return True, video_path, "Success"
        
    except Exception as e:
        return False, video_path, str(e)

def preprocess_videos(dirs, split, target_size=(96, 96), convert_gray=False, crop_mouth_roi=False, num_shards=1, current_shard=0, force=False):
    """
    Preprocess video files in parallel
    
    Args:
        dirs (dict): Dictionary of directory paths
        split (str): Dataset split
        target_size (tuple): Target frame size
        convert_gray (bool): Convert to grayscale
        crop_mouth_roi (bool): Crop mouth region of interest
        num_shards (int): Number of data shards
        current_shard (int): Current shard to process
        force (bool): Force reprocessing
        
    Returns:
        str: Path to output directory
    """

    video_dir = os.path.join(dirs['video'], split)
    output_dir = os.path.join(dirs['video-processed'], split)
    
    # Get all video files
    video_files = sorted(glob.glob(os.path.join(video_dir, "*.mp4"), recursive=True))

    # Apply sharding logic
    if num_shards > 1:
        # Calculate shard size and starting/ending indices
        shard_size = math.ceil(len(video_files) / num_shards)
        start_idx = current_shard * shard_size
        end_idx = min(start_idx + shard_size, len(video_files))
        
        # Get only the files for the current shard
        video_files = video_files[start_idx:end_idx]
        
        print(f"Processing shard {current_shard+1}/{num_shards} with {len(video_files)} files", flush=True)
    
    # Count existing processed files
    existing_count = 0
    to_process_files = []
    
    for video_path in video_files:
        rel_path = os.path.relpath(video_path, video_dir)
        processed_path = os.path.join(output_dir, os.path.splitext(rel_path)[0] + '.pt')
        if os.path.exists(processed_path) and not force:
            existing_count += 1
        else:
            to_process_files.append(video_path)
    
    to_process_count = len(to_process_files)
    
    if to_process_count == 0:
        print(f"All videos already processed for {split} split. Skipping preprocessing.", flush=True)
        return output_dir
    
    print(f"Found {existing_count} existing processed files and {to_process_count} files to process", flush=True)

    # Process files in parallel
    completed = 0
    errors = 0

    # Grab the current device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Initialize preprocessor
    processor = VideoProcessor(
        crop_width=target_size[0],
        crop_height=target_size[1],
        convert_gray=convert_gray,
        do_crop=crop_mouth_roi,
    )

    landmarks_detector = LandmarksDetector(device=device)

    # Process each file
    for file_name in tqdm(to_process_files):

        # Attempt to process the file
        success, output_path, message = process_video_file(landmarks_detector, processor, video_path, output_dir)

        if success:
            completed += 1
        else:
            errors += 1
            print(f"Error processing {video_path}: {message}", flush=True)
    
    print(f"Video preprocessing complete: {completed} successful, {errors} failed", flush=True)
    return output_dir


def main():
    parser = argparse.ArgumentParser(description='Preprocess video datasets')
    parser.add_argument('-d','--dataset', type=str, choices=['lrs3', 'avspeech', 'voxceleb2'],
                      required=True, help='Which dataset to process')
    parser.add_argument('--output_dir', type=str, default=None,
                      help='Base directory for output (default: dataset_name_processing)')
    parser.add_argument('--split', type=str, default=None,
                  help='Which split to process')
    parser.add_argument('--target_size', type=int, nargs=2, default=[96, 96],
                      help='Target size for video frames (default: 96x96)')
    parser.add_argument('--convert_gray', type=int, default=0,
                      help='Convert video to grayscale (default: 0)')
    parser.add_argument('--crop_mouth_roi', type=int, default=0,
                      help='Crop mouth region of interest (default: 0)')
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
        dir_names=['video', 'video-processed'],
    )

    # Process each split
    for split in splits:
        print(f"\nProcessing {args.dataset} {split} split...", flush=True)
        print(f"\nCurrent shard: {args.current_shard+1}/{args.num_shards}", flush=True)

        # Extract audio from videos
        print(f"Extracting {args.dataset} {split} data...", flush=True)
        preprocess_videos(dirs, split, target_size=args.target_size, convert_gray=args.convert_gray, crop_mouth_roi=args.crop_mouth_roi, 
            force=args.overwrite, num_shards=args.num_shards, current_shard=args.current_shard)
    
    print("\nProcessing complete!", flush=True)

if __name__ == "__main__":
    main()
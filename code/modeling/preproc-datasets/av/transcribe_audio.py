# process_speech.py
import os, sys
import glob
import argparse
import itertools
import re
import torch
import math
from torchvision.io import read_video
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

from tqdm import tqdm
import shutil

sys.path.append('../utils/')

import utils

# Constants
chars_to_ignore_regex = r"[\,\?\.\!\-\;\:\"]"

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(itertools.islice(it, size)), ())

def save_transcript(fn, result, audio_dir, transcript_dir):
    """Helper function to save a transcript"""
    rel_path = os.path.relpath(fn, audio_dir)
    transcript_path = os.path.join(transcript_dir, os.path.splitext(rel_path)[0] + '.txt')
    
    transcript = (
        re.sub(chars_to_ignore_regex, "", result["text"])
        .lower()
        .replace("'", "'")
    )
    transcript = " ".join(transcript.split())
    
    if transcript:
        utils.save_transcript(transcript, transcript_path)
    else:
        print (f'No successful transcription: {fn}')

def transcribe_batch(audio_dir, transcript_dir, batch, pipe):

    with torch.no_grad():
        results = pipe(batch, generate_kwargs={"language": "english", "return_timestamps": False})
    
    # Save all successful transcripts
    for fn, result in zip(batch, results):
        save_transcript(fn, result, audio_dir, transcript_dir)

def repair_audio_file(fn):

    video_fn = fn.replace('audio', 'video').replace('.wav', '.mp4')
    video_data, audio_data, info = read_video(video_fn, pts_unit="sec")
    audio_array = audio_data.numpy()[0]
    audio_array, _ = utils.resample_audio(audio_array, orig_sr=info['audio_fps'], target_sr=16000)
    utils.save_audio_file(audio_array, fn, 16000)

def transcribe_audio(dirs, split, model_name="openai/whisper-large-v3-turbo", batch_size=16, force=False, num_shards=1, current_shard=0):
    """
    Transcribe audio files using Whisper
    
    Args:
        dirs (dict): Dictionary of directory paths
        split (str): Dataset split
        gpu_type (str): GPU type for ASR
        force (bool): Force transcription even if transcript files exist
        
    Returns:
        str: Path to transcript directory
    """

    # Get all audio files
    all_fns = sorted(glob.glob(os.path.join(dirs['audio'], "*.wav"), recursive=True))

    # Apply sharding logic
    if num_shards > 1:
        # Calculate shard size and starting/ending indices
        shard_size = math.ceil(len(all_fns) / num_shards)
        start_idx = current_shard * shard_size
        end_idx = min(start_idx + shard_size, len(all_fns))
        
        # Get only the files for the current shard
        all_fns = all_fns[start_idx:end_idx]
        
        print(f"Processing shard {current_shard+1}/{num_shards} with {len(all_fns)} files", flush=True)
    
    # Count existing transcripts
    existing_count = 0
    to_process_count = 0

    audio_fns = []
    
    for fn in all_fns:
        rel_path = os.path.relpath(fn, dirs['audio'])
        transcript_path = os.path.join(dirs['transcripts'], os.path.splitext(rel_path)[0] + '.txt')
        if os.path.exists(transcript_path) and not force:
            existing_count += 1
        else:
            audio_fns.append(fn)
            to_process_count += 1
    
    if to_process_count == 0:
        print(f"All transcripts already exist for {split} split. Skipping transcription.", flush=True)
        return dirs['transcripts']
    
    print(f"Found {existing_count} existing transcripts and {to_process_count} files to process", flush=True)
    
    # Load the ASR model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Load processor
    processor = AutoProcessor.from_pretrained(model_name)

    # Load model
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name, torch_dtype=torch_dtype, low_cpu_mem_usage=True, attn_implementation="sdpa"
    ).to(device)

    # Create a pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        batch_size=batch_size,  # batch size for inference - set based on your device
        torch_dtype=torch_dtype,
        device=device,
    )

    all_batches = list(chunk(audio_fns, size=batch_size))
    
    for batch in tqdm(all_batches, desc=f"Transcribing {split} audio"):
        
        batch = list(batch)
        
        # Try processing the entire batch
        try:
            transcribe_batch(dirs['audio'], dirs['transcripts'], batch, pipe)

        except Exception as batch_error:
            print(f"Batch processing failed. Processing files individually.", flush=True)
            # Process each file individually
            for fn in batch:
                try:
                    # Try processing the single file
                    transcribe_batch(dirs['audio'], dirs['transcripts'], [fn], pipe)
                    
                except Exception as e:
                    # Check if this is a corrupt audio file
                    if "Soundfile is either not in the correct format or is malformed" in str(e):
                        print(f"Repairing corrupt audio: {fn}", flush=True)
                        
                        # Attempt to fix the audio file
                        try:
                            # Fix the file
                            repair_audio_file(fn)

                            # Try again with the fixed file
                            transcribe_batch(dirs['audio'], dirs['transcripts'], [fn], pipe)

                            print(f"Successfully repaired and transcribed: {fn}", flush=True)
                        except Exception as repair_error:
                            print(f"Failed repair and retranscription: {fn}", flush=True)
                    else:
                        print(f"Failed to transcribe: {fn}", flush=True)

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
    parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size for processing transcripts')
    parser.add_argument('--num_shards', type=int, default=1,
                    help='Number of shards to divide the dataset into')
    parser.add_argument('--current_shard', type=int, default=0,
                    help='Current shard to process (0-based indexing)')
    parser.add_argument('--overwrite', type=int, default=0,
                    help='Force transcription even if files exist')
    
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
        dir_names=['audio', 'transcripts'],
    )

    # Process each split
    for split in splits:
        print(f"\nProcessing {args.dataset} {split} split...", flush=True)
        print(f"\nCurrent shard: {args.current_shard+1}/{args.num_shards}", flush=True)

        split_dirs = {k: os.path.join(v, split) for k, v in dirs.items()}

        print(f"Transcribing audio...", flush=True)

        if args.dataset == 'avspeech':

            if args.lang_id:
                lang_dirs = {k: os.path.join(v, args.lang_id) for k, v in split_dirs.items()}
            else:
                continue
            
            transcribe_audio(lang_dirs, split, force=args.overwrite, batch_size=args.batch_size,
                num_shards=args.num_shards, current_shard=args.current_shard)
        else:
            # Transcribe audio files
            transcribe_audio(split_dirs, split, force=args.overwrite, batch_size=args.batch_size,
                num_shards=args.num_shards, current_shard=args.current_shard)

    print("\nProcessing complete!", flush=True)

if __name__ == "__main__":
    main()
# process_speech.py
import os, sys

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import glob
import argparse
import math
import json
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.io import read_video
import shutil
from transformers import AutoImageProcessor, AutoProcessor, AutoModel, AutoTokenizer

sys.path.append('../utils/')

import utils

LANGUAGE_MODELS = {
    'gpt2': "openai-community/gpt2"
}

# Model repositories
AUDIO_MODELS = {
    'wav2vec2': "facebook/wav2vec2-large-960h-lv60",
    'hubert': "facebook/hubert-large-ls960",
}

VIDEO_MODELS = {
    # 'video_swin': "microsoft/videoswin-base-patch244-window877-kinetics400-1k",
    # 'video_clip': "openai/clip-vit-base-patch32",
    # 'videomae': None,
    'resnet18': "microsoft/resnet-18",
    'data2vec': "facebook/data2vec-vision-large",
}

def initialize_models(args):
    print('Initializing models...', flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # At minimum we have a text tokenizer
    models = {
        'tokenizer': AutoTokenizer.from_pretrained(LANGUAGE_MODELS[args.text_model], use_fast=True, add_prefix_space=True),
        'device': device
    }

    if args.audio_model:        
        models['audio'] = {
            'model_name': args.audio_model,
            'processor': AutoProcessor.from_pretrained(AUDIO_MODELS[args.audio_model]),
            'model': AutoModel.from_pretrained(AUDIO_MODELS[args.audio_model]).to(device)
        }

    if args.video_model:
        models['video'] = {
            'model_name': args.video_model,
            'processor': AutoImageProcessor.from_pretrained(VIDEO_MODELS[args.video_model]),
            'model': AutoModel.from_pretrained(VIDEO_MODELS[args.video_model]).to(device)
        }

    return models

def preprocess_data(args, dirs, split, models):
    """Preprocess all data with temporary file handling."""

    temp_dir, errors_dir = [dirs.get(item) for item in ['temp_dir', 'errors_dir']]

    # Get file list 
    all_fns = sorted(os.listdir(dirs["textgrids"]))
    all_fns = [os.path.splitext(fn)[0] for fn in all_fns]

    # Apply sharding logic --> divide dataset into number of shards 
    all_fns = utils.get_shard_data(all_fns, num_shards=args.num_shards, current_shard=args.current_shard)

    # Count existing json files
    existing_count = 0
    to_process_count = 0

    process_fns = []

    for fn in all_fns:
        temp_json_fn = utils.get_temp_json_path(temp_dir, fn)
        error_json_fn = utils.get_temp_json_path(errors_dir, fn)

        # If the file was successfully or unsuccessfully processed, a json exists
        if (os.path.exists(temp_json_fn) or os.path.exists(error_json_fn)) and not args.overwrite:
            existing_count += 1
        else:
            process_fns.append(fn)
            to_process_count += 1

    if to_process_count == 0:
        print(f"All json files exist for {split} split. Skipping transcription.", flush=True)
        return temp_dir

    print(f"Found {existing_count} existing json files and {to_process_count} files to process", flush=True)

    failed_samples_count = 0

    # Process each file
    for file_name in tqdm(process_fns):

        # Attempt to process the file
        temp_json_path = process_single_file(args, dirs, models, file_name)
        
        # If it fails, save a file to the errors directory to log which files didn't process correctly
        if not temp_json_path:
            error_json = {'base_name': file_name}
            errors_path = utils.get_temp_json_path(errors_dir, file_name)
            utils.save_json(errors_path, {'base_name': file_name})
            failed_samples_count += 1

    print(f"Split {split} failed samples: {failed_samples_count}", flush=True)

def load_file_data(args, dirs, models, file_name):
    """Load all necessary data for a single file."""

    textgrid_path = os.path.join(dirs['textgrids'], f"{file_name}.TextGrid")
    audio_path = os.path.join(dirs['audio'], f"{file_name}.wav")
    prosody_path = os.path.join(dirs['prosody'], f"{file_name}.prom")

    file_data = {}
    
    try:
        # Load textgrid words and prosody data
        words = utils.parse_textgrid(textgrid_path)
        prosody_data = utils.process_wavelet_file(prosody_path)
        
        # Verify data matches
        if not words or len(prosody_data) != len(words):
            print ('Mismatched prosody and words')
            return None

        # If there aren't enough or too many words we skip
        if (len(words) < args.min_words) or (len(words) > args.max_words):
            print (f"Number of words {len(words)}, Min words {args.min_words}, Max words {args.max_words}")
            return None
            
        # Extract prosody features
        prominence = np.array([word['prominence'] for word in prosody_data])
        boundary = np.array([word['boundary'] for word in prosody_data])

        file_data = {
            'base_name': file_name,
            'words': words,
            'prominence': prominence,
            'boundary': boundary,
        }

        # Load audio if model specified
        if 'audio' in models:
            waveform, sample_rate = utils.load_audio(audio_path)

            file_data.update({
                'waveform': waveform,
                'audio_sr': sample_rate,
            })

        # Find video file if requested
        if 'video' in models:
            # Load video
            video_path = os.path.join(dirs['video'], f"{file_name}.mp4")
            video_data, _, video_info = read_video(video_path, pts_unit="sec")

            file_data.update({
                'video_data': video_data,
                'video_fps': video_info['video_fps'],
            })

        return file_data
        
    except Exception as e:
        print(f"Error processing {file_name}: {str(e)}", flush=True)
        return None

def process_single_file(args, dirs, models, file_name):
    """Process a single file with temporary JSON caching."""

    # Get the temporary json path
    temp_json_path = utils.get_temp_json_path(dirs['temp_dir'], file_name)
    
    # Return existing temp file if present and not forcing reprocess
    if os.path.exists(temp_json_path) and not args.overwrite:
        return temp_json_path

    # Load file data
    file_data = load_file_data(args, dirs, models, file_name)

    if file_data is None:
        return None

    ##########################################
    ########## Create path names #############
    ##########################################

    # Text information
    text_tokens_path = os.path.join(dirs["text_model"], f"{file_name}_text-tokens.pt")
    attention_mask_path = os.path.join(dirs["text_model"], f"{file_name}_attention-mask.pt")

    # Prosody information
    prominence_path = os.path.join(dirs["prosody_model"], f"{file_name}_prominence.pt")
    boundary_path = os.path.join(dirs["prosody_model"], f"{file_name}_boundary.pt")

    # Audio/video features
    if args.audio_model:
        audio_features_path = os.path.join(dirs["audio_model"], f"{file_name}_audio-features.pt")

    if args.video_model:
        video_features_path = os.path.join(dirs["video_model"], f"{file_name}_video-features.pt")

    ##########################################
    ############## Process text ##############
    ##########################################

    # Process text
    text = " ".join([word['text'] for word in file_data['words']])
    text_tokens = models['tokenizer'](text)

    # Get unique ids and number of tokens
    word_ids, token_counts = np.unique(text_tokens.word_ids(), return_counts=True)
    
    # Verify token counts match words
    if len(word_ids) != len(file_data['words']):
        print (f"Mismatched number of tokens/words: {file_name}", flush=True)
        return None

    # Set up the results --> this is bare minimum is having text + prosody
    result = {
        'base_name': file_data['base_name'],
        'text': text,
        'text_tokens_path': text_tokens_path,
        'attention_mask_path': attention_mask_path,
        'prominence_path': prominence_path,
        'boundary_path': boundary_path
    }

    if 'waveform' in file_data:
        duration = file_data['waveform'] / file_data['audio_sr']
        result['duration'] = duration

    # Save token data
    if args.overwrite or not all(os.path.exists(p) for p in [text_tokens_path, attention_mask_path]):
        torch.save(text_tokens['input_ids'], text_tokens_path)
        torch.save(text_tokens['attention_mask'], attention_mask_path)

    # Process prosody data
    if args.overwrite or not all(os.path.exists(p) for p in [prominence_path, boundary_path]):
        prominence_boundary = np.stack((file_data['prominence'], file_data['boundary'])).T
        prominence, boundary = utils.interpolate_prosody(prominence_boundary, token_counts).T
        torch.save(prominence, prominence_path)
        torch.save(boundary, boundary_path)

    # Process audio if requested
    if 'audio' in models:
        # If we've either specified overwriting or the path doesn't exist
        if (args.overwrite or not os.path.exists(audio_features_path)):
            try:
                audio_features = process_audio(file_data, models, text_tokens)
            except Exception as e:
                print (f'Problem extracting audio features: {str(e)}')
                return None

            # we need to save the features
            if audio_features is not None:
                torch.save(audio_features, audio_features_path)
            else:
                print (f'No audio features: {str(e)}')
                return None

        # Otherwise the path exists --> put into the file
        result['audio_features_path'] = audio_features_path
    
    # Process video if requested
    if 'video' in models:
        # If we've either specified overwriting or the path doesn't exist
        if (args.overwrite or not os.path.exists(video_features_path)):
            try:
                video_features = process_video(file_data, models, text_tokens)
            except Exception as e:
                print (f'Problem extracting video features: {str(e)}')
                return None
            
            # we need to save the features
            if video_features is not None:
                torch.save(video_features, video_features_path)
            else:
                print (f'No video features: {str(e)}')
                return None

        # Otherwise the path exists --> put into the file      
        result['video_features_path'] = video_features_path
    
    utils.save_json(temp_json_path, result)
    return temp_json_path

@torch.no_grad() 
def process_audio(file_data, models, text_tokens, end_tolerance=1):
    """Process audio data and extract embeddings."""

    # Grab information from the file data
    words, waveform, audio_sr = [file_data.get(item) for item in ['words', 'waveform', 'audio_sr']]
    processor, model = [models['audio'].get(item) for item in ['processor', 'model']]

    # Get word token ids
    word_ids, token_counts = np.unique(text_tokens.word_ids(), return_counts=True)
    segments = []

    # Extract audio segments for each word/token
    for i, (word, idx, n_tokens) in enumerate(zip(words, word_ids, token_counts)):
        if n_tokens > 1:
            # If current word has multiple tokens, create ratios based on length of tokens
            ratios = [len(x) for x in models['tokenizer'].batch_decode(text_tokens["input_ids"][idx:idx+n_tokens])]
            ratios = torch.tensor(ratios)
            ratios = ratios / ratios.sum()
        else:
            ratios = None

        # Extract word segments with weighted ratios
        word_segments = utils.extract_media_segment(
            waveform, 
            rate = audio_sr, 
            onset = word["start"], 
            offset = word["end"], 
            ratios = ratios, 
            time_axis = 1, 
            end_tolerance = end_tolerance if (i + 1) == len(words) else None
        )

        segments.extend(word_segments)
    
    # Process all segments at once
    features = processor(segments, sampling_rate=audio_sr, padding=True, 
        return_attention_mask=True, return_tensors="pt")

    features = {k: v.to(models["device"]) for k, v in features.items()}

    # Get embeddings
    audio_features = model(**features).last_hidden_state
    attention_mask = model._get_feature_vector_attention_mask(
        audio_features.shape[1], features['attention_mask']
    )
    audio_features = utils.pool_embeddings(audio_features, attention_mask)
    
    return audio_features.cpu()

@torch.no_grad()
def process_video(file_data, models, text_tokens, end_tolerance=1):
    """Process video data and extract embeddings at word timepoints."""
    
    words, video_data, fps = [file_data.get(item) for item in ['words', 'video_data', 'video_fps']]
    model_name, processor, model = [models['video'].get(item) for item in ['model_name', 'processor', 'model']]

    # Get word token ids
    word_ids, token_counts = np.unique(text_tokens.word_ids(), return_counts=True)

    # Because we're considering videos as images, we can do this thing here
    inputs = processor(video_data, return_tensors="pt")

    # Move inputs to the same device as the model
    inputs = {k: v.to(models['device']) for k, v in inputs.items()}

    with torch.no_grad():
        features = model(**inputs).last_hidden_state

    segments = []

    # Process each word
    for i, (word, idx, n_tokens) in enumerate(zip(words, word_ids, token_counts)):
        if n_tokens > 1:
            # If current word has multiple tokens, create ratios based on length of tokens
            ratios = [len(x) for x in models['tokenizer'].batch_decode(text_tokens["input_ids"][idx:idx+n_tokens])]
            ratios = torch.tensor(ratios)
            ratios = ratios / ratios.sum()
        else:
            ratios = None

        # Extract word segments with weighted ratios
        word_segments = utils.extract_media_segment(
            features, 
            rate = fps, 
            onset = word["start"], 
            offset = word["end"], 
            ratios = ratios, 
            end_tolerance = end_tolerance if (i + 1) == len(words) else None
        )

        segments.extend(word_segments)

    # Lastly perform mean pooling over patches (spatial) and average over frames (time)
    video_features = []

    for segment in segments:
        if model_name in ['resnet18']:
            # Pool the last two dimensions (H x W)
            # CNN = B x D x H x W --> adaptive pool makes B x D x 1 x 1
            segment_features = F.adaptive_avg_pool2d(segment, output_size=1).mean(0).squeeze()
        else:
            # If transformer, we have B x Patch x D
            segment_features = utils.pool_embeddings(segment).mean(0)

        video_features.append(segment_features)

    video_features = torch.stack(video_features)

    return video_features.cpu()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Preprocess audio/video-text dataset')
    parser.add_argument('-d','--dataset', type=str,required=True, 
                    help='Which dataset to process')
    parser.add_argument('--output_dir', type=str, required=True,
                    help='Base directory for output (default: dataset_name_processing)')
    parser.add_argument('--split', type=str, default=None,
                    help='Which split to process')
    parser.add_argument('--lang_id', type=str, default='eng',
                    help='Language ID ISO-639 code for AVSpeech')

    ### Model names
    parser.add_argument('--text_model', type=str, default='gpt2', help='Text model to use')
    parser.add_argument('--audio_model', type=str, default='wav2vec2', choices=list(AUDIO_MODELS.keys()) + [None], 
                    help='Audio model to use, or "None" to skip audio processing')
    parser.add_argument('--video_model', type=str, default=None, choices=list(VIDEO_MODELS.keys()) + [None], 
                    help='Video model to use, or None to skip video processing')

    ### Dataset filtering setup
    parser.add_argument('--min_words', type=int, default=4, help='Minimum number of words per sample')
    parser.add_argument('--max_words', type=int, default=128, help='Maximum number of words per sample')

    ### Sharding for more efficient processing
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

    if args.dataset in ['lrs3', 'avspeech', 'voxceleb2']:
        dir_names = ['audio', 'textgrids', 'video', 'prosody']
    else:
        dir_names = ['audio', 'textgrids', 'prosody']

    # Prepare directory structure --> only this script is needed for video
    dirs, splits = utils.prepare_directory_structure(
        output_dir, 
        splits, 
        dir_names=dir_names,
    )

    model_cache_dirs = {
        'prosody_model': 'prosody',
        'text_model': args.text_model,
    }

    # Set name of the model combo for the metadata
    model_combo = args.text_model

    if args.audio_model:
        model_cache_dirs['audio_model'] = args.audio_model
        model_combo += f'-{args.audio_model}'

    if args.video_model:
        model_cache_dirs['video_model'] = args.video_model
        model_combo += f'-{args.video_model}'


    # Create cache for our features and a temp directory for writing progress
    dirs["cache_dir"] = os.path.join(args.output_dir, 'features')

    # Where to save metadata info 
    dirs["metadata_dir"] = os.path.join(dirs["cache_dir"], 'metadata', model_combo)

    for model_type, dir_name in model_cache_dirs.items():
        model_cache_dir = os.path.join(dirs["cache_dir"], dir_name)
        os.makedirs(model_cache_dir, exist_ok=True)

        # Add to the directories
        dirs[model_type] = model_cache_dir

    # Initialize models used in the preprocessing 
    models = initialize_models(args)

    for split in splits:
        print(f"\nExtracting features for {split} split...", flush=True)
        print(f"\nCurrent shard: {args.current_shard+1}/{args.num_shards}", flush=True)
        split_dirs = {k: os.path.join(v, split) for k, v in dirs.items()}

        if args.dataset == 'avspeech':
            if args.lang_id:
                split_dirs = {k: os.path.join(v, args.lang_id) for k, v in split_dirs.items()}
            else:
                continue
        
        # Make the json directories --> move this earlier TLB 3/14/25
        split_dirs["temp_dir"] = os.path.join(split_dirs["metadata_dir"], 'temp')
        split_dirs["errors_dir"] = os.path.join(split_dirs["metadata_dir"], 'errors')

        # Make directories if they don't exist
        for _, dir_name in split_dirs.items():
            os.makedirs(dir_name, exist_ok=True)
        # os.makedirs(split_dirs["errors_dir"], exist_ok=True)

        preprocess_data(args, split_dirs, split, models)

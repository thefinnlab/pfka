# utils.py
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import math
import json
import subprocess
import librosa
import soundfile as sf
from tqdm import tqdm
import shutil

from typing import Union, Literal, List, Dict, Optional, Any
import string
import numpy as np

import torch
import torchaudio
from torch.nn import functional as F
from praatio import textgrid

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
    },

    ## Audiovisual datasets
    'lrs3': {
        'splits': ['train', 'validation', 'test'],

    }
}

# Add audiovisual datasets
for dataset in ['lrs3', 'avspeech', 'voxceleb2']:
    DATASET_CONFIGS[dataset] = {'splits': ['train', 'val', 'test']}

DATASET_TYPES = {
    'audio': {
        'text_model': 'gpt2',
        'audio_model': 'wav2vec2',
    },
    'video': {
        'text_model': 'gpt2',
        'audio_model': 'wav2vec2',
        'video_model': 'data2vec',
    }
}

def get_shard_data(data, num_shards=1, current_shard=0):
    # Apply sharding logic --> divide dataset into number of shards 
    if num_shards > 1:
        # Calculate shard size and starting/ending indices
        shard_size = math.ceil(len(data) / num_shards)
        start_idx = current_shard * shard_size
        end_idx = min(start_idx + shard_size, len(data))
        
        # Get only the files for the current shard
        data = data[start_idx:end_idx]

    print(f"Processing shard {current_shard+1}/{num_shards} with {len(data)} files", flush=True)
    return data

def prepare_directory_structure(base_dir, splits=None, dir_names=None, video=False):
    """
    Create necessary directories for processing if they don't exist
    
    Args:
        base_dir (str): Base directory for all data
        splits (list): List of dataset splits (e.g., 'train', 'val', 'test')
        
    Returns:
        dict: Dictionary of directory paths
        list: List of split names
    """
    # Normalize split names (replace dots with hyphens)
    if splits:
        normalized_splits = [split.replace('.', '-') for split in splits]
    else:
        normalized_splits = []

    if dir_names is None:
        dir_names = ['src', 'audio', 'transcripts', 'corpus', 'textgrids', 'aligned', 'prosody']

    if video and 'video' not in dir_names:
        dir_names.append('video')

    # Define directory structure
    dirs = {d: os.path.join(base_dir, d) for d in dir_names}

    # Create main directories
    for dir_path in dirs.values():
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            print(f"Created directory: {dir_path}")
    
    # Create split-specific subdirectories
    for split in normalized_splits:
        for key in dir_names:
            split_dir = os.path.join(dirs[key], split)
            if not os.path.exists(split_dir):
                os.makedirs(split_dir, exist_ok=True)
                print(f"Created directory: {split_dir}")
    
    return dirs, normalized_splits

def resample_audio(audio_data, orig_sr, target_sr):
    """
    Resample audio data to target sampling rate if needed
    
    Args:
        audio_data (numpy.ndarray): Audio data
        orig_sr (int): Original sampling rate
        target_sr (int): Target sampling rate
        
    Returns:
        numpy.ndarray: Resampled audio data
        int: New sampling rate
    """
    if orig_sr != target_sr:
        audio_data = librosa.resample(audio_data, orig_sr=orig_sr, target_sr=target_sr)
        return audio_data, target_sr
    return audio_data, orig_sr

def save_audio_file(audio_data, file_path, sampling_rate):
    """
    Save audio data to file
    
    Args:
        audio_data (numpy.ndarray): Audio data
        file_path (str): Path to save audio file
        sampling_rate (int): Sampling rate
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    sf.write(file_path, audio_data, sampling_rate)

def save_transcript(text, file_path):
    """
    Save transcript text to file
    
    Args:
        text (str): Transcript text
        file_path (str): Path to save transcript file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text)

def copy_file_if_not_exists(src_path, dest_path):
    """
    Copy file from source to destination if destination doesn't exist
    
    Args:
        src_path (str): Source file path
        dest_path (str): Destination file path
        
    Returns:
        bool: True if file was copied, False if destination already exists
    """
    if not os.path.exists(dest_path):
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy2(src_path, dest_path)
        return True
    return False

def check_textgrids_exist(dirs, split, threshold=0.9):
    """
    Check if TextGrid files already exist for a given split
    
    Args:
        dirs (dict): Dictionary of directory paths
        split (str): Dataset split
        threshold (float): Completion threshold (0.0-1.0)
        
    Returns:
        bool: True if enough TextGrid files exist, False otherwise
    """
    textgrids_dir = os.path.join(dirs['textgrids'], split)
    corpus_dir = os.path.join(dirs['corpus'], split)
    
    # Get all transcript files in corpus
    transcript_files = []
    for root, _, files in os.walk(corpus_dir):
        for file in files:
            if file.endswith('.txt'):
                transcript_files.append(os.path.join(root, file))
    
    if not transcript_files:
        print(f"No transcript files found in corpus for {split} split.")
        return False
    
    # Check if corresponding TextGrid files exist
    existing_count = 0
    total_count = len(transcript_files)
    
    for transcript_path in transcript_files:
        rel_path = os.path.relpath(transcript_path, corpus_dir)
        base_name = os.path.splitext(rel_path)[0]
        textgrid_path = os.path.join(textgrids_dir, base_name + '.TextGrid')
        
        if os.path.exists(textgrid_path):
            existing_count += 1
    
    # Consider alignment complete if percentage of expected TextGrid files exceeds threshold
    completion_ratio = existing_count / total_count if total_count > 0 else 0
    is_complete = completion_ratio >= threshold
    
    if is_complete:
        print(f"Found {existing_count}/{total_count} TextGrid files for {split} split. Skipping alignment.")
    else:
        print(f"Found {existing_count}/{total_count} TextGrid files for {split} split. Alignment needed.")
    
    return is_complete

######################################################
############## Utility for the dataset ###############
######################################################

def load_json(path):
    """Load metadata from JSON or create new empty list."""
    return json.load(open(path, 'r')) if os.path.exists(path) else []

def save_json(path, data):
    """Save metadata to JSON file."""
    with open(path, 'w') as f:
        json.dump(data, f)

def get_temp_json_path(temp_dir, file_name):
    """Get path for temporary JSON file."""
    return os.path.join(temp_dir, f"{file_name}_processed.json")

def python_remove_punctuation(
    input_text: Union[str, List[str]]
) -> Union[str, List[str]]:
    if isinstance(input_text, str):
        return input_text.translate(str.maketrans("", "", string.punctuation))
    elif isinstance(input_text, list):
        return [python_remove_punctuation(text) for text in input_text]
    else:
        raise ValueError("Input must be a string or a list of strings")
    

def parse_textgrid(file_path):

    # things to remove from the textgrid (indicates laughing, chewing, pauses etc)
    REMOVE_CHARACTERS = ['sp', 'br', 'lg', 'cg', 'ls', 'ns', 'sl', 'ig',
                         '{sp}', '{br}', '{lg}', '{cg}', '{ls}', '{ns}', '{sl}', '{ig}', 
                         'pause', '[bracketed]', '<unk>'
                         ]

    tg = textgrid.openTextgrid(file_path, False)
    words = []

    tier = 'words' if 'words' in tg.tierNames else 'word'
    
    for interval in tg.getTier(tier):
        if interval.label.lower() in REMOVE_CHARACTERS:
            continue
        
        words.append({
            "text": python_remove_punctuation(interval.label),
            "start": interval.start,
            "end": interval.end
        })
    return words


######################################################
################# Prosody utilities ##################
######################################################

def process_wavelet_file(filename: str) -> List[Dict[str, Any]]:
    """
    Processes a wavelet prosody toolkit .prom file and returns a list of words.
    Output format is a list of individual words and their prosody values
        [
            {'text': 'word',
            'prominence': float,  # From prominence strength
            'boundary': float,  # From boundary strength
            'start': float,
            'end': float},
        ]
    """
    words = []

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            if len(parts) >= 6:  # Ensure we have all required columns
                file_name, start_time, end_time, unit, prom_strength, bound_strength = parts[:6]
                
                # Skip bracketed units
                if "[" in unit and "]" in unit:
                    continue

                if "<unk>" in unit:
                    continue
                
                prom_strength = float(prom_strength)
                bound_strength = float(bound_strength)
                
                # Add word to current utterance
                word = {
                    "text": unit,
                    "prominence": prom_strength if prom_strength >= 0 else 0, # adding to align with Helsinki dataset
                    "boundary": bound_strength if bound_strength >= 0 else 0,
                    "start": float(start_time),
                    "end": float(end_time)
                }

                words.append(word)

    return words

def interpolate_prosody(current_values: np.array, token_counts: np.array) -> np.ndarray:
    """
    Matrix-based interpolation across multiple tokens.
    
    Args:
        current_values: 2D array of current prosody values for each word (shape: [num_words, num_features])
        token_counts: 1D array or list of token counts for each word (shape: [num_words])
        
    Returns:
        2D array of interpolated prosody values (shape: [total_tokens, num_features])
    """
    
    # Check that the number of words matches
    num_words = current_values.shape[0]
    if len(token_counts) != num_words:
        raise ValueError(f"Mismatch in number of words: current_values has {num_words} words, but token_counts has {len(token_counts)} words.")

    # Calculate the next values for interpolation
    next_values = np.roll(current_values, -1, axis=0)
    next_values[-1] = current_values[-1]  # Handle the last word

    # Compute the steps for each word (shape: [num_words, num_features])
    steps = (next_values - current_values) / token_counts[:, None]

    # Create a matrix of token indices for each word (shape: [num_words, max_token_count])
    max_token_count = np.max(token_counts)
    token_indices = np.tile(np.arange(max_token_count), (num_words, 1))

    # Mask out indices that exceed the token count for each word
    mask = token_indices < token_counts[:, None]

    # Compute interpolated values for all tokens (shape: [num_words, max_token_count, num_features])
    interpolated = current_values[:, None, :] + steps[:, None, :] * token_indices[:, :, None]

    # Flatten the result and apply the mask to remove invalid tokens
    interpolated = interpolated[mask]

    return interpolated

######################################################
################# Audio utilities ####################
######################################################

def load_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)

    # Average channels if 2 channeled
    if waveform.shape[0] == 2:
        waveform = waveform.mean(0).unsqueeze(0)
    
    return waveform.numpy(), sample_rate

def extract_media_segment(media_data, rate, onset, offset, ratios=None, end_tolerance=None, time_axis=0):
    """
    Extract a segment of media data between onset and offset, divided into parts based on the specified ratios.
    Works universally for any n-dimensional array where one axis represents time.

    Args:
        media_data (torch.Tensor or numpy.ndarray): Any n-dimensional array where one axis represents time.
            - For audio: typically (channels, samples) or (samples,)
            - For video: typically (frames, height, width, channels)
        rate (int): The rate of the time axis (sample rate in Hz for audio, fps for video).
        onset (float): The start time of the segment in seconds.
        offset (float): The end time of the segment in seconds.
        ratios (List[float]): A list of ratios to divide the segment into. Defaults to [1.0] (no splitting).
        time_axis (int): The axis representing time in the media_data array. Default is 0.

    Returns:
        List[torch.Tensor or numpy.ndarray]: A list of media segments.
    """
    if ratios is None:
        ratios = [1.0]  # Default to no splitting
        
    # Determine if we're working with torch tensors or numpy arrays
    is_torch = isinstance(media_data, torch.Tensor)
    
    # Convert onset and offset to indices
    start_idx = int(onset * rate)
    end_idx = int(offset * rate)

    media_length = media_data.shape[time_axis]

    # There are more indices than the length of the media
    if end_idx >= media_length:
        # How many seconds we're over by 
        over_by = offset - (media_length/rate)

        # We have set an end tolerance (in s) and we're within that tolerance
        if end_tolerance and over_by <= end_tolerance:
            print (f'Passed tolerance check: time {offset:.2f}/{media_length/rate:.2f}s // over {over_by:.2f}', flush=True)
            end_idx = media_length
        else:
            print (f'Failed tolerance check: time {offset:.2f}/{media_length/rate:.2f}s // over {over_by:.2f}', flush=True)
            return None
    
    # Create array of indices for the time axis
    time_indices = np.arange(start_idx, end_idx)
    total_indices = len(time_indices)
    
    # Calculate indices for each segment based on ratios
    segment_lengths = [int(ratio * total_indices) for ratio in ratios]
    segment_lengths[-1] = total_indices - sum(segment_lengths[:-1])  # Adjust for rounding
    
    # Calculate the cumulative sum to determine segment boundaries
    cumulative_lengths = np.cumsum([0] + segment_lengths)
    
    # Split the indices into segment groups
    index_segments = [time_indices[cumulative_lengths[i]:cumulative_lengths[i+1]] 
                     for i in range(len(segment_lengths))]
    
    # Extract segments using the appropriate array function
    segments = []
    for indices in index_segments:
        if is_torch:
            # Create a torch tensor of indices
            torch_indices = torch.tensor(indices, device=media_data.device, dtype=torch.long)
            # Use index_select for PyTorch tensors
            segment = torch.index_select(media_data, time_axis, torch_indices)
        else:
            # Use take for NumPy arrays
            segment = np.take(media_data, indices, axis=time_axis).squeeze()
        
        segments.append(segment)
    
    return segments

def pool_embeddings(
    embeddings: torch.Tensor,
    attention_mask: torch.Tensor = None,
    pool_type: Literal["mean", "max"] = "mean",
    normalize: bool = True
) -> torch.Tensor:
    """
    Pool transformer embeddings using mean or max pooling.
    
    Args:
        embeddings: Tensor of shape (batch_size, sequence_length, hidden_size)
        attention_mask: Optional boolean mask of shape (batch_size, sequence_length)
        pool_type: Type of pooling to use ("mean" or "max")
        normalize: Whether to L2-normalize the output embeddings
        
    Returns:
        Pooled embeddings tensor of shape (batch_size, hidden_size)
    """
    if attention_mask is not None:
        # Expand mask to match embedding dimensions
        mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size())
        # Zero out padding tokens
        embeddings = embeddings * mask_expanded
    
    if pool_type == "mean":
        if attention_mask is not None:
            # Calculate mean only over non-padding tokens
            sum_embeddings = torch.sum(embeddings, dim=1)
            # Add small epsilon to avoid division by zero
            token_counts = torch.sum(attention_mask, dim=1, keepdim=True).clamp(min=1e-9)
            pooled = sum_embeddings / token_counts
        else:
            pooled = torch.mean(embeddings, dim=1)
            
    elif pool_type == "max":
        if attention_mask is not None:
            # Set padding tokens to large negative value before max
            mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size())
            embeddings = embeddings.masked_fill(~mask_expanded.bool(), float('-inf'))
        pooled = torch.max(embeddings, dim=1)[0]
    
    else:
        raise ValueError("pool_type must be either 'mean' or 'max'")
        
    if normalize:
        pooled = F.normalize(pooled, p=2, dim=-1)
        
    return pooled

######################################################
######### Language classification functions ##########
######################################################

def prepare_audio_batch(fns, sr=16000):

    batch = []

    for fn in fns:
        waveform, audio_sr = load_audio(fn)
        waveform, audio_sr = resample_audio(waveform, orig_sr=audio_sr, target_sr=16000)
        batch.append(waveform.squeeze())

    return batch

def load_language_classifier(model_name="facebook/mms-lid-256"):
    from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    processor = AutoFeatureExtractor.from_pretrained(model_name)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name).to(device)

    return processor, model

def classify_language(batch, processor, model, audio_sr=16000, return_probs=False):

    # Find current device of the model
    device = model.device

    # Takes a set of file names and prepares to pass them to the model
    inputs = processor(batch, sampling_rate=audio_sr, padding=True, return_tensors="pt")

    # Map to the device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs).logits

    # Find highest probability language
    lang_ids = torch.argmax(outputs, dim=-1)

    # Convert the language id to a label
    detected_langs = [model.config.id2label[lang_id.item()] for lang_id in lang_ids]

    if return_probs:
        probs, _ = torch.softmax(outputs, dim=-1).max(-1)
        return detected_langs, probs
    else:
        return detected_langs

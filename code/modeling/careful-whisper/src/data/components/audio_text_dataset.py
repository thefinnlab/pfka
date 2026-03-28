import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from tqdm import tqdm
import os, sys
from typing import Union, Literal, List, Dict, Optional, Any
import numpy as np
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn import functional as F

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from src.models.token_fusion_module import (
    TokenFusionModule,
)

class AudioTextDataset(Dataset):
    def __init__(
        self, 
        data_dir: str,
        split: str = 'test',
        lang_id: str = 'eng',
        metadata_file: str = None,
        token_fusion_method: str = None,
        token_fusion_weights: Optional[List[float]] = None,
        preload: bool = False,  # New preload option
        ckpt_path: str = None,
        shuffle_context_dims: bool = False,
        context_type: str = 'audio_features',
        context_embed_dim: int = 1024,
    ):
        super().__init__()

        # Metadata paths

        ## TLB HARDCODING FOR NOW --> FIX THIS LATER FOR DIRECTORY STRUCTURE
        # Apparently don't need
        
        # if 'avspeech' in data_dir:
        #     metadata_dir = os.path.join(data_dir, split, lang_id)
        # else:
        #     metadata_dir = os.path.join(data_dir, split)

        if not metadata_file:
            metadata_path = os.path.join(data_dir, split, 'metadata.json')
        else:
            metadata_path = os.path.join(data_dir, split, metadata_file)
            print (metadata_path, flush=True)
        
        # Load or create metadata
        self.metadata_path = metadata_path
        self.metadata = self._load_metadata(self.metadata_path)

        self.token_fusion_method = token_fusion_method
        self.token_fusion_weights = token_fusion_weights
        self.preload = preload
        self.ckpt_path = ckpt_path
        self.shuffle_context_dims = shuffle_context_dims
        self.context_type = context_type

        if self.shuffle_context_dims and context_type == 'prominence':
            self.prominence_proj = nn.Linear(1, context_embed_dim, bias=False)
            self.prominence_proj.weight.requires_grad = False

        if self.token_fusion_method == 'mlp' and self.ckpt_path is not None:
            
            # Option 1: If you have the original Lightning Module class
            model = TokenFusionModule.load_from_checkpoint(
                checkpoint_path=self.ckpt_path
            ).to('cpu')

            model.eval()

            # Detach the model from the computation graph and set requires_grad=False
            for param in model.parameters():
                param.requires_grad = False
            
            self.token_fusion_mlp = model
        else:
            self.token_fusion_mlp = None


        # Preload data if required
        if self.preload:
            # self.preloaded_data = self._preload_multiprocessing(split)
            preloaded_data = []
            for idx in tqdm(range(len(self.metadata)), desc=f"Preloading {split} data"):
                preloaded_data.append(self._load_data(idx))
        else:
            self.preloaded_data = None

    #############################################
    ############ Metadata functions #############
    #############################################

    def _load_metadata(self, path):
        # Load or create metadata
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        else:
            return []
    
    def _load_data(self, idx):
        """Helper function to preload data"""
        
        data = self.metadata[idx]
        item = {'text': data['text']}
        
        item.update(
            {x: torch.tensor(data[x]) for x in ['text_tokens', 'attention_mask', 'prominence', 'boundary']}
        )

        # Load audio features if exists

        if 'audio_features_path' in data:
            audio_features = torch.load(data['audio_features_path'])
            item['audio_features'] = torch.nan_to_num(audio_features)

         # Load video features if exists
        if 'video_features_path' in data:
            video_features = torch.load(data['video_features_path'])
            item['video_features'] = torch.nan_to_num(video_features)

        if self.token_fusion_method is not None:
            if self.token_fusion_method == 'average':
                # take mean of the features
                av_features = torch.mean(torch.stack([audio_features, video_features]), dim=0)
            elif self.token_fusion_method == 'mlerp':
                # maximum norm lerp
                av_features = mlerp(
                    token_list=[audio_features, video_features],
                    weights=self.token_fusion_weights
                )
            elif self.token_fusion_method == 'summation':
                # take sum and then normalize
                av_features = audio_features + video_features
                av_features = av_features / torch.norm(av_features)
            elif self.token_fusion_method == 'mlp' and self.token_fusion_mlp is not None:
                # use an mlp to fuse the two features together
                # if audio_features.device != self.token_fusion_mlp.device:
                #     self.token_fusion_mlp = self.token_fusion_mlp.to(audio_features.device)
                with torch.no_grad():  # Add this line to disable gradient tracking
                    av_features = self.token_fusion_mlp(
                        vec1=audio_features,
                        vec2=video_features
                    )[-1]

            item['audiovisual_features'] = torch.nan_to_num(av_features)

        if self.shuffle_context_dims:
            # If prominence is the context, project it from 1D → N first
            if self.context_type == 'prominence':
                prom = item['prominence'].float().unsqueeze(-1)       # (seq_len, 1)
                with torch.no_grad():
                    item['prominence'] = self.prominence_proj(prom)   # (seq_len, N)

            # Shuffle each token's context dims independently, deterministic per sample
            ctx = item[self.context_type]                             # (seq_len, N)
            gen = torch.Generator()
            gen.manual_seed(idx)
            T, N = ctx.shape
            perms = torch.stack([torch.randperm(N, generator=gen) for _ in range(T)])
            item[self.context_type] = ctx.gather(1, perms)

        return item
    
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if self.preload:
            # If preloaded, just return the item directly from the preloaded data
            return self.preloaded_data[idx]
        else:
            # Otherwise, load the data on the fly
            return self._load_data(idx)

def mlerp(token_list, weights=None):
    """
    Implements Maximum Norm Linear Interpolation (MLERP) for token fusion.
    
    MLERP computes the normalized average of tokens, then scales this 
    normalized average using the maximum norm of the individual tokens.
    
    Args:
        token_list (list of torch.Tensor): List of token tensors to merge
                                          Each tensor has shape [N_tokens x N_dim]
    
    Returns:
        torch.Tensor: The merged token tensor with the same shape as inputs

    """
    if weights is None:
        weights = torch.ones(len(token_list))
    else:
        weights = torch.tensor(weights)
    
    # Normalize weights
    weights = weights / weights.sum()
    
    # Weighted average
    token_list = [w * tokens for w, tokens in zip(weights, token_list)]

    # Stack the tokens along a new dimension
    tokens = torch.stack(token_list, dim=0)  # Shape: [K, N_tokens, N_dim]
    
    # Compute the average
    average = torch.mean(tokens, dim=0)  # Shape: [N_tokens, N_dim]
    
    # Compute the norm of the average
    average_norm = torch.norm(average, dim=-1, keepdim=True)  # Shape: [N_tokens, 1]
    
    # Normalize the average
    normalized_average = average / (average_norm + 1e-8)  # Adding epsilon to avoid division by zero
    
    # Compute norms of all input tokens
    token_norms = torch.norm(tokens, dim=-1)  # Shape: [K, N_tokens]
    
    # Find the maximum norm at each token position
    max_norms, _ = torch.max(token_norms, dim=0, keepdim=False)  # Shape: [N_tokens]
    max_norms = max_norms.unsqueeze(-1)  # Shape: [N_tokens, 1]
    
    # Scale the normalized average by the maximum norm
    merged_tokens = normalized_average * max_norms
    
    return merged_tokens
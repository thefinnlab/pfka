import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from argparse import ArgumentError
from typing import Optional, Tuple, List

import os, sys

from pathlib import Path
from lightning import LightningDataModule
from transformers import AutoTokenizer

import torch
from torch.utils.data import DataLoader, Subset

from src.data.components.audio_text_dataset import AudioTextDataset
# from src.data.components.datasets import TokenTaggingDataset
from src.data.components.collators import audio_text_collator

class AudioTextDataModule(LightningDataModule):
    """
    LightningDataModule for HF Datasets.
    Requires a pre-processed (tokenized, cleaned...) dataset provided within the `data` folder.
    Might require adjustments if your dataset doesn't follow the structure of SNLI or MNLI.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        dataset_name: str,
        data_dir: str,
        text_model_name: str = 'gpt2',
        audio_model_name: str = 'wav2vec2',
        video_model_name: str =  'data2vec',
        token_fusion_method: str = 'average',
        token_fusion_weights: Optional[List[float]] = None,
        preload: bool = False,  # New preload option
        ckpt_path: str = None,
        shuffle_context_dims: bool = False,
        context_type: str = 'audio_features',
        context_embed_dim: int = 1024,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        subset_percentage: int = None, 
    ):
        super().__init__()

        # Backwards compatibility
        if dataset_name == 'libritts-r':
            self.hparams.train_split = 'train-clean-360'
            self.hparams.val_split = 'dev-clean'
            self.hparams.test_split = 'test-clean'
        elif dataset_name in ['gigaspeech', 'tedlium', 'peoples-speech']:
            self.hparams.train_split = 'train'
            self.hparams.val_split = 'validation'
            self.hparams.test_split = 'test'
        else:
            self.hparams.train_split = 'train'
            self.hparams.val_split = 'val'
            self.hparams.test_split = 'test'

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hparams.text_model_name, add_prefix_space=False, use_fast=True
        )
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def prepare_data(self):
        """
        We should not assign anything here, so this function simply ensures
        that the pre-processed data is available.
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning twice for `trainer.fit()` and `trainer.test()`, so be careful if you do a random split!
        The `stage` can be used to differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """

        model_combo = self.hparams.text_model_name

        if self.hparams.audio_model_name:
            model_combo += f'-{self.hparams.audio_model_name}'

        if self.hparams.video_model_name:
            model_combo += f'-{self.hparams.video_model_name}'

        self.dataset_path = os.path.join(self.hparams.data_dir, 'features', 'metadata', model_combo)

        print(f"Loading data from {self.dataset_path}")
        if not os.path.exists(self.dataset_path):
            raise ValueError("The provided folder does not exist.")

        if stage == "fit":

            ####################################
            ######## EXTRACT TRAIN DATA ########
            ####################################

            # create datasets
            if self.hparams.subset_percentage:
                metadata_fn = f"metadata_subset-{str(self.hparams.subset_percentage).zfill(3)}.json"
            else:
                metadata_fn = None

            # # else:
            self.train_dataset = AudioTextDataset(
                data_dir=self.dataset_path,
                split=self.hparams.train_split,
                token_fusion_method=self.hparams.token_fusion_method,
                token_fusion_weights=self.hparams.token_fusion_weights,
                metadata_file=metadata_fn,
                preload=self.hparams.preload,
                ckpt_path=self.hparams.ckpt_path,
                shuffle_context_dims=self.hparams.shuffle_context_dims,
                context_type=self.hparams.context_type,
                context_embed_dim=self.hparams.context_embed_dim,
            )

            # # If specified sample a subset of the data
            # if self.hparams.subset_percentage:

            #     # Calculate size of the subset
            #     subset_size = int(self.hparams.subset_percentage * len(self.train_dataset))

            #     # Randomly split the dataset
            #     indices = torch.randperm(len(self.train_dataset)).tolist()
            #     subset_indices = indices[:subset_size]

            #     # Copy over for reference and create a subset
            #     self.full_train_dataset = self.train_dataset
            #     self.train_dataset = Subset(self.train_dataset, subset_indices)

            # print (f"Train samples: {len(self.train_dataset)}", flush=True)

            ####################################
            ######### EXTRACT VAL DATA #########
            ####################################

            # create datasets
            self.val_dataset = AudioTextDataset(
                data_dir=self.dataset_path,
                split=self.hparams.val_split,
                token_fusion_method=self.hparams.token_fusion_method,
                token_fusion_weights=self.hparams.token_fusion_weights,
                preload=self.hparams.preload,
                ckpt_path=self.hparams.ckpt_path,
                shuffle_context_dims=self.hparams.shuffle_context_dims,
                context_type=self.hparams.context_type,
                context_embed_dim=self.hparams.context_embed_dim,
            )

            print (f"Validation samples: {len(self.val_dataset)}", flush=True)

        if stage == "test":

            # create datasets
            self.test_dataset = AudioTextDataset(
                data_dir=self.dataset_path,
                split=self.hparams.test_split,
                token_fusion_method=self.hparams.token_fusion_method,
                token_fusion_weights=self.hparams.token_fusion_weights,
                preload=self.hparams.preload,
                ckpt_path=self.hparams.ckpt_path,
                shuffle_context_dims=self.hparams.shuffle_context_dims,
                context_type=self.hparams.context_type,
                context_embed_dim=self.hparams.context_embed_dim,
            )

            print (f"Test samples: {len(self.test_dataset)}", flush=True)

    def collate(self, batch):
        return audio_text_collator(batch, pad_token=self.tokenizer.pad_token_id)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate,
            shuffle=False,
        )

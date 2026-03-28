import sys
import os
import glob
import json

DATASET_CONFIG = {
	'lrs3': {
		'ckpt_path': "token-fusion_wav2vec-data2vec/checkpoints/epoch_013.ckpt",
		},
	'voxceleb2': {
		'ckpt_path': "token-fusion_wav2vec-data2vec/checkpoints/epoch_008.ckpt"
	},
	'avspeech': {
		'ckpt_path': "token-fusion_wav2vec-data2vec/checkpoints/epoch_006.ckpt"
	},
	'av-combined': {
		'ckpt_path': "token-fusion_wav2vec-data2vec/checkpoints/epoch_006.ckpt"
	}
}

def attempt_makedirs(d):

	if not os.path.exists(d):
		try:
			os.makedirs(d)
		except Exception:
			pass
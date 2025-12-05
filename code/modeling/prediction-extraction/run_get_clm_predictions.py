import os, sys, glob
import json
import numpy as np
import pandas as pd
import argparse
import torch
from torch.nn import functional as F

sys.path.append('../../utils/')

from config import *
import dataset_utils as utils
from tommy_utils import nlp #nlp_utils as nlp

def preproc_to_input(df_preproc, idxs):
	'''
	Given the preprocessed dataframe, extract the transcript text
	over a set of indices to submit to a model
	'''
	
	# get the GT upcoming word to predict
	ground_truth_word = df_preproc.loc[idxs[-1] + 1, 'Word_Written']
	
	# get the segment to submit to the model
	df_segment = df_preproc.iloc[idxs]
	
	all_items = []
	
	for i, row in df_segment.iterrows():
		# sometimes there is punctuation, other times there is whitespace
		# we add in the punctuation as it helps the model but remove trailing
		# whitespaces
		item = ''.join(row[['Word_Written', 'Punctuation']])
		all_items.append(item.strip())
		
	all_items = ' '.join(all_items)
	
	return all_items, ground_truth_word

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-t', '--task', type=str)
	parser.add_argument('-m', '--model_name', type=str)
	parser.add_argument('-w', '--window_size', type=int, default=25)
	parser.add_argument('-n', '--top_n', type=int, nargs='+', default=5)
	parser.add_argument('-s', '--save_logits', type=int, default=0)
	p = parser.parse_args()

	print (f"Model: {p.model_name}")
	print (f'Task: {p.task}')
	print (f'Window Size: {p.window_size}')

	out_dir = os.path.join(BASE_DIR, 'derivatives/model-predictions', p.task, p.model_name, f'window-size-{str(p.window_size).zfill(5)}')
	logits_dir = os.path.join(BASE_DIR, 'derivatives/model-predictions', p.task, p.model_name, f'window-size-{str(p.window_size).zfill(5)}', 'logits')

	utils.attempt_makedirs(out_dir)
	utils.attempt_makedirs(logits_dir)

	# load the preprocessed file --> this has next-word-candidates selected
	stim_preprocessed_fn = os.path.join(BASE_DIR, 'stimuli/preprocessed', p.task, f'{p.task}_transcript-preprocessed.csv')
	df_preproc = pd.read_csv(stim_preprocessed_fn)

	# remap for our functions
	df_preproc = df_preproc.rename(columns={'Word_Written': 'word', 'Punctuation': 'punctuation'})

	# # load a word-level model --> we use glove here
	# # first function downloads the model if needed, second function loads it as gensim format
	word_models = {model_name: nlp.load_word_model(model_name=model_name, cache_dir=CACHE_DIR) for model_name in nlp.WORD_MODELS.keys()}

	# load the causal language model
	print (f'Loading {p.model_name}', flush=True)
	tokenizer, model =  nlp.load_clm_model(model_name=p.model_name, cache_dir=CACHE_DIR)
	print (f'Model loaded', flush=True)

	# Add in device detection
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")
	model.to(device)

	print (f"Model on device: {model.device}")
	
	# add the first word to the dataframe --> we don't run NWP on this as there is no context
	# to condition, nor do we have humans do it
	df = nlp.create_results_dataframe()
	first_word = df_preproc.iloc[0]['word'].lower()
	df.loc[len(df)] = {'ground_truth_word': first_word}

	# set up variables to be used in the loop
	df_stack = {str(n): [df] for n in p.top_n}
	prev_probs = None

	# create a list of indices that we will iterate through to sample the transcript
	segments = nlp.get_segment_indices(n_words=len(df_preproc), window_size=p.window_size)[:-1]

	# we don't need to get the last word
	for i, segment in enumerate(segments):

		ground_truth_index = segment[-1] + 1
		ground_truth_word = df_preproc.loc[ground_truth_index, 'word']
		
		# also keep track of the current ground truth word
		inputs = nlp.transcript_to_input(df_preproc, segment, add_punctuation=True)		

		# Only save logits for NWP candidates if specified and file doesn't exist yet
		logits_fn = os.path.join(logits_dir, f'{p.task}_window-size-{str(p.window_size).zfill(5)}_logits-{str(ground_truth_index).zfill(5)}.pt')
		logits_fn = logits_fn if (df_preproc.loc[ground_truth_index, 'NWP_Candidate'] and 
								 p.save_logits and 
								 not os.path.exists(logits_fn)) else None

		probs = nlp.get_clm_predictions([inputs], model, tokenizer, out_fn=logits_fn)
		
		# now given the outputs of the model, run our stats of interest
		for n in p.top_n:
			segment_stats = nlp.get_model_statistics(ground_truth_word, probs, tokenizer, prev_probs=prev_probs, word_models=word_models, top_n=n)
			df_stack[str(n)].append(segment_stats)

		# now that we've run our stats, set the previous distribution to the one we just ran
		prev_probs = probs
		
		print (f'Processed segment {i+1}/{len(segments)}', flush=True)

	for n in p.top_n:
		df_results = pd.concat(df_stack[str(n)])
		df_results.to_csv(os.path.join(out_dir, f'task-{p.task}_model-{p.model_name}_window-size-{str(p.window_size).zfill(5)}_top-{n}.csv'), index=False)

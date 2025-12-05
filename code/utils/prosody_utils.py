import os, sys
import numpy as np
import pandas as pd
import regex as re
from subprocess import run
from pathlib import Path

import gensim.downloader
from gensim.models import KeyedVectors
from gensim.models import fasttext
from gensim import downloader as api

import fasttext.util as ftutil

#### STUFF FOR TRANSFORMERS ######
import torch
from torch.nn import functional as F
from scipy.special import rel_entr, kl_div
from scipy import stats
from scipy.spatial.distance import cdist, pdist

######################################
########## Prosody metrics ###########
######################################

REMOVE_WORDS = ["sp", "br", "lg", "cg", "ls", "ns", "sl", "ig", "{sp}", "{br}", "{lg}", 
 "{cg}", "{ls}", "{ns}", "{sl}", "{ig}", "SP", "BR", "LG", "CG", "LS",
 "NS", "SL", "IG", "{SP}", "{BR}", "{LG}", "{CG}", "{LS}", "{NS}", "{SL}", "{IG}", "pause"]

def calculate_prosody_metrics(df_prosody, n_prev=3, remove_characters=[], zscore=False):
    # Extract raw values
    prosody_raw = df_prosody['prominence'].to_numpy()
    boundary_raw = df_prosody['boundary'].to_numpy()

    if zscore:
        prosody_raw = stats.zscore(prosody_raw)
    
    # get mean of past n_words
    indices = np.arange(len(prosody_raw))
    start_idxs = indices - n_prev

    # go through the past x words 
    all_items = []
    
    for idx in tqdm(start_idxs):
        # get the prosody of the n_prev words
        if idx >= 0:
            n_prev_prosody = prosody_raw[idx:idx+n_prev]
            n_prev_boundary = boundary_raw[idx:idx+n_prev]
            
            # get mean and std of n_prev words prosody
            prosody_mean = n_prev_prosody.mean()
            prosody_std = n_prev_prosody.std()

            relative_prosody = prosody_raw[idx+n_prev] - prosody_mean
            relative_prosody_norm = relative_prosody / prosody_std

            # get mean and std of n_prev prosodic boundaries
            boundary_mean = n_prev_boundary.mean()
            boundary_std = n_prev_boundary.std()
            
        else:
            prosody_mean = prosody_std = relative_prosody = relative_prosody_norm = np.nan
            boundary_mean = boundary_std = np.nan
        
        all_items.append(
            (prosody_mean, prosody_std, relative_prosody, relative_prosody_norm, boundary_mean, boundary_std)
        )

    prosody_mean, prosody_std, relative_prosody, relative_prosody_norm, boundary_mean, boundary_std = zip(*all_items)

    df_prosody['prominence_mean'] = prosody_mean
    df_prosody['prominence_std'] = prosody_std
    df_prosody['relative_prominence'] = relative_prosody
    df_prosody['relative_prominence_norm'] = relative_prosody_norm
    df_prosody['boundary_mean'] = boundary_mean
    df_prosody['boundary_std'] = boundary_std

    # remove non-words
    df_prosody = df_prosody[~df_prosody['word'].isin(remove_characters)].reset_index(drop=True)
    
    return df_prosody

#######################################
#### Code taken from tommy_utils.nlp ##
#######################################


WORD_MODELS = {
    'glove': 'glove.42B.300d.zip',
    'word2vec': 'word2vec-google-news-300',
    'fasttext': 'cc.en.300.bin'
}

def load_word_model(model_name, cache_dir=None):
    '''
    Given the path to a glove model file,  load the model
    into the gensim word2vec format for ease
    '''

    if cache_dir:
        os.environ['GENSIM_DATA_DIR'] = cache_dir

    if 'glove' in model_name:
        # find the path to our models
        model_name = os.path.splitext(WORD_MODELS[model_name])[0]
        model_dir = os.path.join(cache_dir, model_name)

        model_fn = os.path.join(model_dir, f'gensim-{model_name}.bin')
        vocab_fn = os.path.join(model_dir, f'gensim-vocab-{model_name}.bin')

        # if the files don't already exist, load the files and save out for next time
        print (f'Loading {model_name} from saved .bin file.')

        model = KeyedVectors.load_word2vec_format(model_fn, vocab_fn, binary=True)

    elif 'word2vec' in model_name:
        print (f'Loading {model_name} from saved .bin file.')
        model = api.load(WORD_MODELS[model_name])   
    elif 'fasttext' in model_name:

        print (f'Loading {model_name} from saved .bin file.')
        curr_dir = os.getcwd()

        # set the fasttext directory
        if cache_dir:
            fasttext_dir = os.path.join(cache_dir, 'fasttext')
        else:
            fasttext_dir = os.path.join(os.environ['HOME'], 'fasttext')

        if not os.path.exists(fasttext_dir):
            os.makedirs(fasttext_dir)

        os.chdir(fasttext_dir)

        # # download to the fasttext directory
        ftutil.download_model('en', if_exists='ignore')  # English
        
        os.chdir(curr_dir)
        model = fasttext.load_facebook_vectors(os.path.join(fasttext_dir, WORD_MODELS[model_name]))

    return model


def get_segment_indices(n_words, window_size, bidirectional=False):
    '''
    Given n_words (a total number of words in a transcript) and the 
    size of a context window, return the indices for extracting segments
    of text.
    '''

    if bidirectional:
        indices = []
        for i in range(0, n_words):
            # add right side context of half the window size while increasing left side context
            if i <= window_size // 2:
                idxs = np.arange(0, (i + window_size // 2) + 1)
            # add left side context while reducing right side context size
            elif i >= (n_words - window_size // 2):
                idxs = np.arange((i - window_size // 2), n_words)
            else:
                idxs = np.arange(i - window_size // 2, (i + window_size // 2) + 1)

            indices.append(idxs)
    else:
        indices = [
            np.arange(i-window_size, i) if i > window_size else np.arange(0, i)
            for i in range(1, n_words + 1)
        ]
        
    return indices

def create_results_dataframe():
    
    df = pd.DataFrame(columns = [
        'ground_truth_word',
        'ground_truth_prob',
        'top_n_predictions', 
        'top_prob',
        'binary_accuracy', 
        'glove_avg_accuracy', 
        'glove_max_accuracy',
        'glove_prediction_density',
        'word2vec_avg_accuracy',
        'word2vec_max_accuracy',
        'word2vec_prediction_density',
        'fasttext_avg_accuracy',
        'fasttext_max_accuracy',
        'fasttext_prediction_density',
        'entropy', 
        'relative_entropy'])
    
    return df

def transcript_to_input(df_transcript, idxs, add_punctuation=False):
    '''
    Given the transcript dataframe, extract the transcript text
    over a set of indices to submit to a model
    '''
    
    inputs = []
    
    # go through rows of current segment
    for i, row in df_transcript.iloc[idxs].iterrows():
        # sometimes there is punctuation, other times there is whitespace
        # we add in the punctuation as it helps the model but remove trailing whitespaces

        if add_punctuation:
            item = row['word'] + row['punctuation'].replace('’', "'")
            # prosody = row['prominence']
        else:
            item = row['word']
            # prosody = row['prominence']

        inputs.append(str(item).strip()) #, prosody))
    
    # join together into the sentence to submit
    # inputs, prosody = zip(*inputs)
    inputs = ' '.join(inputs)

    return inputs, df_transcript.iloc[idxs]

def get_word_prob(tokenizer, word, logits, softmax=True):
    
    # use the tokenizer to find the index of each word, 
    idxs = tokenizer(word)['input_ids']

    if softmax:
        probs = F.softmax(logits, dim=-1)
    else:
        probs = logits

    word_prob = probs[:, idxs]
    
    return word_prob.squeeze().mean().item()


def get_word_vector_metrics(word_model, predicted_words, ground_truth_word, method='mean'):
    '''
    Given a word model, a set of response words, and a ground truth word
    evaluate:
         1) semantic similarity to ground truth
         2) cluster density of responses 
    '''

    # how similar was the ground truth to the list of top words
    # make sure we have a word model to use and that the word of interest is a key
    # if the model is fasttext we can perform inference on unknown words
    words_in_model =  any([word in word_model for word in predicted_words])

    if (ground_truth_word in word_model) and (words_in_model):
        # get word vectors from model
        ground_truth_vector = word_model[ground_truth_word][np.newaxis]
        predicted_vectors = [word_model[word] for word in predicted_words if word in word_model]
        predicted_vectors = np.stack(predicted_vectors)

        # calculate cosine similarity
        gt_pred_similarity = 1 - cdist(ground_truth_vector, predicted_vectors, metric='cosine')

        if method == 'max':
            gt_pred_similarity = np.nanmax(gt_pred_similarity)
        elif method == 'mean':
            gt_pred_similarity = np.nanmean(gt_pred_similarity)
        else:
            gt_pred_similarity = gt_pred_similarity

        # calculate spread of predictions as average pairwise distances
        if predicted_vectors.shape[0] != 1:
            pred_distances = pdist(predicted_vectors, metric='cosine')
            pred_distances = np.nanmean(pred_distances).squeeze()
        else:
            pred_distances = np.nan
    else:
        gt_pred_similarity = np.nan
        pred_distances = np.nan

    return gt_pred_similarity, pred_distances

def get_model_statistics(ground_truth_word, probs, tokenizer, prev_probs=None, word_models=None, top_n=1):
    '''
    Given a probability distribution, calculate the following statistics:
        - binary accuracy (was GT word in the top_n predictions)
        - continuous accuracy (similarity of GT to top_n predictions)
        - entropy (certainty of the model's prediction)
        - kl divergence 
    '''
    
    df = create_results_dataframe()
    
    # sort the probability distribution --> apply flip so that top items are returned in order
    top_predictions = np.argsort(probs.squeeze()).flip(0)[:top_n]
    top_prob = probs.squeeze().max().item()
    
    # convert the tokens into words
    top_words = [tokenizer.decode(item).strip().lower() for item in top_predictions]
    ground_truth_word = ground_truth_word.lower()

    # softmax already performed by here, dont need to do again
    ground_truth_prob = get_word_prob(tokenizer, word=ground_truth_word, logits=probs, softmax=False)

    ############################
    ### MEASURES OF ACCURACY ###
    ############################
    
    # is the ground truth in the list of top words?
    binary_accuracy = ground_truth_word in top_words
    
    # go through each model and compute continuous accuracy
    # make sure a word model is defined
    word_model_scores = {}

    if word_models:
        for model_name, word_model in word_models.items():

            avg_pred_similarity, pred_distances = get_word_vector_metrics(word_model, top_words, ground_truth_word)

            max_pred_similarity, _ = get_word_vector_metrics(word_model, top_words, ground_truth_word, method='max')
            
            word_model_scores[model_name] = {
                'avg_accuracy': avg_pred_similarity,
                'max_accuracy': max_pred_similarity,
                'cluster_density': pred_distances
            }
    
    ###############################
    ### MEASURES OF UNCERTAINTY ###
    ###############################
    
    # get entropy of the distribution
    entropy = stats.entropy(probs, axis=-1)[0]
    
    # if there was a previous distribution that we can use, get the KL divergence
    # between current distribution and previous distribution
    if prev_probs is not None:
        kl_divergence = kl_div(probs, prev_probs)
        kl_divergence[torch.isinf(kl_divergence)] = 0
        kl_divergence = kl_divergence.sum().item()
    else:
        kl_divergence = np.nan
        
    df.loc[len(df)] = {
        'ground_truth_word': ground_truth_word,
        'ground_truth_prob': ground_truth_prob,
        'top_n_predictions': top_words,
        'top_prob': top_prob,
        'binary_accuracy': binary_accuracy,
        'glove_avg_accuracy': word_model_scores['glove']['avg_accuracy'],
        'glove_max_accuracy': word_model_scores['glove']['max_accuracy'],
        'glove_prediction_density': word_model_scores['glove']['cluster_density'],
        'word2vec_avg_accuracy': word_model_scores['word2vec']['avg_accuracy'],
        'word2vec_max_accuracy': word_model_scores['word2vec']['max_accuracy'],
        'word2vec_prediction_density': word_model_scores['word2vec']['cluster_density'],
        'fasttext_avg_accuracy': word_model_scores['fasttext']['avg_accuracy'],
        'fasttext_max_accuracy': word_model_scores['fasttext']['max_accuracy'],
        'fasttext_prediction_density': word_model_scores['fasttext']['cluster_density'],
        'entropy': entropy,
        'relative_entropy': kl_divergence,
    }
    
    return df
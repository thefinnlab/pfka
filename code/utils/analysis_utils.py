import os, sys
import glob
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from natsort import natsorted

import nltk
from functools import lru_cache
from itertools import product as iterprod
from statsmodels.stats.multitest import multipletests

from scipy import stats
from scipy.spatial import distance
import scipy.special as special

import torch
from torch.nn import functional as F

from config import *
from tommy_utils import nlp
from preproc_utils import divide_nwp_dataframe, load_model_results
from text_utils import get_pos_tags, get_lemma, strip_punctuation

###############################################
########## Lemmatization functions ############
###############################################

def make_transcript_context(word, df_transcript, word_index, range_display=10):
    """
    Create a context for the given word using the transcript information.

    Parameters:
    word (str): The word for which the context needs to be created.
    df_transcript (pandas.DataFrame): DataFrame containing the transcript information.
    word_index (int): The index of the word in the transcript.
    range_display (int, optional): The number of words to include before and after the target word. Defaults to 10.

    Returns:
    tuple:
        - context (str): The context for the given word.
        - index (int): The index of the word within the context.
    """
    # Calculate the start and end indices for the context
    start_index = max(0, word_index - range_display)
    end_index = min(len(df_transcript), word_index + range_display + 1)

    # Get the words before and after the target word
    start_context = df_transcript['word'].iloc[start_index:word_index]
    end_context = df_transcript['word'].iloc[word_index + 1:end_index]

    # Construct the full context
    context = " ".join(start_context) + " " + str(word) + " " + " ".join(end_context)

    # Calculate the index of the target word within the context
    index = word_index - start_index

    return context, index

def lemmatize_word(word, df_transcript, word_index, remove_stopwords=False):
    """
    Lemmatize a word using the context from the transcript.

    Parameters:
    word (str): The word to be lemmatized.
    df_transcript (pandas.DataFrame): DataFrame containing the transcript information.
    word_index (int): The index of the word in the transcript.
    remove_stopwords (bool, optional): If True, remove stopwords from the lemmatized word. Defaults to False.

    Returns:
    str: The lemmatized word.
    """
    context, index = make_transcript_context(word, df_transcript, word_index)
    lemmatized_word, _ = get_pos_tags([context])[index]
    lemmatized_word = get_lemma(lemmatized_word, _, remove_stopwords=remove_stopwords)
    
    return lemmatized_word

def lemmatize_responses(df_results, df_transcript, response_column='response', debug=False):
    """
    Lemmatize the responses and ground truth words in the given DataFrame.

    Parameters:
    df_results (pandas.DataFrame): DataFrame containing the participant responses.
    df_transcript (pandas.DataFrame): DataFrame containing the transcript information.
    response_column (str, optional): Name of the column containing the responses. Defaults to 'response'.
    debug (bool, optional): If True, print the original and lemmatized words. Defaults to False.

    Returns:
    pandas.DataFrame: The input DataFrame with lemmatized responses and ground truth.
    """

    print (f'Lemmatizing column: {response_column}')

    for _, row in tqdm(df_results.iterrows()):

        # Lemmatize the response
        response_lemma = lemmatize_word(
            word=row[response_column], 
            df_transcript=df_transcript, 
            word_index=row['word_index']
        )

        # Replace response with the lemma
        df_results.at[_, response_column] = response_lemma

        if debug:
            print(f'Word: {row[response_column]} \t Lemma: {response_lemma}')
    
    return df_results

def calculate_response_accuracy(df):
    # compare response to ground truth --> cast as integer
    df['accuracy'] = df['response'] == df['ground_truth']
    df['accuracy'] = df['accuracy'].astype(int)

    return df

# def calculate_results_accuracy(df_results):

#     # compare response to ground truth --> cast as integer
#     df_results['accuracy'] = df_results['response'] == df_results['ground_truth']
#     df_results['accuracy'] = df_results['accuracy'].astype(int)

#     df_accuracy = df_results.groupby(['prolific_id', 'modality', 'subject'])['accuracy'].mean() \
#     .reset_index() \
#     .sort_values(by='accuracy', ascending=True)

#     return df_results, df_accuracy

###############################################
###### Human data compilation functions #######
###############################################

def get_audio_duration(filepath):
    """
    Gets the duration of an audio file in milliseconds.

    Parameters:
    filepath (str): Path to the audio file.

    Returns:
    float: Duration of the audio file in milliseconds.
    """
    y, sr = librosa.load(filepath, sr=None)
    return librosa.get_duration(y=y, sr=sr) * 1000

def get_subject_audio_durations(audio_dir, n_orders=3):
    """
    Get the audio durations for all stimulus files in a directory.

    Parameters:
    audio_dir (str): Directory path containing the stimulus files.
    n_orders (int, optional): Number of stimulus presentation orders. Defaults to 3.

    Returns:
    pandas.DataFrame: DataFrame containing the stimulus order, audio filename, and duration.
    """
    # Create a DataFrame to store the results
    columns = ['stim_order', 'audio_filename', 'duration']
    df = pd.DataFrame(columns=columns)

    # Iterate over the stimulus orders
    for order in range(1, n_orders + 1):
        order_dir = os.path.join(audio_dir, f'sub-{str(order).zfill(5)}')
        audio_files = sorted(glob.glob(os.path.join(order_dir, '*')))

        # Iterate over the audio files and get the durations
        for audio_file in audio_files:
            duration = get_audio_duration(audio_file)
            df = df.append({
                'stim_order': order - 1,
                'audio_filename': audio_file,
                'duration': duration
            }, ignore_index=True)

    return df

def load_participant_results(sub_dir, sub):
    """
    Load and preprocess participant results from a CSV file.

    Parameters:
    sub_dir (str): Directory path containing the CSV file.
    sub (str): Participant ID.

    Returns:
    tuple:
        - prolific_id (str): The participant's Prolific ID.
        - demographics (pandas.DataFrame): Demographic information (age, race, ethnicity, gender).
        - experience (pandas.DataFrame): Participant's experience with moths and stories.
        - responses (pandas.DataFrame): Participant's responses during the test phase.
    """

    # Load and filter down to response trials
    df_results = pd.read_csv(os.path.join(sub_dir, f'{sub}_next-word-prediction.csv')).fillna(False)
    df_results['word_index'] = df_results['word_index'].astype(int)

    # Grab the prolific id
    prolific_id = list(set(df_results['prolific_id']))[0]

    # Filter down demographics
    demographics = df_results[df_results['experiment_phase'].str.contains('demographics').fillna(False)]
    demographics = demographics[['experiment_phase', 'response']].reset_index(drop=True)
    assert len(demographics) == 4, "Expected 4 demographic answers"

    # Filter down to questions about moth/story experience
    experience = df_results[df_results['experiment_phase'].str.contains('experience').fillna(False)]
    experience = experience[['experiment_phase', 'response']].reset_index(drop=True)
    assert len(experience) == 2, "Expected 2 experience answers"
    
    # Filter down to get the responses
    response_indices = df_results['experiment_phase'] == 'test'

    ## TLB DOUBLE CHECK THAT AN EQUIVALENT THING WAS IMPLEMENTED IN THE CLEANING NOTEBOOK
    # if 'video' in sub_dir:
    #     response_indices = np.where(response_indices)[0]
    #     responses = df_results.iloc[response_indices[:-1], :].reset_index(drop=True)
    # else:
    responses = df_results[response_indices]   

    responses.loc[:,'response'] = responses['response'].str.lower()
    responses = responses[['critical_word', 'word_index', 'entropy_group', 'accuracy_group', 'response', 'rt']].reset_index(drop=True)

    return prolific_id, demographics, experience, responses


def aggregate_participant_responses(results_dir, audio_dir, task, modality, n_orders=3, debug=False):
    """
    Aggregate participant responses for a given task and modality.

    Parameters:
    results_dir (str): Directory path containing the participant results.
    audio_dir (str): Directory path containing the stimulus files.
    task (str): Name of the task.
    modality (str): Modality of the task (e.g., 'audio', 'visual').
    n_orders (int, optional): Number of stimulus presentation orders. Defaults to 3.

    Returns:
    pandas.DataFrame: Aggregated participant responses.
    """
    print(f'Aggregating {task} - {modality}')

    # Initialize the results DataFrame
    columns = ['prolific_id', 'modality', 'subject', 'word_index', 'response', 'ground_truth', 'entropy_group', 'accuracy_group', 'rt']
    df_results = pd.DataFrame(columns=columns)

    # Get the subject audio durations
    df_order_durations = get_subject_audio_durations(os.path.join(audio_dir, task), n_orders=n_orders)

    # Get subject directories
    sub_dirs = sorted(glob.glob(os.path.join(results_dir, task, modality, f'sub*')))
    
    print(f'Total of {len(sub_dirs)} subjects')

    for sub_dir in tqdm(sub_dirs):
        src_dir = sub_dir.replace('cleaned-results', 'results')
        sub = os.path.basename(sub_dir)

        # Get the current stimulus order
        current_order = (int(sub.split('-')[-1]) - 1) % n_orders
        df_duration = df_order_durations[df_order_durations['stim_order'] == current_order].reset_index(drop=True)

        if debug:
            print(f'Subject: {sub}')
            print(f'Current order: {current_order}')

        if os.path.exists(sub_dir):
            # Load original responses
            _, _, _, src_responses = load_participant_results(src_dir, sub)

            # Load cleaned results
            current_id, demographics, experience, responses = load_participant_results(sub_dir, sub)
            responses['response'] = responses['response'].fillna('')

            if modality == 'video':
                src_responses = src_responses.iloc[:-1, :].reset_index(drop=True)

            # Append responses to the results DataFrame
            for index, row in responses.iterrows():

                if isinstance(row['rt'], str) and (row['rt'].lower() == 'false'):
                    rt = np.nan
                else:
                    rt = float(row['rt']) - df_duration.loc[index, 'duration']
                
                src_response = src_responses.iloc[index, :]

                # Find whether it was one-word or multiple words
                correction_lengths = len(str(src_response['response']).split())

                # Divide into spell-corrected vs. shortened
                spell_corrected = correction_lengths == 1
                shortened = correction_lengths > 2

                df_results = df_results.append({
                    'prolific_id': current_id,
                    'modality': modality,
                    'subject': sub,
                    'word_index': row['word_index'],
                    'response': row['response'],
                    'ground_truth': row['critical_word'].lower(),
                    'entropy_group': row['entropy_group'],
                    'accuracy_group': row['accuracy_group'],
                    'rt': rt,
                    'spell_corrected': spell_corrected,
                    'shortened': shortened,
                }, ignore_index=True)
        else:
            print(f'File not exists: {modality}, {sub}')

    # Calculate results one-shot accuracy
    df_results['accuracy'] = (df_results['response'] == df_results['ground_truth']).astype(int)

    return df_results

###############################################
########### Phoneme leakage checks ############
###############################################

try:
    arpabet = nltk.corpus.cmudict.dict()
except LookupError:
    nltk.download('cmudict')
    arpabet = nltk.corpus.cmudict.dict()

@lru_cache()
def wordbreak(s):
    s = s.lower()
    if s in arpabet:
        return arpabet[s]
    middle = len(s)/2
    partition = sorted(list(range(len(s))), key=lambda x: (x-middle)**2-x)
    for i in partition:
        pre, suf = (s[:i], s[i:])
        if pre in arpabet and wordbreak(suf) is not None:
            return [x+y for x,y in iterprod(arpabet[pre], wordbreak(suf))]
    return None

def compare_wrong_phonemes(responses, ground_truth, n_phones=1):
    '''
    Find match of first n_phonemes to the ground truth
    '''

    # Grab ground truth phoneme
    ground_truth_phoneme = wordbreak(ground_truth)[0][0]

    # Get first phoneme of each response
    predicted_phoneme = [wordbreak(word) for word in responses]
    predicted_phoneme = [phone[0][0] if phone is not None else '' for phone in predicted_phoneme]
    
    unique, counts = np.unique(predicted_phoneme, return_counts=True)

    # Find accuracy of first letter
    phoneme_accuracy = np.asarray([phone == ground_truth_phoneme for phone in predicted_phoneme])

    # Focus on wrong responses
    wrong_responses = np.argwhere(responses != ground_truth)

    # Make sure there is at least 1 wrong response
    if len(wrong_responses) >= 1:

        # Extract the wrong response phonemes and accompanying accuracy
        wrong_phonemes = np.asarray(predicted_phoneme)[wrong_responses]
        wrong_accuracy = phoneme_accuracy[wrong_responses]

        # Get number of correct responses for the wrong response
        wrong_resp_correct = np.sum(wrong_accuracy)
        wrong_resp_incorrect = len(wrong_accuracy) - wrong_resp_correct
        wrong_resp_accuracy = wrong_resp_correct / len(wrong_accuracy)
    else:
        wrong_resp_correct = np.nan 
        wrong_resp_incorrect = np.nan 
        wrong_resp_accuracy = np.nan

    return wrong_resp_correct, wrong_resp_incorrect, wrong_resp_accuracy

def get_leakage_stats(df_cleaned_behavior, comparison_modality='text', stat_test='barnard', alpha=0.05, correction=None):
    """
    Calculate leakage statistics by comparing each modality to a specified comparison modality.
    
    Parameters:
    -----------
    df_cleaned_behavior : pandas.DataFrame
        DataFrame containing behavior data
    comparison_modality : str
        The modality to compare all other modalities against (default: 'text')
    stat_test : str
        Statistical test to use ('barnard', 'fisher', or 'chi2')
    correction : str
        Multiple comparison correction method
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added statistics columns
    """
    df_stack = []
    phone_columns = ['wrong_resp_n_correct', 'wrong_resp_n_incorrect', 'wrong_resp_accuracy']
    stats_columns = [f'{stat_test}_stat', f'{stat_test}_pvalue']

    # Go through each word index for the current task
    for word_index, df_index in df_cleaned_behavior.groupby(['word_index']): 
        
        # First, calculate phoneme comparisons for all modalities
        modality_data = {}
        for modality, df_modality in df_index.groupby('modality'):
            # Grab ground truth and responses
            ground_truth = df_modality['ground_truth'].unique()[0]
            responses = df_modality['response'].to_numpy()
            
            # Compare response phonemes to the ground truth
            phoneme_match = compare_wrong_phonemes(responses, ground_truth, n_phones=1)
            
            # Save the contingency data and dataframe for this modality
            modality_data[modality] = {
                'contingency': phoneme_match[:-1],  # Exclude accuracy which is the last element
                'df': df_modality.assign(**dict(zip(phone_columns, phoneme_match)))
            }
        
        # Skip if we don't have the comparison modality
        if comparison_modality not in modality_data:
            continue
            
        comparison_contingency = modality_data[comparison_modality]['contingency']
        comparison_results = {}

        # Now compare each modality to the comparison modality
        for modality, data in modality_data.items():
            # Skip comparing the comparison modality to itself
            if modality == comparison_modality:
                # Just add NaN for stat and pvalue for the comparison modality
                comparison_results[modality] = [np.nan, np.nan]
                continue
                
            # Create a 2x2 contingency table for this modality vs comparison
            contingency = np.stack([data['contingency'], comparison_contingency])
            
            # Conduct statistical test
            if not np.isnan(contingency).any():
                if stat_test == 'barnard':
                    results = stats.barnard_exact(contingency, alternative='greater')
                elif stat_test == 'fisher':
                    results = stats.fisher_exact(contingency, alternative='greater')
                elif stat_test == 'chi2':
                    results = stats.chi2_contingency((contingency + 1))
                statistic, pvalue = results.statistic, results.pvalue
            else:
                statistic, pvalue = np.nan, np.nan
                
            # Store the statistics
            comparison_results[modality] = [statistic, pvalue]

        # Grab the significant for all modalities tested
        pvals = np.array([results[-1] for modality, results in comparison_results.items() if modality != comparison_modality])
        filter_significant = (pvals < alpha).all()

        # Now add the results into the dataframe including the filter column
        for modality, leakage_stats in comparison_results.items():
            data = modality_data[modality]

            insert_columns = stats_columns + ['filter_for_leakage'] 
            insert_values = leakage_stats + [filter_significant]
            data['df'].loc[:, insert_columns] = insert_values
            df_stack.append(data['df'])    # Concatenate all dataframes

    if not df_stack:
        return pd.DataFrame()  # Return empty DataFrame if no data
        
    df_stack = pd.concat(df_stack).reset_index(drop=True)
    
    # Correct for multiple comparisons across word indices
    if correction is not None:
        # Add stat in as modified column
        stat_name = stats_columns[-1] + f'_{correction}'
        
        # Correct for multiple comparisons separately for each modality
        for modality in df_stack['modality'].unique():
            # Skip the comparison modality
            if modality == comparison_modality:
                continue
                
            df_modality = df_stack[df_stack['modality'] == modality]
            word_indices = df_modality['word_index'].unique()
            
            # Grab each unique pvalue
            pvalues = np.asarray([
                df_modality[df_modality['word_index'] == idx][stats_columns[-1]].unique()[0]
                for idx in word_indices
            ])
            
            # Filter nans before correcting for multiple comparisons
            nan_filter = ~np.isnan(pvalues)
            if np.any(nan_filter):
                pvalues[nan_filter] = multipletests(pvalues[nan_filter], method=correction)[1]
            
            # Update the values in the original dataframe
            for idx, pval in zip(word_indices, pvalues):
                df_stack.loc[
                    (df_stack['modality'] == modality) & (df_stack['word_index'] == idx),
                    stat_name
                ] = pval
    
    df_stack = df_stack.sort_values(by=['modality', 'subject']).reset_index(drop=True)
    
    return df_stack

###############################################
########### Analysis of human data ############
###############################################

def get_human_probs(responses):
    
    unique, counts = np.unique(responses, return_counts=True)
    probs = counts / sum(counts)
    
    return probs, unique

# def strip_punctuation(text):
    
#     full_text = re.sub('[^A-Za-z0-9]+', '', text)
    
#     return full_text

def analyze_human_results(df_transcript, df_results, word_model_info, window_size=25, top_n=None, drop_rt=None, drop_shortened=False):
    """

    Compile results from all subjects into a distribution of "humans"

    """

    # Decide if we want to filter RTs
    if drop_rt:
        print (f'Dropping trials with RTs longer than {drop_rt} seconds')
        drop_rt = drop_rt * 1000

    # Load word-level and sentence-level models
    word_model_name, word_model = word_model_info
    modality = np.unique(df_results['modality'])[0]

    # Clean responses for nans
    df_results.loc[:, 'response'] = df_results['response'].apply(lambda x: strip_punctuation(x).strip() if isinstance(x, str) else '')

    # Instantiate the dataframe
    df_analysis = pd.DataFrame(columns=[
        'modality',
        'word_index',
        'ground_truth',
        'top_pred',
        'accuracy',
        'predictability',
        'top_prob',
        'n_predictions',
        'entropy',
        'normalized_entropy',
        'bert_top_word_accuracy',
        f'{word_model_name}_top_word_accuracy',
        f'{word_model_name}_avg_accuracy',
        f'{word_model_name}_max_accuracy',
        f'{word_model_name}_weighted_pred-gt_accuracy',
        f'{word_model_name}_prediction_distances',
        f'{word_model_name}_weighted_prediction_distances',
        f'{word_model_name}_centroid_prediction_distances',
        'entropy_group',
        'accuracy_group',
        'n_rt_drops',
        'average_rt',
        'std_rt',
    ])

    # Load sentence transformer + get indices of transcript segments for the current window size
    tokenizer, model = nlp.load_mlm_model(model_name='sentence-transformers/all-mpnet-base-v2', cache_dir=CACHE_DIR)
    segments = nlp.get_segment_indices(n_words=len(df_transcript), window_size=window_size, bidirectional=True)
    
    # Go through each response word 
    for (modality, index), df_index in tqdm(df_results.groupby(['modality', 'word_index'])):

        # Grab global information across participants
        ground_truth, entropy_group, accuracy_group = df_index \
            .reset_index(drop=True) \
            .loc[0, ['ground_truth', 'entropy_group', 'accuracy_group']]

        ##############################################
        ##### Drop responses that were shortened #####
        ##############################################

        if drop_shortened:
            df_index = df_index.loc[df_index["shortened"] == False]

        ##############################################
        ### If we want to drop trials based on RTs ###
        ##############################################

        # find RTs less than the desired RT and filter responses
        if drop_rt:
            rt_filter = df_index['rt'] <= drop_rt
            df_index = df_index.loc[rt_filter, :]
        
            print (f'Kept {sum(rt_filter)} responses')
    
        # Responses for the current modality
        human_responses = df_index['response'].apply(strip_punctuation)
        human_responses = list(filter(None, human_responses))

        rts = df_index['rt']

        ####################################################
        ### Get probabilities for words across responses ###
        ####################################################

        # get probabilities and unique words
        human_probs, unique_words = get_human_probs(human_responses)
        
        #######################################
        #### Calculate probability metrics ####
        #### 1. Predictability             ####
        #### 2. Entropy                    ####
        #### 3. Top probabiltity           ####
        #######################################

        # predictability is the count of number of accurate predictions over total predictions
        predictability = sum(np.asarray(human_responses) == ground_truth) / len(human_responses)
        
        # entropy + entropy normalized by the number of items in the distribution
        entropy = stats.entropy(human_probs)
        normalized_entropy = entropy / np.log(len(human_probs))

        #############################################
        #### Calculate accuracy metrics          ####
        #### 1. Binary accuracy                  ####
        #### 2. Word continuous accuracy         ####
        #### 3. Contextual continuous accuracy   ####
        #############################################

        # Sort probabilities from highest to lowest
        sorted_prob_idxs = np.argsort(human_probs)[::-1]
        sorted_probs = human_probs[sorted_prob_idxs]
    
        # If we are using top-N words (e.g., top 5), trim down to those words
        if top_n is not None and len(unique_words) < top_n:
            # Select top-N from all unique words and sort from most to least
            all_word_idxs = sorted_prob_idxs[:len(unique_words)]
            top_n_words = unique_words[all_word_idxs]
        else:
            # Otherwise top-N words are all unique words --> just sort the words
            top_word_idxs = sorted_prob_idxs[:top_n]
            top_n_words = unique_words[top_word_idxs]

        # Top word is the first word -- Top prob is the associated probability
        top_word = top_n_words[0]
        top_prob = sorted_probs[0]

        #####################################################
        ####### 1. Binary accuracy: one-shot accuracy #######
        #####################################################

        accuracy = int(top_word == ground_truth)

        ##########################################################################
        ####### 2. Word continuous accuracy:                             #########
        #######    Cosine similarity between prediction and ground-truth #########
        #######    using the current word model                          #########
        ##########################################################################

        top_word_accuracy, _ = nlp.get_word_vector_metrics(word_model, [top_word], ground_truth)
        max_pred_similarity, _ = nlp.get_word_vector_metrics(word_model, top_n_words, ground_truth, method='max')

        ##########################################################################
        ####### 3. Contextual continuous accuracy:                       #########
        #######    Cosine similarity in BERT space using surrounding     #########
        #######    transcript context for prediction and ground-truth    #########
        ##########################################################################

        # Find index of the word within its surrounding context
        current_segment = segments[index]
        word_index = np.where(current_segment == index)[0]

        # Create a dataframe and substitute the top word
        df_substitute = df_transcript.copy()
    
        # Make input and calculate contextual embedding for ground truth
        df_substitute.loc[index, 'word'] = ground_truth
        inputs = nlp.transcript_to_input(df_substitute, idxs=current_segment)

        if window_size < 400:
            bert_ground_truth_embedding = nlp.extract_word_embeddings([inputs], tokenizer, model, word_indices=word_index).squeeze()
            bert_ground_truth_embedding = bert_ground_truth_embedding[-1, :][np.newaxis]

            # Repeat for the predicted word
            df_substitute.loc[index, 'word'] = top_word
            inputs = nlp.transcript_to_input(df_substitute, idxs=current_segment)
            bert_top_word_embedding = nlp.extract_word_embeddings([inputs], tokenizer, model, word_indices=word_index).squeeze()
            bert_top_word_embedding = bert_top_word_embedding[-1, :][np.newaxis]

            # Calculate cosine similarity between ground truth and prediction   
            bert_similarity = 1 - distance.cdist(bert_ground_truth_embedding, bert_top_word_embedding, metric='cosine').squeeze()
        else:
            bert_similarity = np.nan

        #############################################
        #### Prediction density metrics          ####
        #### 1. Binary accuracy                  ####
        #### 2. Word continuous accuracy         ####
        #### 3. Contextual continuous accuracy   ####
        #############################################

        # pred_similarity = similarity of all words to ground truth word
        # pred_distances = similarity of all words from each other
        pred_similarity, pred_distances = nlp.get_word_vector_metrics(word_model, top_n_words, ground_truth)
        
        # Weight these scores by the probability (e.g., more probable items contribute more)
        weighted_pred_distances = np.nanmean(pred_distances * sorted_probs)
        weighted_pred_similarity = np.nanmean(pred_similarity * sorted_probs)

        # Calculate distance spread of predicted vectors from the centroid of the vectors
        predicted_vectors = [word_model[word] for word in unique_words if word in word_model]
        predicted_vectors = np.stack(predicted_vectors)

        centroid_distance = distance.cdist(predicted_vectors.mean(0)[np.newaxis], predicted_vectors, metric='cosine')
        centroid_distance = np.nanmean(centroid_distance)

        # pred_distances = nan when there was only one prediction
        if np.isnan(pred_distances):
            pred_distances = 0

        #############################################
        ######## Grab phoneme control stats #########
        #############################################

        df_analysis.loc[len(df_analysis)] = {
            'modality': modality,
            'word_index': index,
            'ground_truth': ground_truth,
            'top_pred': top_word,
            'accuracy': accuracy,
            'predictability': predictability,
            'top_prob': top_prob,
            'n_predictions': len(unique_words),
            'entropy': entropy,
            'normalized_entropy': np.nan_to_num(normalized_entropy), 
            'bert_top_word_accuracy': bert_similarity,
            f'{word_model_name}_top_word_accuracy': top_word_accuracy,
            f'{word_model_name}_avg_accuracy': np.nanmean(pred_similarity),
            f'{word_model_name}_max_accuracy': max_pred_similarity,
            f'{word_model_name}_weighted_pred-gt_accuracy': weighted_pred_similarity,
            f'{word_model_name}_prediction_distances': pred_distances,
            f'{word_model_name}_weighted_prediction_distances': weighted_pred_distances,
            f'{word_model_name}_centroid_prediction_distances': centroid_distance,
            'entropy_group': entropy_group,
            'accuracy_group': accuracy_group,
            'n_rt_drops': sum(rt_filter) if drop_rt is not None else 0,
            'average_rt': rts.mean(),
            'std_rt': rts.std(),
        }
    
    return df_analysis

###############################################
########### Analysis of LLM data ##############
###############################################

def load_logits(model_dir, model_name, task, window_size, word_index):
    '''
    Loads model data from directory
    '''

    if 'whisper' in model_name:
        model_dir = os.path.join(model_dir, task, 'careful-whisper', model_name, f'window-size-{str(window_size).zfill(5)}')
    else:
        model_dir = os.path.join(model_dir, task, model_name, f'window-size-{str(window_size).zfill(5)}')

    logits_fns = natsorted(glob.glob(os.path.join(model_dir, 'logits', f'*{str(word_index).zfill(5)}*.pt')))
    
    assert (len(logits_fns) == 1)
    
    return torch.load(logits_fns[0])
    
def get_model_word_quadrants(models_dir, model_name, task, window_size=25, top_n=5, candidate_rows=None, accuracy_type='fasttext_avg_accuracy', accuracy_percentile=50):

    # select based on model quadrants --> trim down to only the words of interest
    df_model_results = load_model_results(models_dir, model_name=model_name, task=task, window_size=window_size, top_n=top_n)
    df_model_results = df_model_results.iloc[candidate_rows]
    
    # now grab the current model divided over the 50th percentile
    # while we originally divided words on the 45th percentile of gpt2, we want to see patterns across models
    df_divide = divide_nwp_dataframe(df_model_results, accuracy_type=accuracy_type, percentile=accuracy_percentile, drop=False)
    
    return df_divide

def analyze_model_accuracy(df_transcript, word_model_info, models_dir, model_name, task, top_n=1, window_size=25, candidate_rows=None, lemmatize=False):
    """
    Perform analysis of model predictions (similar to analyze_human_results). Formats
    the model predictions dataframe similar to the human dataframe so the two can be collapsed.

    """

    # Get word model information
    word_model_name, word_model = word_model_info

    # Rows for words used in the next-word prediction experiment
    # Columns of the model predictions that we care about for comparison
    selected_rows = np.where(df_transcript['NWP_Candidate'])[0]
    selected_columns = ['ground_truth_word', 'top_n_predictions', 'top_prob', 'ground_truth_prob', f'{word_model_name}_max_accuracy', f'{word_model_name}_avg_accuracy', 'entropy']

    # Load results for the specified model -- this is output from the prediction-extraction script
    df_model_results = load_model_results(models_dir, model_name=model_name, task=task, top_n=top_n, window_size=window_size)

    # Grab model binarized entropy/accuracy quadrants
    df_model_quadrants = get_model_word_quadrants(models_dir, model_name=model_name, task=task, window_size=window_size, candidate_rows=candidate_rows)
    df_model_quadrants = df_model_quadrants.loc[selected_rows, ['entropy_group', 'accuracy_group']]

    # Select only the words used for next-word prediction
    df_model_results = df_model_results.loc[selected_rows, selected_columns].reset_index()
    df_model_results['top_n_predictions'] = df_model_results['top_n_predictions'].str[0] # get the top predicted word

    # Rename the columns to make the dataframe in line with human results
    df_model_results = df_model_results.rename(columns={
            'index': 'word_index',
            'ground_truth_word': 'ground_truth',
            'ground_truth_prob': 'predictability', 
            'top_n_predictions': 'top_pred',
            f'{word_model_name}_max_accuracy': f'{word_model_name}_top_word_accuracy',
    })

    # Add other information to the dataframe
    df_model_results['modality'] = model_name
    df_model_results['accuracy'] = (df_model_results['top_pred'] == df_model_results['ground_truth']).astype(int) # binary accuracy
    df_model_results[['entropy_group', 'accuracy_group']] = df_model_quadrants.loc[:, ['entropy_group', 'accuracy_group']].reset_index(drop=True)

    print (f"Total missing values: {df_model_results[f'{word_model_name}_top_word_accuracy'].isna().sum()}")

    # If specified lemmatize and recalculate accuracy
    if lemmatize:
        df_model_results = lemmatize_responses(df_model_results, df_transcript, response_column='top_pred')
        df_model_results = lemmatize_responses(df_model_results, df_transcript, response_column='ground_truth')

        for i, row in tqdm(df_model_results.iterrows()):

            top_word, ground_truth = row[['top_pred', 'ground_truth']]

            # Calculate binary & continuous accuracy
            accuracy = int(top_word == ground_truth)
            top_word_accuracy, _ = nlp.get_word_vector_metrics(word_model, [top_word], ground_truth)

            # Update dataframe
            df_model_results.loc[i, ['accuracy', f'{word_model_name}_top_word_accuracy']] = [accuracy, top_word_accuracy]

    return df_model_results

def compare_human_model_distributions(df_human_results, word_model_info, models_dir, model_name, task, top_n=1, window_size=25, lemmatize=False):

    # Get word model information
    word_model_name, word_model = word_model_info

    # Load the tokenizer 
    tokenizer, _ = nlp.load_clm_model(
        model_name='gpt2' if 'whisper' in model_name else model_name, 
        cache_dir=CACHE_DIR
    )

    df_comparison = pd.DataFrame(columns=[
        'model_name',
        'modality',
        'word_index',
        'ground_truth',
        'entropy_group',
        'accuracy_group',
        'human_top_word',
        'human_prob',
        'human_predictability',
        'human_log_odds_predictability',
        'human_entropy',
        'model_top_word',
        'model_prob',
        'model_predictability',
        'model_log_odds_predictability',
        'model_entropy',
        'model_prob_adjusted',
        'model_prob_human_prediction',
        'kl_divergence',
        'earthmovers_dist',
        'jensenshannon_dist',
        'ks_stat',
        'human_model_pred_similarity',
        'human_model_accuracy_diff',
    ])

    # Go through each word index in the current modality
    for (modality, word_index), df_index in tqdm(df_human_results.groupby(['modality', 'word_index'])):

        # Grab global information across participants
        ground_truth, entropy_group, accuracy_group = df_index \
            .reset_index(drop=True) \
            .loc[0, ['ground_truth', 'entropy_group', 'accuracy_group']]
        
        # All responses (across modalities) for the current predicted word
        word_index_filter = df_human_results['word_index'] == word_index
        all_responses = df_human_results.loc[word_index_filter, 'response'].apply(strip_punctuation)
        all_responses = list(filter(None, all_responses))

        # Responses for the current modality
        human_responses = df_index['response'].apply(strip_punctuation)
    
        ##############################################
        #### Remove blank reponses from human data ###
        #############################################
        
        pre_filter = len(human_responses)
        human_responses = list(filter(None, human_responses))
        post_filter = len(human_responses)
    
        if pre_filter != post_filter:
            print (f'Removed {pre_filter - post_filter} empty responses')

        ###############################################
        ##### Calculate model probability metrics #####
        ###############################################

        # Load logits for the current word
        try:
            model_logits = load_logits(models_dir, model_name, task, window_size, word_index)
        except:
            print (f'Missing logits for {task} {word_index}', flush=True)
            continue
            
        # Turn logits into probabilities and get top probability and prediction
        model_dist = F.softmax(model_logits, dim=-1).squeeze()
        model_prob = model_dist.max().item()
        model_prediction = tokenizer.decode(model_dist.argmax()).strip()

        # Entropy of model distribution
        model_entropy = stats.entropy(model_dist)
        
        # Token for ground truth word & predictability (probability of ground truth word)
        ground_truth_token = tokenizer.encode(ground_truth)
        model_predictability = model_dist[ground_truth_token].mean(0).item() # average over items in case it is multiple tokens
        model_log_odds_predictability = special.logit(model_predictability)

        ###############################################
        ##### Calculate human probability metrics #####
        ###############################################

        # Probability distribution for humans & associated words
        human_dist, unique_words = get_human_probs(human_responses)
        human_prob = human_dist.max()
        human_prediction = unique_words[human_dist.argmax()]

        # Entropy of human distribution
        human_entropy = stats.entropy(human_dist)

        # Find respective human predictability
        human_predictability = sum(np.asarray(human_responses) == ground_truth) / len(human_responses)
        
        # Substitute small value to avoid inf error
        if human_predictability == 0:
            human_log_odds_predictability = special.logit(1e-2)
        else:
            human_log_odds_predictability = special.logit(human_predictability)

        ########################################################
        ###### Compare semantic similarity of predictions ######
        ########################################################

        if lemmatize:
            try:
                _lemmatized = get_lemma(model_prediction)
                model_prediction = _lemmatized if _lemmatized is not None else model_prediction
                
                # Modify ground truth to include lemma
                _ground_truth = get_lemma(ground_truth)
                ground_truth = _ground_truth if _ground_truth is not None else ground_truth

            except:
                pass
        
        human_vector = word_model[human_prediction]
        model_vector = word_model[model_prediction]

        pred_similarity = 1 - distance.cdist(human_vector[np.newaxis], model_vector[np.newaxis], metric='cosine').squeeze()

        ########################################################
        ######## Get difference in prediction accuracy #########
        ########################################################

        # Grab human and model top word accuracy
        human_top_word_accuracy, _ = nlp.get_word_vector_metrics(word_model, [human_prediction], ground_truth)
        model_top_word_accuracy, _ = nlp.get_word_vector_metrics(word_model, [model_prediction], ground_truth)

        accuracy_difference = human_top_word_accuracy - model_top_word_accuracy

        ########################################################
        ##### Trim model distribution to human predictions #####
        ########################################################

        # Add entries for words predicted within the other modality
        temp = np.zeros(len(all_responses)) # initialize an empty array
        word_idxs = [all_responses.index(word) for word in unique_words] # get indices of current responses within that array
        temp[word_idxs] = human_dist # insert probabilities
        human_dist = temp # set to the human distribution
        
        # Match the model distribution to the human distribution
        # Find probability of words human chose in the model distribution --> then normalize
        model_adjusted_dist = np.asarray([nlp.get_word_prob(tokenizer, word, model_logits) for word in all_responses])
        model_adjusted_dist = model_adjusted_dist / model_adjusted_dist.sum()

        # Find probability of the top word in the adjusted distribution & word humans chose
        model_prob_adjusted = model_adjusted_dist[model_adjusted_dist.argmax()]
        model_prob_human_prediction = model_adjusted_dist[human_dist.argmax()]

        # Grab the human and model top words
        model_prediction_adjusted = all_responses[model_adjusted_dist.argmax()]

        ###############################################
        #### Compare human and model distributions ####
        ###############################################

        # KL divergence between human (P) and model (Q) distributions
        # Measures how different P is from Q --> expected excess surprise
        # from using Q as a model when the actual distribution is P
        kl_divergence = special.kl_div(human_dist, model_adjusted_dist)
        kl_divergence[np.isinf(kl_divergence)] = 0
        kl_divergence = kl_divergence.sum().item()

        # Jensen-Shannon distance (or divergence) is a metric version of KL divergence
        # Measures the similarity between two distributions (a symmetric version of KL)
        jensenshannon_dist = distance.jensenshannon(human_dist, model_adjusted_dist)
        
        # Earth Movers Distance (EMD)
        # Measures the dissimilarity of two frequency distributions
        earthmovers_dist = stats.wasserstein_distance(human_dist, model_adjusted_dist)
        
        # Kolmogorov-Smirnov test for goodness-of-fit
        # Tests if two distributions significantly differ 
        ks_stat = stats.kstest(human_dist, model_adjusted_dist)

        ###############################################
        ######### Create dataframe and return #########
        ###############################################

        modality_word_index_results = {
            'model_name': model_name,
            'modality': modality,
            'word_index': word_index,
            'ground_truth': ground_truth,
            'entropy_group': entropy_group,
            'accuracy_group': accuracy_group,

            # Human information
            'human_top_word': human_prediction,
            'human_prob': human_prob,
            'human_predictability': human_predictability, 
            'human_log_odds_predictability': human_log_odds_predictability.astype(float),
            'human_entropy': human_entropy,

            # Model information
            'model_top_word': model_prediction,
            'model_prob': model_prob,
            'model_predictability': model_predictability,
            'model_log_odds_predictability': model_log_odds_predictability.astype(float),
            'model_entropy': model_entropy,

            # Model adjusted distribution information
            'model_prob_adjusted': model_prob_adjusted,
            'model_prob_human_prediction': model_prob_human_prediction,
            
            # Comparison of distributions 
            'kl_divergence': kl_divergence,
            'earthmovers_dist': earthmovers_dist,
            'jensenshannon_dist': jensenshannon_dist,
            'ks_stat': ks_stat[0],

            # Comparison of top prediction
            'human_model_pred_similarity': pred_similarity,
            'human_model_accuracy_diff': accuracy_difference,
        }
        
        # Create a dataframe to store the information
        df_comparison.loc[len(df_comparison)] = modality_word_index_results

    return df_comparison
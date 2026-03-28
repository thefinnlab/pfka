import os, sys
import pandas as pd
import numpy as np
import math, random
import json
import re
import subprocess
import librosa
from natsort import natsorted
import glob
import ast
from scipy import stats

from tqdm import tqdm

import string
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from collections import defaultdict

from praatio import textgrid as tgio

import spacy
import pandas as pd
from collections import defaultdict

# nltk.download('tagsets')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('averaged_perceptron_tagger_eng')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

############################################
##### Functions for cutting audio files ####
############################################

def cut_stimulus_segments(df_preproc, task, stim_fn, stim_out_dir, segment_indices, stim_type):
    """
    Cut audio segments based on a nested list of indices.

    :param df_preproc: DataFrame containing preprocessed data.
    :param task: Task identifier (used in naming output files).
    :param audio_fn: Path to the input audio file.
    :param audio_out_dir: Directory to save the output audio segments.
    :param segment_indices: Nested list of indices where each sublist contains [start_idx, end_idx].
    :return: List of output filenames and a DataFrame with segment information.
    """

    if stim_type == 'video':
        # out_cmds = f'-async 1'
        out_cmds = '-map_metadata -1'
        out_ftype = 'mp4'
    else:
        out_cmds = f'-ar 16000'
        out_ftype = 'wav'

    # Load the stimulus and find the length in time
    stim_length = librosa.get_duration(path=stim_fn)

    # Initialize DataFrame to store segment information
    df_segments = pd.DataFrame(columns=['filename', 'word_index', 'critical_word', 'checked', 'adjusted'])
    out_fns = []

    for i, (start_idx, end_idx) in enumerate(tqdm(segment_indices, desc="Cutting stimulus segments")):

        # Current word is one greater than the end index
        current_word_idx = end_idx

        # Calculate onset and duration based on the segment indices
        onset, _, duration = get_cut_times(df_preproc, start_idx, end_idx)

        # If it's the last trial, we go all the way to the end of the stimulus
        # There is also not a word index
        if (i == (len(segment_indices) - 1)):
            current_word_idx = None
            duration = stim_length - onset
        # Ensure the duration does not exceed the remaining audio length
        elif (onset + duration > stim_length):
            duration = stim_length - onset

        # Generate output filename
        out_fn = os.path.join(stim_out_dir, f'{task}_segment-{str(i+1).zfill(5)}.{out_ftype}')
        out_fns.append(out_fn)

        # Use ffmpeg to cut the audio segment
        cmd = f'ffmpeg -hide_banner -loglevel error -y -ss {onset} -t {duration} -i {stim_fn} {out_cmds} {out_fn}'
        subprocess.run(cmd, shell=True)

        # Store segment information in the DataFrame
        df_segments.loc[len(df_segments)] = {
            'filename': out_fn,
            'word_index': current_word_idx,
            'critical_word': df_preproc.loc[current_word_idx]['Word_Written'] if current_word_idx else None,
            'checked': 0,
            'adjusted': 0
        }

    return out_fns, df_segments

def get_cut_times(df_preproc, start_idx, end_idx, use_prev_offset=False):
    """
    Calculate the onset, offset, and duration for a segment.

    :param df_preproc: DataFrame containing preprocessed data.
    :param start_idx: Start index of the segment.
    :param end_idx: End index of the segment.
    :return: Onset, offset, and duration of the segment.
    """
    if start_idx == 0:
        onset = 0
    else:
        # We want to start from where the previous clip was trimmed to 
        if use_prev_offset:
            onset = df_preproc.loc[start_idx-1]['Offset']
        else:
            onset = df_preproc.loc[start_idx]['Onset']

    # TLB changing from Onset to Onset (instead of to offset)
    offset = df_preproc.loc[end_idx]['Onset']
    duration = offset - onset

    return onset, offset, duration

############################################
##### Functions for editing transcripts ####
############################################

def update_dataframe_from_praat(df, textgrid):
    
    df = df.copy()
    
    for idx in range(len(df)):
        
        word = textgrid.getTier('word').entries[idx]
        
        df.loc[idx, 'Onset'] = word.start
        df.loc[idx, 'Offset'] = word.end
        df.loc[idx, 'Duration'] = word.end - word.start
        
    return df

def dataframe_to_textgrid(df, audio_fn, tier_name='word'):
    """
    Take a filename and its associated transcription and fill in all the gaps
    """

    duration = librosa.get_duration(path=audio_fn)
    
    # with contextlib.closing(wave.open(audio_fn, 'r')) as f:
    #     frames = f.getnframes()
    #     rate = f.getframerate()
    #     duration = frames / float(rate)
    rearranged_words = []
    file_ons = 0

    rearranged_words = []

    for ix, word in df.iterrows():

        # if word['Case'] == 'success' or word['Case'] == 'assumed':
        word_ons = word['Onset']#, 3)
        word_off = word['Offset']#, 3)

        target = word['Word_Written']
        rearranged_words.append((word_ons, word_off, target))
        # else:
        #     # search forwards and backwards to find the previous and next word
        #     # use the end and start times to get word times 
        #     target = content['words'][ix]['Word_Written']
        #     prev_end, next_start = align_missing_word(content, ix)
        #     rearranged_words.append((prev_end, next_start, target))

    # adjust for overlap in times
    for ix, word_times in enumerate(rearranged_words):
        if ix != 0:
            prev_start, prev_end, prev_word = rearranged_words[ix-1]
            curr_start, curr_end, curr_word = word_times

            # if the current start time is before the previous end --> adjust
            # to be the previous end time
            if curr_start < prev_end:
                rearranged_words[ix] = (prev_end, curr_end, curr_word)
                curr_start, curr_end, curr_word = rearranged_words[ix]

            # if the current end time is after the current start time
            # set to be the next start time
            if curr_end < curr_start and (ix+1 != len(rearranged_words)):
                next_start, next_end, next_word = rearranged_words[ix+1]
                rearranged_words[ix] = (curr_start, next_start, curr_word)
                curr_start, curr_end, curr_word = rearranged_words[ix]

            # final catch is adding a tiny bit of padding to the end word to adjust
            if curr_end == curr_start:
                rearranged_words[ix] = (curr_start, curr_end+0.0001, curr_word)


    tg = tgio.Textgrid()
    tg.addTier(tgio.IntervalTier(tier_name, rearranged_words))
    return tg

def gentle_to_textgrid(alignment_fn, path):
    """
    Take a filename and its associated transcription and fill in all the gaps
    """
    with contextlib.closing(wave.open(path, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
    
    rearranged_words = []
    file_ons = 0
    
    # load the alignment file
    with open(alignment_fn, encoding='utf-8') as f:
        content = json.load(f)
    all_ons = content['words'][0]['start']
    
    for ix, word in enumerate(content['words']):
        # if the word was successfully aligned
        if word['case'] == 'success' or word['case'] == 'assumed':
            word_ons = np.round(word['start'], 3)
            word_off = np.round(word['end'], 3)
            target = word['alignedWord']
            rearranged_words.append((word_ons, word_off, target))
        else:
            # search forwards and backwards to find the previous and next word
            # use the end and start times to get word times 
            target = content['words'][ix]['word']
            prev_end, next_start = align_missing_word(content, ix)
            
            rearranged_words.append((prev_end, next_start, target))
    
    # adjust for overlap in times
    for ix, word_times in enumerate(rearranged_words):
        if ix != 0:
            prev_start, prev_end, prev_word = rearranged_words[ix-1]
            curr_start, curr_end, curr_word = word_times

            # if the current start time is before the previous end --> adjust
            # to be the previous end time
            if curr_start < prev_end:
                rearranged_words[ix] = (prev_end, curr_end, curr_word)
                curr_start, curr_end, curr_word = rearranged_words[ix]

            # if the current end time is after the current start time
            # set to be the next start time
            if curr_end < curr_start and (ix+1 != len(rearranged_words)):
                next_start, next_end, next_word = rearranged_words[ix+1]
                rearranged_words[ix] = (curr_start, next_start, curr_word)
                curr_start, curr_end, curr_word = rearranged_words[ix]

            # final catch is adding a tiny bit of padding to the end word to adjust
            if curr_end == curr_start:
                rearranged_words[ix] = (curr_start, curr_end+0.0001, curr_word)
    
    tg = tgio.Textgrid()
    tg.addTier(tgio.IntervalTier('word', rearranged_words))
    return content, tg

def gentle_fill_missing_words(alignment_fn):
    '''
    A simple way to fill missing aligned words
    '''
    
    # load the alignment file
    with open(alignment_fn, encoding='utf-8') as f:
        content = json.load(f)
        
    for ix, word in enumerate(content['words']):
        if word['case'] != 'success':
            prev_end, next_start = align_missing_word(content, ix)
            content['words'][ix].update({'start': prev_end, 'end': next_start, 'case': 'assumed'})
            
    return content

def align_missing_word(content, ix):
    '''
    Searches from a word in both directions and then distributes time evenly
    '''
    # keep track of how many are missing
    forward_ix = ix
    forward_missing = 0
    
    # search forward
    while True:
        # move one forward
        forward_ix += 1
        if content['words'][forward_ix]['case'] == 'success':
            next_start = np.round(content['words'][forward_ix]['start'], 3)
            break
        else:
            forward_missing += 1
    
    # keep track of how many are missing
    back_ix = ix
    back_missing = 0
    
    while True:
        # move one backwards
        back_ix -= 1
        
        if content['words'][back_ix]['case'] == 'success':
            prev_end = np.round(content['words'][back_ix]['end'], 3)
            break
        else:
            back_missing += 1
    
    # space evenly between the number of missing items
    total_missing = back_missing + forward_missing + 1 # add one to include current item
    x_vals = np.linspace(prev_end, next_start, total_missing + 2)[1:-1] # add 2 to pad the points on either side
    
    # if there is anything missing
    # normalize indices to 0
    missing_ixs = np.arange(ix-back_missing,ix+forward_missing+1)
    
    # index of the value in the interpolated array
    arr_ix = np.argwhere(ix == missing_ixs)
    
    # then extract value from that array and round
    next_start = x_vals[arr_ix].squeeze()
    next_start = np.round(next_start, 3)
    
    # have to adjust prev end to be the interpolated value
    if len(missing_ixs) > 1 and arr_ix:
        prev_end = x_vals[np.argwhere(ix == missing_ixs)-1].squeeze()
        prev_end = np.round(prev_end, 3)
    
    return prev_end, next_start


########################################################
##### Functions for selecting prediction candidates ####
########################################################

lemmatizer = WordNetLemmatizer()

# generate the explained tags --> we will use these to make more sense of the outputs
tags_explained = nltk.data.load('help/tagsets/upenn_tagset.pickle')
STOP_WORDS = stopwords.words('english')

STOP_UTTERANCES = [
    'yes', 'yeah', 'alright', 'no', 'nope', 'nah', 'well', 'like', 'eh', 'huh', 
    'mm', 'ick', 'ch', 'hm', 'oh', 'mhm', 'ah', 'um', 'uh', 'uh-huh', 'uh-oh', 
    'boom', 'bam', 'wha', 'ra', 'ba', 'bla', 'ugh', 'okay', 'hi', 'hey', 
    'hello', 'ya', 'us', 'really','sh', 'said', 'know',
]

# add alphanumeric characters
ALPHANUM = set(string.ascii_lowercase+string.ascii_uppercase+string.digits)
STOP_UTTERANCES += [e for e in string.printable if e in ALPHANUM]

STOP_WORDS.extend(STOP_UTTERANCES)

NAMED_ENTITIES = [
    'Red-Headed', 'German', 'Sean', 'Googled' #custom removal from odetostepfather
]

# POS tag mapping, format: {Treebank tag (1st letter only): Wordnet}
tagset_mapping = defaultdict(
    lambda: 'n',   # defaults to noun
    {
        'N': 'n',  # noun types
        'P': 'n',  # pronoun types, predeterminers
        'V': 'v',  # verb types
        'J': 'a',  # adjective types
        'D': 'a',  # determiner
        'R': 'r'   # adverb types
    })


def create_word_prediction_df(align_fn, fill_missing_times=False):
    '''

    Preprocessing to get a dataframe for next word prediction. Applies the following steps:
        1. Loads the file
        2. Applies part of speech tagging to each word (used for lemmatization)
        3. Lemmatizes each word and evaluates whether a stop word
        4. If the word was aligned, adds the times of onset and offset
        5. Lastly interpolates onset times to recover those of any missing words
    '''
    
    # load the alignment file
    if fill_missing_times:
        data = gentle_fill_missing_words(align_fn)
    else:   
        with open(align_fn, encoding='utf-8') as f:
            data = json.load(f)
    
    # grab the original transcript
    transcript = data['transcript']
    words_list = data['words']

    # go and extract each word --> pos tagging here incorporates context
    all_words = [word['word'] for word in words_list]
    _, pos_tags = map(list,zip(*pos_tag(all_words)))

    # go through each word
    df_stack = []

    for i, current_word in enumerate(words_list):
        
        # first tokenize each word --> this makes it easier to 
        tokens = word_tokenize(current_word['word'])
        tag = tagset_mapping[pos_tags[i][0]]

        # as some words may be broken into multiple tokens, we need to lemmatize all tokens
        # then make sure we don't have any stopwords as part of the tokens
        lemmas = [lemmatizer.lemmatize(re.sub("[^a-zA-Z\s-]+", '', token.lower()), pos=tag) for token in tokens]
        stop_word = any([lemma in STOP_WORDS for lemma in lemmas if lemma]) # evaluate if not empty string
        is_digit = any([token.isdigit() for token in tokens])
        
        # we'll start the word dictionary here, but only add the times if we have the aligned times
        word_dict = {
            'Word_Written': current_word['word'],
            'Case': current_word['case'],
            'POS': pos_tags[i], # extract pos_tag for the word
            'POS_Definition': tags_explained[pos_tags[i]][0], # get the explained tag
            'Punctuation': transcript[current_word['endOffset']:words_list[i+1]['startOffset']]
                                      if i+1 < len(words_list) else transcript[current_word['endOffset']:], # punctuation following the word (use subsequent word)
            'Stop_Word': stop_word, # true or false if a stopword
            'Digit': is_digit,
        }

        # make sure that we've aligned the word --> could also check its a word in the vocabulary
        if fill_missing_times or 'alignedWord' in current_word:
            aligned_dict = {
                'Word_Vocab': current_word['word'] if fill_missing_times else current_word['alignedWord'],
                'Onset': current_word['start'],
                'Offset': current_word['end'],
                'Duration': current_word['end'] - current_word['start'], # calculate duration of the current word
            }
            word_dict.update(aligned_dict)
        
        df_stack.append(
            pd.DataFrame(
                word_dict,
                index=[i],
            )
        )
    
    df_stack = pd.concat(df_stack)
    
    # if interpolate_missing_times:
    #   df_stack['Onset'].interpolate()
    
    return df_stack

def clean_hyphenated_words(df):
    
    print ("\nHYPHEN CLEANING\n You will see a hyphenated word. Enter \'y' if the word is meant to be hyphenated or \'n' if not.\n")
    
    hyphenated = df['Punctuation'].str.contains('-')
    hyphenated_idxs = np.where(hyphenated)[0]
    
    for idx in hyphenated_idxs:
        # grab the current row and following row from the dataframe
        df_rows = df.iloc[idx:idx+2]
        hyphenated_word = '-'.join(df_rows['Word_Written'])
        
        # establish some context for the word
        precontext = ' '.join(df.iloc[idx-10:idx]['Word_Written']).encode('latin-1', 'ignore')  
        postcontext = ' '.join(df.iloc[idx+2:idx+10]['Word_Written']).encode('latin-1', 'ignore')  
        print (f'\nContext: {precontext} ___ {postcontext}')
        print (f'Word: {hyphenated_word.encode("latin-1", "ignore")}')

        response = input()
    
        if response == 'y':
            
            hyphenated_entry = {
                'Word_Written': hyphenated_word,
                'Case': df_rows['POS'].to_list()[-1],
                'POS': df_rows['POS'].to_list()[-1],
                'POS_Definition': df_rows['POS_Definition'].to_list()[-1],
                'Punctuation': df_rows['Punctuation'].to_list()[-1] ,
                'Stop_Word': hyphenated_word.lower() in STOP_WORDS,
                'Digit': any(df_rows['Digit'].to_list()),
                'Word_Vocab': hyphenated_word,
                'Onset': df_rows['Onset'].to_list()[0],
                'Offset': df_rows['Offset'].to_list()[-1],
                'Duration': df_rows['Offset'].to_list()[-1] - df_rows['Onset'].to_list()[0]
            }

            df.loc[idx, :] = pd.Series(hyphenated_entry)
            df = df.drop(idx+1).reset_index(drop=True)

            # we've dropped an index
            hyphenated_idxs -= 1 
            
            print (f'Word updated to: {hyphenated_word.encode("latin-1", "ignore")}')
        else:
            # otherwise add padding on each side to ensure it's not hyphenated
            df.at[idx, 'Punctuation'] =  ' - '
            
            hyphenated_word = ' - '.join(df_rows['Word_Written'])
            print (f'Words separated to: {hyphenated_word.encode("latin-1", "ignore")}')
      
    df = df.reset_index(drop=True)
    
    return df

def clean_named_entities(df):
    '''
    Label the named entities in the transcript to avoid selecting them as candidates.
    '''
    
    print ("\nNAMED ENTITY CLEANING\nYou will see a potential named entity (e.g., person, place). Enter \'y' if the word is or refers to a named entity and \'n' otherwise.\n")
    
    # fix instructions?
    named_entities = pd.Series(df['POS'] == 'NNP') & pd.Series(df['Stop_Word'] == False)
    named_entity_idxs = np.where(named_entities)[0]

    # add custom indices
    custom_entity_idxs = np.where(df['Word_Written'].isin(NAMED_ENTITIES))[0]
    named_entity_idxs = np.unique(np.concatenate([named_entity_idxs, custom_entity_idxs]))

    df['Named_Entity'] = False

    for idx in named_entity_idxs:
        # grab the current row and following row from the dataframe
        df_rows = df.iloc[idx]
        ne_word = df_rows['Word_Written']

        precontext = ' '.join(df.iloc[idx-10:idx]['Word_Written']).encode('latin-1', 'ignore') 
        postcontext = ' '.join(df.iloc[idx+1:idx+10]['Word_Written']).encode('latin-1', 'ignore') 

        # precontext = re.sub(u"(\u2018|\u2019)", "'", precontext)
        # postcontext = re.sub(u"(\u2018|\u2019)", "'", postcontext)
        print (f'\nContext: {precontext} ___ {postcontext}')
        print (f'Word: {ne_word.encode("latin-1", "ignore")}')
        
        response = input()
    
        if response == 'y':
            df.at[idx, 'Named_Entity'] = True

    df = df.reset_index(drop=True)
    return df

def add_syntax_info(df, text_column="Word_Written", punctuation_column="Punctuation"):
    """
    Adds POS, dependency, sentence, clause, and position-in-clause info
    to the dataframe, aligned one-to-one with its words.
    """

    nlp = spacy.load("en_core_web_sm", disable=["ner"])  # we need syntax only

    text = (df[text_column] + df[punctuation_column]).tolist()
    # text = df[text_column].tolist()
    text = " ".join(text)    
    doc = nlp(text)

    spacy_tokens = [t.text for t in doc]

    # --- Clean words: strip leading punctuation ---
    def clean_word(word):
        return word.lstrip(string.punctuation)

    df_words = [clean_word(w) for w in df[text_column].tolist()]

    # --- Greedy alignment (merges spaCy tokens back into transcript words) ---
    aligned = []
    i, j = 0, 0
    
    while i < len(df_words) and j < len(spacy_tokens):

        spacy_tokens[j] = spacy_tokens[j].lstrip()

        # Remove trailing punctuation ('e.g., St.')
        spacy_tokens[j] = spacy_tokens[j].rstrip(string.punctuation)

        if spacy_tokens[j] in string.punctuation or spacy_tokens[j] in string.whitespace or spacy_tokens[j] in ['...', '…', '—']:
            j += 1
            continue

        if df_words[i] == spacy_tokens[j]:
            aligned.append(doc[j])
            i += 1
            j += 1
        else:
            buffer = spacy_tokens[j]
            k = j + 1
            while k < len(spacy_tokens) and buffer != df_words[i]:
                buffer += spacy_tokens[k]
                k += 1
            if buffer == df_words[i]:
                aligned.append(doc[j])  # first token as representative
                j = k
                i += 1
            else:
                print (f"Could not align word {df_words[i]}")
                print (f"Buffer: {buffer}")
                print (f"Spacy tokens: {spacy_tokens[j]}")
                print (f"df_words: {df_words[i]}")
                print (f"doc: {doc[j]}")
                print (f"doc[j].text: {doc[j].text}")
                print (f"df_words[i]: {df_words[i]}")
                raise ValueError(f"Could not align word {df_words[i]}")

    if len(aligned) != len(df):
        raise ValueError(f"Alignment failed: {len(aligned)} vs {len(df)}")

    # --- Step 1: assign consecutive clause IDs using a counter ---
    clause_ids = {}
    clause_counter = 1
    tok_to_clause_head = {}

    for tok in doc:
        head = tok
        while True:
            # Clause heads: non-aux verbs or ROOT AUX/VERB
            if (head.pos_ == "VERB" and head.dep_ != "aux") or (head.dep_ == "ROOT" and head.pos_ in {"AUX", "VERB"}):
                main_head = head
                while main_head.dep_ == "conj" and main_head in tok_to_clause_head:
                    main_head = main_head.head
                if main_head not in tok_to_clause_head:
                    tok_to_clause_head[main_head] = clause_counter
                    clause_counter += 1
                clause_ids[tok] = tok_to_clause_head[main_head]
                break
            if head.head == head:  # reached root
                if head not in tok_to_clause_head:
                    tok_to_clause_head[head] = clause_counter
                    clause_counter += 1
                clause_ids[tok] = tok_to_clause_head[head]
                break
            head = head.head

    # --- Step 2: assign consecutive Clause_Pos using local counters ---
    clause_pos_counter = defaultdict(int)
    clause_positions = {}

    for tok in aligned:  # iterate in sentence order
        cid = clause_ids[tok]
        clause_pos_counter[cid] += 1
        clause_positions[tok] = clause_pos_counter[cid]

    # --- Step 3: extract syntactic info ---
    syntax_info = []
    for tok in aligned:
        syntax_info.append({
            "POS_spaCy": tok.pos_,
            "Dep": tok.dep_,
            "Head": tok.head.text,
            "Clause_ID": clause_ids[tok],
            "Clause_Pos": clause_positions[tok],
        })

    return pd.concat([df.reset_index(drop=True), pd.DataFrame(syntax_info)], axis=1)

############# Frequency stats #############

def get_word_frequency(df_preproc, columns=['stim_name', 'order', 'feature', 'value']):

    from pliers.stimuli import TextStim
    from pliers.extractors import PredefinedDictionaryExtractor, merge_results

    extractor = PredefinedDictionaryExtractor(['subtlexusfrequency/Lg10WF'],  missing=np.nan)
    word_list = df_preproc['Word_Written']
    
    ### Get frequency info
    stims = [TextStim(text=word.lower(), order=i) for i, word in enumerate(word_list)]
    df_results = extractor.transform(stims)
    df_results = merge_results(df_results, extractor_names='column', format='long')

    # trim down to the columns to keep
    df_results = df_results[columns]
    df_results['stim_name'] = df_results['stim_name'].str.extract(r'\[(.*?)\]')
    
    # now add that info to the main dataframe
    idxs = df_results['order']
    df_preproc.loc[idxs, 'Lg10WF'] = df_results['value'].tolist()

    return df_preproc


def match_df_distributions(source_df, target_df, source_col, target_col, alpha=0.05, n_iter=10):
    """
    Match the distribution of source_data to target_dist without replacement.
    
    Args:
    source_df: DataFrame containing source data
    target_df: DataFrame containing target data
    source_col: Column name in source_df to use for matching
    target_col: Column name in target_df to use for matching
    alpha: Significance level for the t-test
    n_samples: Minimum number of samples required
    n_iter: Number of iterations to perform distribution fitting
    
    Returns:
    matched_df: DataFrame containing matched samples and all original columns from source_df
    t_statistic: Final t-statistic
    p_value: Final p-value
    """
    
    source_data = source_df[source_col].to_numpy()
    target_data = target_df[target_col].to_numpy()
    
    # Fit the target data to a normal distribution
    mu, std = stats.norm.fit(target_data)
    target_dist = stats.norm(loc=mu, scale=std)
    
    best_indices = []
    best_stat = np.inf
    
    for i in range(n_iter):
        np.random.seed(i)
        
        source_indices = np.arange(len(source_data))
        
        np.random.shuffle(source_indices)
        
        matched_indices = []
        
        while len(source_indices) > 0:
            next_index = source_indices[0]
            next_sample = source_data[next_index]
            
            # Calculate acceptance probability
            target_pdf = target_dist.pdf(next_sample)
            max_source_pdf = np.max(target_dist.pdf(source_data[source_indices]))
            accept_prob = min(1, target_pdf / max_source_pdf)
            
            if np.random.random() < accept_prob:
                matched_indices.append(next_index)
            
            source_indices = source_indices[1:]
        
        matched_samples = source_data[matched_indices]
        t_stat, p_val = stats.ttest_ind(matched_samples, target_data)
        
        # minimize the tstat while increasing the number of items
        if (p_val > alpha) and (abs(t_stat) < abs(best_stat)) and (len(matched_indices) > len(best_indices)):
            best_indices = matched_indices.copy()
            best_stat = t_stat
            print(f'Updating distribution -- retained {(len(best_indices)/len(source_data)) * 100:.2f}% of samples')
    
        print(f'Completed iter {str(i+1).zfill(3)}')
    
    # Final t-test
    best_samples = source_data[best_indices]
    t_stat, p_val = stats.ttest_ind(best_samples, target_data)
    
    # Create the output dataframe
    matched_df = source_df.iloc[best_indices].copy().sort_index()
    
    return matched_df, t_stat, p_val

### FUNCTIONS FOR STRATIFYING WORDS BASED ON A MODEL #####

def load_model_results(model_dir, model_name, task, window_size, top_n):
    '''
    Loads model data from directory
    '''

   # Define a safe evaluation function
    def safe_eval(x):
        if pd.isna(x):
            return []  # or return None, depending on your needs
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return []  # or return None, depending on your needs

    if 'whisper' in model_name:
        model_dir = os.path.join(model_dir, task, 'careful-whisper', model_name, f'window-size-{str(window_size).zfill(5)}')
    else:
        model_dir = os.path.join(model_dir, task, model_name, f'window-size-{str(window_size).zfill(5)}')

    results_fn = natsorted(glob.glob(os.path.join(model_dir, f'*top-{top_n}*')))[0]
    
    # load the data, remove nans
    df_model_results = pd.read_csv(results_fn)
    df_model_results['top_n_predictions'] = df_model_results['top_n_predictions'].astype(object)
    df_model_results.loc[1:, 'top_n_predictions'] = df_model_results.loc[1:, 'top_n_predictions'].apply(safe_eval)
    
    return df_model_results

def divide_nwp_dataframe(df, accuracy_type, percentile, drop=True):

    df_divide = df.copy()

    # first find the lowest and highest percentile for entropy
    low_entropy_idxs = df['entropy'] < np.nanpercentile(df['entropy'], percentile)
    high_entropy_idxs = df['entropy'] >= np.nanpercentile(df['entropy'], 100-percentile)

    ## set names for entropy group
    df_divide.loc[low_entropy_idxs, 'entropy_group'] = 'low'
    df_divide.loc[high_entropy_idxs, 'entropy_group'] = 'high'

    # repeat for continuous accuracy
    low_accuracy_idxs = df[accuracy_type] < np.nanpercentile(df[accuracy_type], percentile)
    high_accuracy_idxs = df[accuracy_type] >= np.nanpercentile(df[accuracy_type], 100-percentile)

    ## set names for accuracy group
    df_divide.loc[low_accuracy_idxs, 'accuracy_group'] = 'low'
    df_divide.loc[high_accuracy_idxs, 'accuracy_group'] = 'high'

    # TLB --> i think this was commented out to try to 
    # plot word2vec results
    if drop:
        return df_divide.dropna()
    else:
        return df_divide

def get_quadrant_distributions(df_divide, indices):
    '''
    Given a set of indices, returns the distributions of words
    in entropy/accuracy quadrants
    '''
    
    df_idx = df_divide.loc[indices]
    
    # get the items as a dictionary for passing out to aggregate
    quadrant_dist = {f'{labels[0]}-entropy_{labels[1]}-accuracy': round(len(df)/len(df_idx), 2) 
                 for labels, df in df_idx.groupby(['entropy_group', 'accuracy_group'])}

    df_quadrants = pd.DataFrame.from_dict(quadrant_dist, orient='index').T
    
    return df_quadrants

def select_prediction_words(df_divide, remove_perc, select_perc, min_spacing_thresh=3):
    '''
    
    df_divide: candidate words divided into quartiles based on entropy and accuracy
    
    remove_perc: percentage of words to remove based on proximity to other words
        helps ensure decent spacing between presented words
        
    select_perc: percentage of words to select for presentation    
    
    '''
    
    # calculate spacing between each word and the subsequent words
    df_divide['spacing'] = np.hstack([np.nan, np.diff(df_divide.index)])
    quadrant_distributions = get_quadrant_distributions(df_divide, df_divide.index).to_numpy()
    
    updated = []

    for i, df in df_divide.groupby(['entropy_group', 'accuracy_group']):
        # find how many words to remove in the quadrant based on the percent
        n_words = round(remove_perc * len(df))
        df = df.sort_values(by='spacing').iloc[n_words:]
        updated.append(df.sort_index())

    # ensure that quadrant distributions remain the same after removing words
    updated = pd.concat(updated).sort_index()
    updated_distributions = get_quadrant_distributions(updated, updated.index).to_numpy()

    print (quadrant_distributions)
    print (updated_distributions)

    assert (np.allclose(quadrant_distributions, updated_distributions, atol=0.01))
    
    # make sure it is scaled to the original dataframe
    select_perc = select_perc/(1-remove_perc)
    min_spacing = 0
    RANDOM_STATE = 0
    
    print (f'Selecting {select_perc*100:.2f}% of remaining items')
    
    # now we pseudo-randomly sample meeting the constraint that the word with minimum spacing
    # must be greater or equal to the spacing threshold
    while (min_spacing < min_spacing_thresh):
        # now sample the words from each quadrant
        sampled = []

        print (f'Tried random state: {RANDOM_STATE}')

        for i, df in updated.groupby(['entropy_group', 'accuracy_group']):

            df_sampled = df.sample(frac=select_perc, random_state=RANDOM_STATE).sort_index()
            sampled.append((len(df_sampled), df_sampled))

        n_sampled, sampled = zip(*sampled)
        sampled = pd.concat(sampled).sort_index()

        min_spacing = np.diff(sampled.index).min()
        
        RANDOM_STATE += 1
    
    print (f'Min spacing of {min_spacing}')
    print (f'{len(sampled)} total words')

    return sampled

def random_chunks(lst, n, shuffle=False):
    """Created randomized n-sized chunks from lst."""
    
    tmp_lst = lst.copy()
    n_total = len(lst)
    
    if shuffle:
        random.shuffle(tmp_lst)
    
    all_chunks = []
    
    for i in range(0, len(tmp_lst), n):
        all_chunks.append(tmp_lst[i:i + n])
    
    # distribute remaining items across orders
    if len(all_chunks) != n_total//n:
        remainder = all_chunks.pop()
        
        for i, item in enumerate(remainder):      
            all_chunks[i%n].append(item)
    
    # lastly sort for ordered indices
    all_chunks = [sorted(chunk) for chunk in all_chunks]
    
    return all_chunks

### FUNCTIONS FOR RANDOMIZATION OF ORDERS #####
### modified from https://stackoverflow.com/questions/93353/create-many-constrained-random-permutation-of-a-list

def get_pool(items, n_elements_per_subject, use_each_times):
    pool = {}
    for n in items:
        pool[n] = use_each_times
    
    return pool

def rebalance(ret, pool, n_elements_per_subject):
    max_item = None
    max_times = None
    
    for item, times in pool.items():
        if max_times is None:
            max_item = item
            max_times = times
        elif times > max_times:
            max_item = item
            max_times = times
    
    next_item, times = max_item, max_times

    candidates = []
    for i in range(len(ret)):
        item = ret[i]

        if next_item not in item:
            candidates.append( (item, i) )
    
    swap, swap_index = random.choice(candidates)

    swapi = []
    for i in range(len(swap)):
        if swap[i] not in pool:
            swapi.append( (swap[i], i) )
    
    which, i = random.choice(swapi)
    
    pool[next_item] -= 1
    pool[swap[i]] = 1
    swap[i] = next_item

    ret[swap_index] = swap

def create_balanced_orders(items, n_elements_per_subject, use_each_times, consecutive_limit=2,  error=1):
    '''
    Returns a set of unique lists under the constraints of 
    - n_elements_per_subject (must be less than items)
    - use_each_times: number of times each item should be seen across subjects

    Together these define the number of subjects

    '''

    n_subjects = math.ceil((use_each_times * len(items)) / n_elements_per_subject)

    print (f'Creating orders for {n_subjects} subjects')

    pool = get_pool(items, n_elements_per_subject, use_each_times)
    
    ret = []
    while len(pool.keys()) > 0:
        while len(pool.keys()) < n_elements_per_subject:
            rebalance(ret, pool, n_elements_per_subject)
        
        selections = sorted(random.sample(pool.keys(), n_elements_per_subject))
        
        for i in selections:
            pool[i] -= 1
            if pool[i] == 0:
                del pool[i]

        ret.append( selections )
        
        unique, counts = np.unique(ret, return_counts=True)
        
        if all(np.logical_and(counts <= use_each_times + error, counts >= use_each_times)):
               break
    return ret

def consecutive(data, stepsize=1):
    '''
    Split data into sets where the spacing between consecutive numbers is larger 
    than the stepsize. A given set will contain one or more items. 
    
    In the case that the set has more than one item, these items are separated
    by less than the step size.
    '''
    return np.split(data, np.where(np.diff(data) > stepsize)[0]+1)

def get_consecutive_list_idxs(orders, consecutive_spacing):
    '''
    Given a list of arrays, where each array contains numbers, find
    lists that contain consecutive items within consecutive_spacing difference.
    
    Returns a list of indices corresponding to which orders have violations of 
    the consecutive constraint
    '''
    
    consecutive_item_lists = []
    
    for order in orders:
        consecutive_items = consecutive(order, consecutive_spacing)
        contains_consecutive_items = any([len(item) for item in consecutive_items if len(item) > 1])
        consecutive_item_lists.append(contains_consecutive_items)
        
    return np.where(consecutive_item_lists)[0]

def check_consecutive_spacing(arr, consecutive_spacing, item=None):
    if item:
        return all(abs(item - arr) > consecutive_spacing)
    else:
        return all(np.diff(arr) > consecutive_spacing)

def get_swap_choices(all_lists, current_list_idx, swap_item, consecutive_spacing):
    
    # get indices of all lists
    all_list_idxs = np.arange(len(all_lists))
    
    # sample a random list that is not the current list
    random_list_options = np.setdiff1d(all_list_idxs, current_list_idx)
    
    # then pull the current list out
    current_list = np.asarray(all_lists[current_list_idx])
    swap_choices = []    
    
    while not len(swap_choices):
        random_list_idx = random.choice(random_list_options)
        random_list = np.asarray(all_lists[random_list_idx])
    
        # Find choices of items not within our current list
        swap_choices = np.setdiff1d(random_list, current_list)
    
    # then select an item to swap for
    swap_choice = random.choice(swap_choices)
    
    return random_list, random_list_idx, swap_choice

def sort_consecutive_constraint(orders, consecutive_spacing=2, pass_threshold=None):
    '''
    Make sure all indices are separated by at least consecutive_spacing items
    '''

    # Find lists with consecutive items violating our constraint
    consecutive_order_idxs = get_consecutive_list_idxs(orders, consecutive_spacing)

    passes = 1

    while len(consecutive_order_idxs):
        
        print (f'Starting pass #{passes}')
        
        # go through each list that contains a violation
        for order_idx in consecutive_order_idxs:
            
            # Select the current list violating the constraint
            current_list = np.asarray(orders[order_idx])

            # Find all sets of consecutive items in the current list --> find their lengths
            consecutive_items = consecutive(current_list, consecutive_spacing)
            consecutive_spacings = np.asarray(list(map(len, consecutive_items)))

            # Find sets of slices that violate the constraint --> this would be any list that has a length 
            # greater than 1
            violations = np.where(consecutive_spacings > 1)[0]
            
            for violation in violations:
                # Select items that need to be swapped --> these will be swapped into a randomly selected list
                swap_items = consecutive_items[violation][1::]

                for item in swap_items:
                    swap_idx = np.where(current_list == item)[0]

                    random_list, random_list_idx, choice = get_swap_choices(orders, order_idx, item, consecutive_spacing)
        
                    # Make sure we didn't violate our constraint again with either list
                    while (
                        np.isin(choice,current_list) or 
                        np.isin(item,random_list) or
                        not check_consecutive_spacing(item=choice, arr=current_list, consecutive_spacing=consecutive_spacing)
                    ):
                        random_list, random_list_idx, choice = get_swap_choices(orders, order_idx, item, consecutive_spacing)
                    
                    # Find the index to swap to
                    choice_idx = np.where(random_list == choice)[0]

                    # Swap the two items
                    current_list[swap_idx] = choice
                    random_list[choice_idx] = item

                    # Set them in the overall orders
                    orders[order_idx] = sorted(current_list)
                    orders[random_list_idx] = sorted(random_list)

        # Find lists with consecutive items violating our constraint
        consecutive_order_idxs = get_consecutive_list_idxs(orders, consecutive_spacing)
        print (f'Number of lists w/ violation: {len(consecutive_order_idxs)}')
        passes += 1

        if pass_threshold and (passes > pass_threshold):
            break

    passed_constraint = not any(consecutive_order_idxs)

    return passed_constraint, orders


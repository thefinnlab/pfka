import os, sys, glob
import json
import re
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from itertools import product
from natsort import natsorted
from scipy.spatial.distance import cdist

sys.path.append('../../utils/')

from config import *
from tommy_utils.nlp import CLM_MODELS_DICT

STIM_DIR = os.path.join(BASE_DIR, 'stimuli')

def load_model_data(model_dir, model_name, task, window_size, top_n):
	'''
	Loads model data from directory
	'''
	
	model_dir = os.path.join(model_dir, task, model_name, f'window-size-{str(window_size).zfill(5)}')
	results_fn = natsorted(glob.glob(os.path.join(model_dir, f'*top-{top_n}*')))[0]
	
	# load the data, remove nans
	model_results = pd.read_csv(results_fn)
	model_results['glove_avg_accuracy'] = np.nan_to_num(model_results['glove_avg_accuracy']) #.apply(np.nan_to_num)
	model_results['word2vec_avg_accuracy'] = np.nan_to_num(model_results['word2vec_avg_accuracy']) #.apply(np.nan_to_num)
	model_results['fasttext_avg_accuracy'] = np.nan_to_num(model_results['fasttext_avg_accuracy']) #.apply(np.nan_to_num)
	
	return model_results
	
def get_stim_candidate_idxs(task):
	'''
	Find the NWP candidate indices of a preprocessed transcript
	'''
	
	preproc_fn = os.path.join(STIM_DIR, 'preprocessed', task, f'{task}_transcript-preprocessed.csv')
	df_preproc = pd.read_csv(preproc_fn)
	nwp_idxs = np.where(df_preproc['NWP_Candidate'])[0]
	
	return nwp_idxs

def plot_accuracy_by_group(df_group, x, y, hue, iterate_col=None, order=None, hue_order=None, figsize=5):
	
	# group by the iterate column
	if not iterate_col:
		df_group['temp'] = 1
		iterate_col = 'temp'
	
	df_grouped = df_group.groupby(iterate_col)
	n_cols = len(df_grouped)
	
	# make the plot to plot over
	fig, axes = plt.subplots(1, n_cols, figsize=(figsize*n_cols, figsize))
	axes = np.array(axes).flatten()
	
	# now go through each grouped dataframe and plot
	for ax, (i, df) in zip(axes, df_grouped):
		ax = sns.boxplot(data=df, x=x, y=y, hue=hue, ax=ax, order=order, hue_order=hue_order, palette='Paired')
		if iterate_col != 'temp':
			ax.title.set_text(f'{iterate_col}: {i}')
	
	return fig, axes

def get_model_pairwise_similarity(all_results, result_type='binary_accuracy', metric='dice'):
	
	# for each task, examine model correlation matrix with each other
	df_results = []

	for i, ((task, window_size, top_n), df) in enumerate(all_results.groupby(['task', 'window_size', 'top_n'])):
		
		model_result = df.pivot(columns=['model_name'], values=[result_type]).droplevel(level=0, axis=1)
		model_names = model_result.columns
		
		# calculate dice coef
		similarity = 1 - cdist(model_result.T, model_result.T, metric=metric)

		pair_names = [f'pair{str(i).zfill(2)}-{pair[0]}-{pair[1]}' for i, pair in enumerate(product(model_names, model_names))]

		df = pd.DataFrame({
			'model_pair': pair_names,
			'task': task,
			'window_size': window_size,
			'top_n': top_n,
			metric: similarity.flatten()
		})

		df_results.append(df)
		
	df_results = pd.concat(df_results)
	
	avg_result = df_results.groupby(['model_pair', 'window_size', 'top_n'], as_index=False) \
		.agg({metric: np.nanmean}).reset_index(drop=True) 
	
	return avg_result

def plot_model_similarity(all_results, groupvar, result_type, metric, filtervar=None):
	
	model_names = all_results['model_name'].unique().tolist()
	n_models = len(model_names)
	
	avg_results = get_model_pairwise_similarity(all_results, result_type=result_type, metric=metric)
	
	# chop down to the window size
	if filtervar:
		avg_results = avg_results[avg_results[filtervar[0]] == filtervar[1]]

	if metric == 'dice':
		vmin = 0
		vmax = 1
		center = 0
	elif metric == 'correlation':
		vmin = -1
		vmax = 1
		center = 0
	
	fig, axes = plt.subplots(1, 3, figsize=(21, 7))
	axes = axes.flatten()

	for ax, (group, df) in zip(axes, avg_results.groupby(groupvar)):
		
		# TLB --> SHOULD ENSURE ZTRANSFORM BEFORE AVERAGE
		df = df.groupby('model_pair').agg({metric: 'mean'})
		
		similarity_matrix = np.reshape(df[metric].to_numpy(), (n_models,n_models))
		
		sns.heatmap(similarity_matrix, annot=True, square=True, xticklabels=model_names, 
						 yticklabels=model_names, vmin=vmin, vmax=vmax, center=center,
						 cmap='RdBu_r', cbar_kws={"shrink": 0.6}, ax=ax)
		
		ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha='right')
		ax.title.set_text(f'{groupvar}: {group}')
		
	return fig, axes

def divide_nwp_dataframe(df, accuracy_type, percentile):
	
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
	
	return df_divide.dropna()

def plot_quadrant_distributions(model_results, accuracy_type, percentile):
	
	# get xmedian and ymedian --> needs to happen before otherwise plot is off
	x_median = np.nanmedian(model_results[accuracy_type])
	y_median = np.nanmedian(model_results['entropy'])
	
	xmin, xmax = model_results[accuracy_type].max(), model_results[accuracy_type].min()
	ymin, ymax = model_results['entropy'].max(), model_results['entropy'].min()
	
	# divide the data into quadrants based on percentile
	# we use a form of continuous accuracy and entropy
	df_divide = divide_nwp_dataframe(model_results, accuracy_type=accuracy_type, percentile=percentile)
	
	fig, axes = plt.subplots(1, 2, figsize=(13,5))
	axes = axes.flatten()
	
	sns.scatterplot(data=df_divide, x=accuracy_type, y='entropy', hue='binary_accuracy', 
					 color='.6', palette="BuPu", alpha=0.75, ax=axes[0])
	
	# turn off top and right axes
	axes[0].spines["top"].set_visible(False)
	axes[0].spines["right"].set_visible(False)
	
	axes[0].vlines(x=x_median, ymin=ymin, ymax=ymax, linestyles='dashed', color='k')
	axes[0].hlines(y=y_median, xmin=xmin, xmax=xmax, linestyles='dashed', color='k')
	
	axes[0].title.set_text('Division of NWP candidates by median entropy/accuracy')
	
	### now plot number of words in each quadrant
	x_len = np.unique(df_divide['entropy_group']).shape[0]
	y_len = np.unique(df_divide['accuracy_group']).shape[0]
	
	# get the items as a dictionary for passing out to aggregate
	quadrant_dist = {f'{labels[0]}-entropy_{labels[1]}-accuracy': round(len(df)/len(df_divide), 2) 
				 for labels, df in df_divide.groupby(['entropy_group', 'accuracy_group'])}
	
	df_quadrants = pd.DataFrame.from_dict(quadrant_dist, orient='index').T
	df_quadrants['task'] = task
	
	# grab all except text column and plot
	# fmt = "g" disables scientific notation
	quadrants_mat = df_quadrants.loc[:, df_quadrants.columns != 'task'].to_numpy().reshape((x_len, y_len))[:, ::-1]
	sns.heatmap(quadrants_mat, annot=True, fmt='g', ax=axes[1], cbar=False, square=True)
	
	# labels, title and ticks
	axes[1].set_xlabel('entropy_group')
	axes[1].set_ylabel('accuracy_group')
	
	axes[1].xaxis.set_ticklabels(['low', 'high'])
	axes[1].yaxis.set_ticklabels(['high', 'low'])
	
	axes[1].set_title(f'Quadrant distribution - {percentile} perc')
	
	return fig, axes, df_quadrants

if __name__ == "__main__":

	models_dir = os.path.join(BASE_DIR, 'derivatives/model-predictions/')
	plots_dir = os.path.join(BASE_DIR, 'derivatives/plots/model-diagnostics/')

	if not os.path.exists(plots_dir):
		os.makedirs(plots_dir)

	# get all names of models 
	# model_names = sorted(CLM_MODELS_DICT.keys())
	model_names = ['gpt2-xl']

	# set sizes of correct/context window to plot
	top_ns = [1,5]
	window_sizes = [25]
	accuracy_types = ['binary_accuracy', 'glove_avg_accuracy', 'word2vec_avg_accuracy', 'fasttext_avg_accuracy']

	# get names of stimuli
	tasks = ['black', 'howtodraw', 'wheretheressmoke']
	# tasks = [os.path.basename(d) for d in sorted(glob.glob(os.path.join(STIM_DIR, 'preprocessed', '*')))]
	# tasks = [task for task in tasks if task not in ['example_trial', 'nwp_practice_trial']]

	###### COMPLETE SUMMARY PLOT ########
	# Compare all models against each other

	df_scores = pd.DataFrame(columns=['model_name', 'task', 'top_n', 'window_size', 
		'avg_binary_accuracy', 'avg_glove_accuracy', 'avg_word2vec_accuracy', 'avg_fasttext_accuracy'])

	all_results = []

	for model_name, task, window_size, top_n in product(model_names, tasks, window_sizes, top_ns):
		print (f'Loading {model_name}, {task}, {window_size}, {top_n}')

		# load indices of next word candidates for the current task
		candidate_idxs = get_stim_candidate_idxs(task)

		# load model results --> filter to next word candidates
		model_results = load_model_data(models_dir, model_name=model_name, task=task, window_size=window_size, top_n=top_n)
		model_results = model_results.iloc[candidate_idxs]

		# add model data to the dataframe
		df_scores.loc[len(df_scores)] = {
			'model_name': model_name,
			'task': task,
			'window_size': window_size,
			'top_n': top_n,
			'avg_binary_accuracy': np.nanmean(model_results['binary_accuracy']),
			'avg_glove_accuracy': np.nanmean(model_results['glove_avg_accuracy']),
			'avg_word2vec_accuracy': np.nanmean(model_results['word2vec_avg_accuracy']),
			'avg_fasttext_accuracy': np.nanmean(model_results['fasttext_avg_accuracy'])
		}

		# also aggregate results from all models
		model_results[['model_name', 'task', 'top_n', 'window_size']] = [model_name, task, top_n, window_size]
		all_results.append(model_results)

	### Get order of models by binary accuracy
	grouped_accuracy = df_scores.loc[:,['model_name', 'avg_binary_accuracy']] \
		.groupby(['model_name']) \
		.median() \
		.sort_values(by='avg_binary_accuracy')

	# now melt the dataframe to make it easier to plot
	df_scores = df_scores.melt(id_vars=['model_name', 'task', 'top_n', 'window_size'], 
						   var_name='accuracy_type', value_name='accuracy')

	# get accuracy min/max for all plots (across types of accuracy)
	accuracy_max = df_scores['accuracy'].to_numpy().max() * 1.2
	accuracy_min = df_scores['accuracy'].to_numpy().min() * 1.2

	if accuracy_min == 0:
		accuracy_min = -0.05

	# #################################
	# ### Plot average model similarity 
	# #################################

	# out_dir = os.path.join(plots_dir, 'model-similarity')

	# if not os.path.exists(out_dir):
	# 	os.makedirs(out_dir)

	# # assemble all results and get pairwise binary accuracy
	# all_results = pd.concat(all_results)

	# ## start by running accuracy metrics
	# result_metrics = [
	# 	('binary_accuracy', 'dice'),
	# 	('glove_avg_accuracy', 'correlation'),
	# 	('word2vec_avg_accuracy', 'correlation'),
	# 	('fasttext_avg_accuracy', 'correlation'),
	# ]

	# for (result_type, metric), window_size in product(result_metrics, window_sizes):

	# 	fig, axes = plot_model_similarity(all_results, groupvar='top_n', filtervar=['window_size', window_size], 
	# 		  result_type=result_type, metric=metric)
	
	# 	plt.suptitle(f'Model {result_type} similarity ({metric} coef) - window size {window_size}', y=0.9)
	# 	plt.tight_layout()

	# 	out_fn = os.path.join(out_dir, f'model-{result_type}-similarity_window-size-{window_size}.jpg')
	# 	plt.savefig(out_fn, dpi=300)
	# 	plt.close('all')

	# # then plot entropy by window size -- top_n doesn't matter here since distribution is the same
	# # across the top prediction sampling
	# fig, axes = plot_model_similarity(all_results, groupvar='window_size',
	# 	result_type='entropy', metric='correlation')

	# plt.suptitle(f'Model entropy similarity ({metric} coef)', y=0.9)
	# plt.tight_layout()

	# out_fn = os.path.join(out_dir, f'model-entropy-similarity.jpg')
	# plt.savefig(out_fn, dpi=300)
	# plt.close('all')

	# ############################
	# ### Plot overall performance
	# ############################

	# fig, axes = plot_accuracy_by_group(df_scores, iterate_col='accuracy_type', x='model_name', 
	# 	y='accuracy', hue='model_name', order=grouped_accuracy.index, figsize=7)

	# for ax in axes:
	# 	ax.xaxis.set_tick_params(rotation=35)
	# 	ax.set_ylim(accuracy_min, accuracy_max)
	# 	ax.axhline(0, linestyle='--', c='k')

	# plt.suptitle(f'Model performance - all averages')
	# plt.tight_layout()

	# out_fn = os.path.join(plots_dir, 'all-models_avg-accuracies.jpg')
	# plt.savefig(out_fn, dpi=300)
	# plt.close('all')

	# ###########################
	# ### Plot accuracy by top n
	# # Group by window size --> is accuracy getting better with more context? 
	# ###########################

	# out_dir = os.path.join(plots_dir, 'model-accuracy')

	# if not os.path.exists(out_dir):
	# 	os.makedirs(out_dir)

	# for accuracy_type, df in df_scores.groupby('accuracy_type'):
	# 	fig, axes = plot_accuracy_by_group(df, iterate_col='top_n', x='model_name', y='accuracy', 
	# 		hue='window_size', order=grouped_accuracy.index, figsize=6)

	# 	for ax in axes:
	# 		ax.xaxis.set_tick_params(rotation=35)
	# 		ax.set_ylim(accuracy_min, accuracy_max)
	# 		ax.axhline(0, linestyle='--', c='k')

	# 	plt.suptitle(f'Model performance - {accuracy_type} (N={len(tasks)} stories)')
	# 	plt.tight_layout()

	# 	out_fn = os.path.join(out_dir, f'model-{accuracy_type}_paired-window-size.jpg')

	# 	plt.savefig(out_fn, dpi=300)
	# 	plt.close('all')

	# ###########################
	# ### Plot accuracy by accuracy type
	# # Group by top_n --> are there differences in accuracy based on accuracy type?
	# ###########################

	# fig, axes = plot_accuracy_by_group(df_scores, iterate_col='accuracy_type', x='model_name', 
	# 	y='accuracy', hue='top_n', order=grouped_accuracy.index, figsize=7)
	
	# for ax in axes:
	# 	ax.xaxis.set_tick_params(rotation=35)
	# 	ax.set_ylim(accuracy_min, accuracy_max)
	# 	ax.axhline(0, linestyle='--', c='k')

	# plt.suptitle(f'Accuracy by number of guesses (N={len(tasks)} stories)')
	# plt.tight_layout()

	# out_fn = os.path.join(out_dir, f'model-accuracies_paired-top-n.jpg')

	# plt.savefig(out_fn, dpi=300)
	# plt.close('all')

	###########################################
	### Plot task quadrant plots for each model
	###########################################

	all_tasks_quadrants = []

	for model_name in model_names:

		out_dir = os.path.join(plots_dir, 'model-quadrant-distributions', model_name)

		if not os.path.exists(out_dir):
			os.makedirs(out_dir)

		for task in tasks:
			candidate_idxs = get_stim_candidate_idxs(task)

			model_results = load_model_data(models_dir, model_name=model_name, task=task, top_n=5, window_size=100)
			model_results.loc[:, 'binary_accuracy'] = model_results['binary_accuracy'].astype(bool)
			model_results = model_results.iloc[candidate_idxs]

			fig, axes, df_quadrants = plot_quadrant_distributions(model_results.dropna(), 'fasttext_avg_accuracy', 45)

			plt.suptitle(f'{model_name} - task {task}')
			out_fn = os.path.join(out_dir, f'{model_name}-{task}_quadrant-distributions.jpg')

			plt.savefig(out_fn, dpi=300)
			plt.close('all')

			df_quadrants['model_name'] = model_name
			all_tasks_quadrants.append(df_quadrants)

	all_tasks_quadrants = pd.concat(all_tasks_quadrants)
	all_tasks_quadrants = pd.melt(all_tasks_quadrants, id_vars=['model_name', 'task'], var_name=['quadrant_type'], value_name='proportion')

	fig, axes = plot_accuracy_by_group(all_tasks_quadrants, iterate_col=None, x='quadrant_type', y='proportion', 
		hue='model_name', figsize=10)

	# Add means on top of each box
	for ax in (axes if isinstance(axes, (list, np.ndarray)) else [axes]):
		grouped = all_tasks_quadrants.groupby(['quadrant_type', 'model_name'])['proportion'].median()
		
		for i, (group, mean_val) in enumerate(grouped.items()):
			quadrant_type, model_name = group
			
			# Find the position of the box in seaborn (this matches category index + hue offset)
			x = list(all_tasks_quadrants['quadrant_type'].unique()).index(quadrant_type)
			hue_offset = list(all_tasks_quadrants['model_name'].unique()).index(model_name)
			
			# You may need to adjust the offset scaling depending on hue count
			ax.text(
				x + hue_offset * 0.2,  # shift horizontally for hue
				mean_val + 0.01,       # slightly above box
				f"{mean_val:.2f}",
				ha='center', va='bottom', fontsize=9, color='black'
			)

	plt.suptitle(f'All model quadrant distributions')
	plt.tight_layout()

	out_dir = os.path.join(plots_dir, 'model-quadrant-distributions')
	out_fn = os.path.join(out_dir, 'all-model_quadrant-distributions.jpg')
	
	plt.savefig(out_fn, dpi=300)
	plt.close('all')
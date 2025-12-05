import sys
import os
import glob

import numpy as np
import itertools  
from collections import defaultdict

DATASETS = {
	'huth-moth': {
		'tasks': [
			'alternateithicatom', 'avatar', 'legacy', 'odetostepfather', 'souls',
			'howtodraw', 'myfirstdaywiththeyankees', 'naked', 'undertheinfluence', #'life',
			'exorcism', 'fromboyhoodtofatherhood', 'sloth', 'stagefright', 'tildeath',
			'adollshouse', 'adventuresinsayingyes', 'buck', 'haveyoumethimyet', 'inamoment', 'theclosetthatateeverything',
			'eyespy', 'hangtime', 'itsabox', 'swimmingwithastronauts', 'thatthingonmyarm', 'wheretheressmoke'
		],
		'n_trs': [
			354, 378, 410, 414, 360, 
			365, 368, 433, 314, #440,
			478, 357, 448, 304, 334,
			252, 402, 343, 507, 215, 325,
			389, 334, 365, 395, 444, 300
		]
	},
	'deniz-readinglistening': {
		'tasks':  [
			'alternateithicatom', 'avatar', 'legacy', 'odetostepfather', 'souls',
			'howtodraw', 'myfirstdaywiththeyankees', 'naked', 'undertheinfluence', 'life', 'wheretheressmoke'
		],
		'n_trs':  [
			354, 378, 410, 414, 360, 
			365, 368, 433, 314, 440, 300
		]		
	}
}

# These two functions let us instantiate and convert nested dictionaries easily
def make_nested_dictionary():
	factory = lambda: defaultdict(factory)
	defdict = factory()
	return defdict

def all_combinations(x):
	return itertools.chain.from_iterable(
		itertools.combinations(x, i + 1)
		for i in range(len(x)))

def all_equal(iterable):
	g = itertools.groupby(iterable)
	return next(g, True) and not next(g, False)

def attempt_makedirs(d):

	if not os.path.exists(d):
		try:
			os.makedirs(d)
		except Exception:
			pass
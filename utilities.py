# coding: utf-8
import numpy as np
import csv
import codecs
import os
from collections import defaultdict

SPACE = "_"
EMPTY = ""
INV_PUNCTUATION_CODES = {EMPTY:0, SPACE:0, ',':1, '.':2, '?':3, '!':4, '-':5, ';':6, ':':7, '':0}
PUNCTUATION_VOCABULARY = {0:SPACE, 1:',', 2:'.', 3:'?', 4:'!', 5:'-', 6:';', 7:':'}
REDUCED_PUNCTUATION_VOCABULARY = {0:SPACE, 1:',', 2:'.', 3:'?'}
REDUCED_INV_PUNCTUATION_CODES = {EMPTY:0, SPACE:0, ',':1, '.':2, '?':3, '':0}
EOS_PUNCTUATION_CODES = [2,3,4,5,6,7]

END = "<END>"
UNK = "<UNK>"
EMP = "<EMP>"

PAUSE_FEATURE_NAME = 'pause_before'

def read_proscript(filename):
	dict = {}
	columns = defaultdict(list) # each value in each column is appended to a list

	with open(filename) as f:
		reader = csv.DictReader(f, delimiter='\t') # read rows into a dictionary format
		for row in reader: # read a row as {column1: value1, column2: value2,...}
			for (k,v) in row.items(): # go over each column name and value 
				if k == "word" or k == "punctuation":
					columns[k].append(v) # append the value into the appropriate list
				else:
					try:
						columns[k].append(float(v)) # real value
					except ValueError:
						print("ALARM:%s"%v)
						columns[k].append(0.0)
	return columns

def checkArgument(argname, isFile=False, isDir=False):
	if not argname:
		return False
	else:
		if isFile and not os.path.isfile(argname):
			return False
		if isDir and not os.path.isdir(argname):
			return False
	return True

def iterable_to_dict(arr):
	return dict((x.strip(), i) for (i, x) in enumerate(arr))

def read_vocabulary(file_name):
	with codecs.open(file_name, 'r', 'utf-8') as f:
		return iterable_to_dict(f.readlines())

def to_array(arr, dtype=np.int32):
	# minibatch of 1 sequence as column
	return np.array([arr], dtype=dtype).T

def create_pause_bins():
	bins = np.arange(0, 1, 0.05)
	bins = np.concatenate((bins, np.arange(1, 2, 0.1)))
	bins = np.concatenate((bins, np.arange(2, 5, 0.2)))
	bins = np.concatenate((bins, np.arange(5, 10, 0.5)))
	bins = np.concatenate((bins, np.arange(10, 20, 1)))
	return bins

def create_semitone_bins():
	bins = np.arange(-20, -10, 1)
	bins = np.concatenate((bins, np.arange(-10, -5, 0.5)))
	bins = np.concatenate((bins, np.arange(-5, 0, 0.25)))
	bins = np.concatenate((bins, np.arange(0, 5, 0.25)))
	bins = np.concatenate((bins, np.arange(5, 10, 0.5)))
	bins = np.concatenate((bins, np.arange(10, 20, 1)))
	return bins

def convert_value_to_level_sequence(value_sequence, bins):
	levels = []
	for value in value_sequence:
		level = 0
		for bin_no, bin_upper_limit in enumerate(bins):
			if value > bin_upper_limit:
				level += 1
			else:
				break
		levels.append(level)
	return levels

def reducePuncCode(puncCode):
	if puncCode in [4, 5, 6, 7]: #period
		return 2
	else:
		return puncCode

def reducePunc(punc):
	puncCode = INV_PUNCTUATION_CODES[punc]
	reducedPuncCode = reducePuncCode(puncCode)
	return PUNCTUATION_VOCABULARY[reducedPuncCode]

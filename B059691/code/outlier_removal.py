import sys
import os
import shutil
from os.path import isfile, join
import csv
import pandas as pd
import numpy as np
from collections import defaultdict

def mmx_outliers():
	df = pd.read_csv('mmx_local_norms.data')
	df.set_index('label', inplace=True)
	
	max_3std = defaultdict()
	min_3std = defaultdict()
	for i in range(1,517):
		max_3std[str(i)] = 3 * np.std(df['{}_max'.format(i)])
		min_3std[str(i)] = 3 * np.std(df['{}_min'.format(i)])
	
	max_997 = defaultdict()
	min_997 = defaultdict()
	for i in range(1,517):
		max = df.loc[df.loc[:,'{}_max'.format(i)] > max_3std[str(i)], :]
		max_997[str(i)] = max.index.values
		min = df.loc[df.loc[:,'{}_min'.format(i)] < min_3std[str(i)], :]
		min_997[str(i)] = min.index.values

	l = []
	for i in range(1,513):
		for j in max_997[str(i)]:
			l.append(j)
		for j in min_997[str(i)]:
			l.append(j)
	for i in min_997[str(514)]: # hardcode min ap
			l.append(i)
	for i in max_997[str(515)]: # hardcode max f0
			l.append(i)

    l = list(set(l))
	
	for file in l:
		shutil.move('{}.mmx'.format(file), 'outliers/{}.mmx'.format(file))

def mvn_outliers():
	df = pd.read_csv('mvn_local_norms.data')
	df.set_index('label', inplace=True)
	
	max_3std = defaultdict()
	min_3std = defaultdict()
	for i in range(1,517):
		max_3std[str(i)] = 3 * np.std(df['{}_max'.format(i)])
		min_3std[str(i)] = 3 * np.std(df['{}_min'.format(i)])
	
	max_997 = defaultdict()
	min_997 = defaultdict()
	for i in range(1,517):
		max = df.loc[df.loc[:,'{}_max'.format(i)] > max_3std[str(i)], :]
		max_997[str(i)] = max.index.values
		min = df.loc[df.loc[:,'{}_min'.format(i)] < min_3std[str(i)], :]
		min_997[str(i)] = min.index.values

	l = []
	for i in range(1,513):
		for j in max_997[str(i)]:
			l.append(j)
		for j in min_997[str(i)]:
			l.append(j)
	for i in min_997[str(514)]: # hardcode min ap
			l.append(i)
	for i in max_997[str(515)]: # hardcode max f0
			l.append(i)

    l = list(set(l))
	
	for file in l:
		shutil.move('{}.mvn'.format(file), 'outliers/{}.mvn'.format(file))
		
def box_outliers():
	df = pd.read_csv('box_local_norms.data')
	df.set_index('label', inplace=True)
	
	max_3std = defaultdict()
	min_3std = defaultdict()
	for i in range(1,517):
		max_3std[str(i)] = 3 * np.std(df['{}_max'.format(i)])
		min_3std[str(i)] = 3 * np.std(df['{}_min'.format(i)])
	
	max_997 = defaultdict()
	min_997 = defaultdict()
	for i in range(1,517):
		max = df.loc[df.loc[:,'{}_max'.format(i)] > max_3std[str(i)], :]
		max_997[str(i)] = max.index.values
		min = df.loc[df.loc[:,'{}_min'.format(i)] < min_3std[str(i)], :]
		min_997[str(i)] = min.index.values

	l = []
	for i in range(1,513):
		for j in max_997[str(i)]:
			l.append(j)
		for j in min_997[str(i)]:
			l.append(j)
	for i in min_997[str(514)]: # hardcode min ap
			l.append(i)
	for i in max_997[str(515)]: # hardcode max f0
			l.append(i)

    l = list(set(l))
	
	for file in l:
		shutil.move('{}.box'.format(file), 'outliers/{}.box'.format(file))

mmx_outliers()
mvn_outliers()
box_outliers()
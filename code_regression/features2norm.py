import sys
import os
from os.path import isfile, join
import csv
import pandas as pd
import numpy as np
from scipy import stats
from collections import defaultdict

class MinMaxNormalise:

	def __init__(self):
		self.global_values = defaultdict()

	def set_local_norms(self, csvs):
	
		with open('mmx_local_norms.data', 'w') as f:
			wr = csv.writer(f, lineterminator = '\n')
			
			# set header (column names) for 512 (sp) + 1 (ap) + 1 (f0) + 1 (vuv)
			header = ['label']
			for i in range(1,517):
				header.append('{}_max'.format(i))
				header.append('{}_min'.format(i))
			wr.writerow(header)

			for file in csvs:
				label = os.path.split(os.path.splitext(os.path.normpath(file))[0])[1]
				
				# set initial arbitrary min/max values to be iterated over
				local_values_max = defaultdict()
				local_values_min = defaultdict()
				
				for i in range(1,517):
					local_values_max[str(i)] = 0 # default max
					local_values_min[str(i)] = 1000 # default min
					
				# iterate through and update minmax values for WORLD features
				df = pd.read_csv(file, header=None)
				
				for feature in range(1,517): # 516 WORLD features
					max = df.iloc[:,feature][:].max()
					min = df.iloc[:,feature][:].min()
					if max >= local_values_max[str(feature)]:
						local_values_max[str(feature)] = max
					if min <= local_values_min[str(feature)]:
						local_values_min[str(feature)] = min
				
				row = [label]
				for i in range(1,517):
					row.append(local_values_max[str(i)])
					row.append(local_values_min[str(i)])

				wr.writerow(row)
				
	def set_global_norms(self):
	
		df = pd.read_csv('mmx_local_norms.data')
		
		with open('mmx_global_norms.data', 'w') as f:
			wr = csv.writer(f, lineterminator = '\n')

			# set header (column names) for 512 (sp) + 1 (ap) + 1 (f0) + 1 (vuv)
			header = []
			for i in range(1,517):
				header.append('{}_max'.format(i))
				header.append('{}_min'.format(i))
			wr.writerow(header)
			
			# get global min/max values from local
			global_values_max = defaultdict()
			global_values_min = defaultdict()
			for i in range(1,517):
				global_values_max[str(i)] = df['{}_max'.format(i)].max()
				global_values_min[str(i)] = df['{}_min'.format(i)].min()
			
			row = []
			for i in range(1,517):
				row.append(global_values_max[str(i)])
				row.append(global_values_min[str(i)])
			
			wr.writerow(row)

	def get_global_norms(self):
	
		df = pd.read_csv('mmx_global_norms.data')
		header = []
		for i in range(1,517):
			header.append('{}_max'.format(i))
			header.append('{}_min'.format(i))
		
		for i in header:
			self.global_values[i] = np.float64(df.iloc[:,i][:])
	
	def normalise_by_local(self, csvs):
	
		local_minmax = pd.read_csv('mmx_local_norms.data')
		local_minmax.set_index('label', inplace=True)
		
		for file in csvs:

			# read in data file (to normalise)
			df = pd.read_csv(file, header=None)
			
			# get file label and minmax values
			label = os.path.split(os.path.splitext(os.path.normpath(file))[0])[1]
			minmax = local_minmax.loc[label,:]
			
			max = defaultdict()
			min = defaultdict()
			rng = defaultdict()
			
			feature = 1
			for i in range(0,len(minmax),2): # set max values for feature
				max[str(feature)] = minmax[i]
				feature += 1
			feature = 1
			for i in range(1,len(minmax),2): # set min values for feature
				min[str(feature)] = minmax[i]			
				feature += 1
			for i in range(1,517): 			 # set value range for feature
				rng[str(i)] = max[str(i)] - min[str(i)]						
			
			normalised = '{}.mmx'.format(os.path.splitext(os.path.normpath(file))[0])
			
			with open(normalised, 'w') as f:
				wr = csv.writer(f, lineterminator = '\n')
				for row in range(len(df)):
					world_516 = [df.iloc[row,:][0]] # start with list containing label
					for i in range(1,517):
						value = df.iloc[row,:][i]
						value_prime = (((value - min[str(i)]) * (0.9-0.1)) / rng[str(i)]) + 0.1
						world_516.append(value_prime)
					
					if np.isnan(world_516[514]): # hardcode replace NaNs if ap range == 0
						world_516[514] = (1*(0.9-0.1))+0.1
					if np.isnan(world_516[515]): # hardcode replace NaNs if f0 range == 0	
						world_516[515] = (0*(0.9-0.1))+0.1
					if np.isnan(world_516[516]): # hardcode replace NaNs if vuv range == 0	
						world_516[516] = (0*(0.9-0.1))+0.1
								
					wr.writerow(world_516) # minmax mormalisation scaled to [0.1, 0.9]

class MVN:

	def __init__(self):
		self.global_values = defaultdict()

	def set_local_norms(self, csvs):
	
		with open('mvn_local_norms.data', 'w') as f:
			wr = csv.writer(f, lineterminator = '\n')
			
			# set header (column names) for 512 (sp) + 1 (ap) + 1 (f0) + 1 (vuv)
			header = ['label']
			for i in range(1,517):
				header.append('{}_mean'.format(i))
				header.append('{}_std'.format(i))
			wr.writerow(header)

			for file in csvs:
				label = os.path.split(os.path.splitext(os.path.normpath(file))[0])[1]
				
				# set initial default dicts for mean/variance values
				local_values_mean = defaultdict()
				local_values_std = defaultdict()
					
				# iterate through and update mean/variance for WORLD features
				df = pd.read_csv(file, header=None)
				
				for feature in range(1,517): # 516 WORLD features
					local_values_mean[str(feature)] = np.mean(df.iloc[:,feature][:])
					local_values_std[str(feature)] = np.std(df.iloc[:,feature][:])
				
				row = [label]
				for i in range(1,517):
					row.append(local_values_mean[str(i)])
					row.append(local_values_std[str(i)])

				wr.writerow(row)
				
	def set_global_norms(self):
	
		df = pd.read_csv('mvn_local_norms.data')
		
		with open('mvn_global_norms.data', 'w') as f:
			wr = csv.writer(f, lineterminator = '\n')

			# set header (column names) for 512 (sp) + 1 (ap) + 1 (f0) + 1 (vuv)
			header = []
			for i in range(1,517):
				header.append('{}_mean'.format(i))
				header.append('{}_std'.format(i))
			wr.writerow(header)
			
			# get global mean/variance values from local
			global_values_mean = defaultdict()
			global_values_std = defaultdict()
			for i in range(1,517):
				global_values_mean[str(i)] = np.mean(df['{}_mean'.format(i)])
				global_values_std[str(i)] = np.std(df['{}_std'.format(i)])
			
			row = []
			for i in range(1,517):
				row.append(global_values_mean[str(i)])
				row.append(global_values_std[str(i)])
			
			wr.writerow(row)

	def get_global_norms(self):
	
		df = pd.read_csv('mvn_global_norms.data')
		header = []
		for i in range(1,517):
			header.append('{}_mean'.format(i))
			header.append('{}_std'.format(i))
		
		for i in header:
			self.global_values[i] = np.float64(df.iloc[:,i][:])
	
	def normalise_by_local(self, csvs):
	
		local_mvn = pd.read_csv('mvn_local_norms.data')
		local_mvn.set_index('label', inplace=True)
		
		for file in csvs:

			# read in data file (to normalise)
			df = pd.read_csv(file, header=None)
			
			# get file label and mean/variance values
			label = os.path.split(os.path.splitext(os.path.normpath(file))[0])[1]
			mvn = local_mvn.loc[label,:]
			
			mean = defaultdict()
			std = defaultdict()
			
			feature = 1
			for i in range(0,len(mvn),2): # set mean values for feature
				mean[str(feature)] = mvn[i]
				feature += 1
			feature = 1
			for i in range(1,len(mvn),2): # set variance values for feature
				std[str(feature)] = mvn[i]			
				feature += 1					
			
			normalised = '{}.mvn'.format(os.path.splitext(os.path.normpath(file))[0])
			
			with open(normalised, 'w') as f:
				wr = csv.writer(f, lineterminator = '\n')
				for row in range(len(df)):
					world_516 = [df.iloc[row,:][0]] # start with list containing label
					for i in range(1,517):
						value = df.iloc[row,:][i]
						try:
							value_prime = (value - mean[str(i)]) / std[str(i)]
						except ZeroDivisionError: # guard against ap feature == 1 or f0/vuv == 0
							value_prime = 0
						world_516.append(value_prime)
								
					wr.writerow(world_516) # mean/variance mormalisation

class BoxStandardise:

	def __init__(self):
		self.global_values = defaultdict()

	def set_local_norms(self, csvs):
	
		with open('box_local_norms.data', 'w') as f:
			wr = csv.writer(f, lineterminator = '\n')
			
			# set header (column names) for 512 (sp) + 1 (ap) + 1 (f0) + 1 (vuv)
			header = ['label']
			for i in range(1,517):
				header.append('{}_var'.format(i))
			wr.writerow(header)

			for file in csvs:
				label = os.path.split(os.path.splitext(os.path.normpath(file))[0])[1]
				
				# set initial default dicts for var values
				local_values_var = defaultdict()
					
				# iterate through and update var for WORLD features
				df = pd.read_csv(file, header=None)
				
				for feature in range(1,517): # 516 WORLD features
					local_values_var[str(feature)] = np.var(df.iloc[:,feature][:])
				
				row = [label]
				for i in range(1,517):
					row.append(local_values_var[str(i)])

				wr.writerow(row)
				
	def set_global_norms(self):
	
		df = pd.read_csv('box_local_norms.data')
		
		with open('box_global_norms.data', 'w') as f:
			wr = csv.writer(f, lineterminator = '\n')

			# set header (column names) for 512 (sp) + 1 (ap) + 1 (f0) + 1 (vuv)
			header = []
			for i in range(1,517):
				header.append('{}_var'.format(i))
			wr.writerow(header)
			
			# get global var values from local
			global_values_var = defaultdict()
			for i in range(1,517):
				global_values_var[str(i)] = np.var(df['{}_var'.format(i)])
			
			row = []
			for i in range(1,517):
				row.append(global_values_var[str(i)])
			
			wr.writerow(row)

	def get_global_norms(self):
	
		df = pd.read_csv('box_global_norms.data')
		header = []
		for i in range(1,517):
			header.append('{}_var'.format(i))
		
		for i in header:
			self.global_values[i] = np.float64(df.iloc[:,i][:])
	
	def normalise_by_local_old(self, csvs):
	
		local_box = pd.read_csv('box_local_norms.data')
		local_box.set_index('label', inplace=True)
		
		for file in csvs:

			# read in data file (to normalise)
			df = pd.read_csv(file, header=None)
			
			# get file label and var values
			label = os.path.split(os.path.splitext(os.path.normpath(file))[0])[1]
			box = local_box.loc[label,:]
			
			var = defaultdict()

			for i in range(0,len(box)): # set var values for feature
				var[str(i+1)] = box[i]				
			
			normalised = '{}.box'.format(os.path.splitext(os.path.normpath(file))[0])
			
			with open(normalised, 'w') as f:
				wr = csv.writer(f, lineterminator = '\n')
				
				for row in range(len(df)):
					world_516 = [df.iloc[row,:][0]] # start with list containing label
					for i in range(1,517):
						value = df.iloc[row,:][i]
						value_prime = (((value + 10e-7)**0.043) -1)/0.043 # lambdas as per https://danielsdiscoveries.wordpress.com/2017/09/29/spectrogram-input-normalisation-for-neural-networks/
						world_516.append(value_prime)

					wr.writerow(world_516) # boxcox standardisation
					
		def normalise_by_local_new(self, csvs):
	
		local_box = pd.read_csv('box_local_norms.data')
		local_box.set_index('label', inplace=True)
		
		for file in csvs:

			# read in data file (to normalise)
			df = pd.read_csv(file, header=None)
			
			# get file label and var values
			label = os.path.split(os.path.splitext(os.path.normpath(file))[0])[1]
			box = local_box.loc[label,:]
			
			var = defaultdict()

			for i in range(0,len(box)): # set var values for feature
				var[str(i+1)] = box[i]				
			
			normalised = '{}.box'.format(os.path.splitext(os.path.normpath(file))[0])
			
			with open(normalised, 'w') as f:
				wr = csv.writer(f, lineterminator = '\n')
				
				f0 = df.iloc[:,515].values
				for index, value in enumerate(f0):
					if value <= 0:
						f0[index] = 10e-7 # smooth zero values for f0
				df.iloc[:,515] = f0
			
				vuv = df.iloc[:,516].values
				for index, value in enumerate(vuv):
					if value <= 0:
						vuv[index] = 10e-7  # smooth zero values for vuv
				df.iloc[:,516] = vuv
	
				for column in range(1,517):
					df.iloc[:,column] = stats.boxcox(df.iloc[:,column])[0] # in-place boxcox standardisation
				for row in range(len(df)):
					wr.writerow(df.iloc[row,:]) # boxcox standardisation

data = [f for f in os.listdir('.') if os.path.splitext(f)[1] == '.csv']

minmax = MinMaxNormalise()
minmax.set_local_norms(data)
minmax.set_global_norms()
minmax.normalise_by_local(data)

mvn = MVN()
mvn.set_local_norms(data)
mvn.set_global_norms()
mvn.normalise_by_local(data)

boxcox = BoxStandardise()
boxcox.set_local_norms(data)
boxcox.set_global_norms()
#boxcox.normalise_by_local_old(data)
boxcox.normalise_by_local_new(data)
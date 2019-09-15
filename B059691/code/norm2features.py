import sys
import os
from os.path import isfile, join
import csv
import pandas as pd
import numpy as np
from collections import defaultdict

class MinMaxDenormalise:

	def __init__(self):
	
		self.local_values = defaultdict()
		self.global_values = defaultdict()

	def get_local_norms(self, csvs):
		
		try:
			local_norms = pd.read_csv('mmx_local_norms.data', index_col=0)
			header = []
			for i in range(1,517):
				header.append('{}_max'.format(i))
				header.append('{}_min'.format(i))
			
			for i in local_norms.index.values.tolist():
				self.local_values[i] = defaultdict()
				for j in header:
					self.local_values[i][j] = np.float64(local_norms.loc[i,j])
		
		except FileNotFoundError:	
			print("'mmx_global_norms.data' does not exist!")
			exit(1)
		
	def get_global_norms(self):
		
		try:
			global_norms = pd.read_csv('mmx_global_norms.data')
			header = []
			for i in range(1,517):
				header.append('{}_max'.format(i))
				header.append('{}_min'.format(i))
			
			for i in header:
				self.global_values[i] = np.float64(global_norms.loc[:,i][:])
	
		except FileNotFoundError:	
			print("'mmx_global_norms.data' does not exist!")
			exit(1)
	
	def denormalise_by_local(self, csvs):
	
		local_minmax = pd.read_csv('mmx_local_norms.data')
		local_minmax.set_index('label', inplace=True)
		
		for file in csvs:

			# read in data file (to denormalise)
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
			
			denormalised = '{}_denormalised.csv'.format(os.path.splitext(os.path.normpath(file))[0])
			
			with open(denormalised, 'w') as f:
				wr = csv.writer(f, lineterminator = '\n')
				for row in range(len(df)):
					world_516 = [df.iloc[row,:][0]] # start with list containing label
					for i in range(1,517):
						value = df.iloc[row,:][i]
						if value <= 0:
							value_prime = 0
						else:
							value_prime = abs((((value - 0.1) * rng[str(i)])/(0.9-0.1))+min[str(i)])
						
						world_516.append(value_prime)
					
					if world_516[514] >= 1: # hardcode replace ap if above 1
						world_516[514] = 1
					
					if world_516[515] <= 50: # hardcode replace f0 below 50 as 0
						world_516[515] = 0
						world_516[516] = 0 # hardcode vuv flag if not f0
					else:
						world_516[516] = 1 # hardcode vuv flag if f0
								
					wr.writerow(world_516) # minmax denormalisation

class MVNDenormalise:

	def __init__(self):
	
		self.local_values = defaultdict()
		self.global_values = defaultdict()

	def get_local_norms(self, csvs):
		
		try:
			local_norms = pd.read_csv('mvn_local_norms.data', index_col=0)
			header = []
			for i in range(1,517):
				header.append('{}_mean'.format(i))
				header.append('{}_std'.format(i))
			
			for i in local_norms.index.values.tolist():
				self.local_values[i] = defaultdict()
				for j in header:
					self.local_values[i][j] = np.float64(local_norms.loc[i,j])
		
		except FileNotFoundError:	
			print("'mvn_global_norms.data' does not exist!")
			exit(1)
		
	def get_global_norms(self):
		
		try:
			global_norms = pd.read_csv('mvn_global_norms.data')
			header = []
			for i in range(1,517):
				header.append('{}_mean'.format(i))
				header.append('{}_std'.format(i))
			
			for i in header:
				self.global_values[i] = np.float64(global_norms.loc[:,i][:])
	
		except FileNotFoundError:	
			print("'mvn_global_norms.data' does not exist!")
			exit(1)
	
	def denormalise_by_local(self, csvs):
	
		local_mvn = pd.read_csv('mvn_local_norms.data')
		local_mvn.set_index('label', inplace=True)
		
		for file in csvs:

			# read in data file (to denormalise)
			df = pd.read_csv(file, header=None)
			
			# get file label and mean/variance values
			label = os.path.split(os.path.splitext(os.path.normpath(file))[0])[1]
			mvn = local_mvn.loc[label,:]
			
			mean = defaultdict()
			std = defaultdict()
			
			feature = 1
			for i in range(0,len(mvn),2): # set max values for feature
				mean[str(feature)] = mvn[i]
				feature += 1
			feature = 1
			for i in range(1,len(mvn),2): # set min values for feature
				std[str(feature)] = mvn[i]			
				feature += 1					
			
			denormalised = '{}_denormalised.csv'.format(os.path.splitext(os.path.normpath(file))[0])
			
			with open(denormalised, 'w') as f:
				wr = csv.writer(f, lineterminator = '\n')
				for row in range(len(df)):
					world_516 = [df.iloc[row,:][0]] # start with list containing label
					for i in range(1,517):
						value = df.iloc[row,:][i]
						if value <= 0:
							value_prime = 0
						else:
							value_prime = (value * std[str(i)]) + mean[str(i)]
						
						world_516.append(value_prime)
					
					if world_516[514] >= 1: # hardcode replace ap if above 1
						world_516[514] = 1
					
					if world_516[515] <= 50: # hardcode replace f0 below 50 as 0
						world_516[515] = 0
						world_516[516] = 0 # hardcode vuv flag if not f0
					else:
						world_516[516] = 1 # hardcode vuv flag if f0
								
					wr.writerow(world_516) # mvn denormalisation

class BoxDenormalise:

	def __init__(self):
	
		self.local_values = defaultdict()
		self.global_values = defaultdict()

	def get_local_norms(self, csvs):
		
		try:
			local_norms = pd.read_csv('box_local_norms.data', index_col=0)
			header = []
			for i in range(1,517):
				header.append('{}_var'.format(i))
			
			for i in local_norms.index.values.tolist():
				self.local_values[i] = defaultdict()
				for j in header:
					self.local_values[i][j] = np.float64(local_norms.loc[i,j])
		
		except FileNotFoundError:	
			print("'box_global_norms.data' does not exist!")
			exit(1)
		
	def get_global_norms(self):
		
		try:
			global_norms = pd.read_csv('box_global_norms.data')
			header = []
			for i in range(1,517):
				header.append('{}_var'.format(i))
			
			for i in header:
				self.global_values[i] = np.float64(global_norms.loc[:,i][:])
	
		except FileNotFoundError:	
			print("'box_global_norms.data' does not exist!")
			exit(1)
	
	def denormalise_by_local(self, csvs):
	
		local_box = pd.read_csv('box_local_norms.data')
		local_box.set_index('label', inplace=True)
		
		for file in csvs:

			# read in data file (to denormalise)
			df = pd.read_csv(file, header=None)
			
			# get file label and var values
			label = os.path.split(os.path.splitext(os.path.normpath(file))[0])[1]
			box = local_box.loc[label,:]
			
			var = defaultdict()
			
			for i in range(0,len(box)): # set var values for feature
				var[str(i+1)] = box[i]						
			
			denormalised = '{}_denormalised.csv'.format(os.path.splitext(os.path.normpath(file))[0])
			
			with open(denormalised, 'w') as f:
				wr = csv.writer(f, lineterminator = '\n')
				for row in range(len(df)):
					world_516 = [df.iloc[row,:][0]] # start with list containing label
					for i in range(1,517):
						value = df.iloc[row,:][i]
						if value <= 0:
							value_prime = 0
						else:
							value_prime = (((value*0.043)+1)**(1/0.043))-10e-7
						
						world_516.append(value_prime)
					
					if world_516[514] >= 1: # hardcode replace ap if above 1
						world_516[514] = 1
					
					if world_516[515] <= 50: # hardcode replace f0 below 50 as 0
						world_516[515] = 0
						world_516[516] = 0 # hardcode vuv flag if not f0
					else:
						world_516[516] = 1 # hardcode vuv flag if f0
								
					wr.writerow(world_516) # boxcox denormalisation

def mmx():
	data = [f for f in os.listdir('.') if os.path.splitext(f)[1] == '.mmx']
	minmax = MinMaxDenormalise()
	minmax.get_local_norms(data)
	minmax.get_global_norms()
	minmax.denormalise_by_local(data)

def mvn():
	data = [f for f in os.listdir('.') if os.path.splitext(f)[1] == '.mvn']
	mvn = MVNDenormalise()
	mvn.get_local_norms(data)
	mvn.get_global_norms()
	mvn.denormalise_by_local(data)
	
def box():
	data = [f for f in os.listdir('.') if os.path.splitext(f)[1] == '.box']
	boxcox = BoxDenormalise()
	boxcox.get_local_norms(data)
	boxcox.get_global_norms()
	boxcox.denormalise_by_local(data)

mmx()
#mvn()
#box()
import scipy.io
import mne.io
import numpy as np
from copy import deepcopy
import _pickle as pkl
import pdb
import glob
import os

class BatchData:

	shortest_trial = 0 # necessary to ensure dims of stacking arrays

	def __init__(self):
		
		self.batch = {}
		self.data = []
		
		self.no_trials = 0
		self.no_channels = 0
		self.no_features = 0
		self.no_raw_channels = 0
		
		self.thinking_inds = None
		self.thinking_data = None
		self.thinking_feats = None
		
		self.prompts = None
		self.prompts_index = []
		self.prompts_binary = []

	def load_raw_data(self, path_to_data):
				
		montage = mne.channels.read_montage('standard_1020')
		montage.ch_names = [ch_name.upper() for ch_name in montage.ch_names]
		raw = mne.io.read_raw_eeglab(path_to_data,montage,'auto', preload = True)
		self.thinking_data = raw
		self.no_raw_channels = len(raw.ch_names)
	
	def load_epoch_inds(self, path_to_data):
		
		inds = scipy.io.loadmat(path_to_data)

		#thinking_inds = inds['thinking_inds'][0,0]
		#clearing_inds = inds['clearing_inds'][0,0]
		#speaking_inds = inds['speaking_inds'][0,0]
		
		self.thinking_inds = inds['thinking_inds'][0]
		
	def load_data(self, path_to_data, task):
	
		mat = scipy.io.loadmat(path_to_data)
		
		#print(mat['__header__'])
		#print(mat['__version__'])
		#print(mat['__globals__'])
		#main = mat['all_features'][0,0]
		#eeg_features = main[0]
		#thinking_feats = eeg_features[0,0][0]
		#clearing_feats = eeg_features[0,0][1]
		#stimuli_feats = eeg_features[0,0][2]
		#speaking_feats = eeg_features[0,0][3]
		#wav_features = main[1][0]
		#face_features = main[2][0]
		#feature_labels = main[3][0]
		#face_labels = main[4][0]
		#prompts = main[5][0]
		
		self.thinking_feats = mat['all_features'][0,0][0][0,0][0]
		self.prompts = mat['all_features'][0,0][5][0]
		for i in range(0,len(self.prompts)):
			if self.prompts[i][0] in task[0]:
				self.prompts[i] = self.prompts[i][0]
				self.prompts_index.append(i)
				self.prompts_binary.append(0)	
			elif self.prompts[i] in task[1]:
				self.prompts[i] = self.prompts[i][0]
				self.prompts_index.append(i)
				self.prompts_binary.append(1)
			else:
				self.prompts[i] = None
		self.prompts = [i for i in self.prompts if i]
		
		self.no_trials = len(self.thinking_feats[0])
		self.no_channels = len(self.thinking_feats[0,0])
		self.no_features = len(self.thinking_feats[0,0][0])

		self.no_windows = 19

	def batch_data(self, dim):

		if dim == '3d_raw':
			print('reading raw data from trials...')
			trial_no = 0
			for trial in self.prompts_index: # selected trials
				data1 = []
				for channel in range(0,self.no_raw_channels): # 62 channels
					stream = self.thinking_data[channel][0][0][self.thinking_inds[trial][0,0]:self.thinking_inds[trial][0,0]+self.shortest_trial]
					data1.append(stream.tolist())
				self.data.append(data1)
				print(' {} (trial no. {})\t\t'.format(self.prompts[trial_no], trial), end='\r')
				trial_no += 1
			print(' '*20, end='\r')
		elif dim == '2d_features':
			for trial in self.prompts_index: # selected trials
				data1 = []
				for feature in range(0,self.no_features,self.no_windows): # 63 features
					for window in range(0,self.no_windows): # 19 windows
						for channel in range(0,self.no_channels): # 64 channels
							data1.append(self.thinking_feats[0,trial][channel][window+feature])
				self.data.append(data1)
		elif dim == '3d_features':
			for trial in self.prompts_index: # selected trials
				data1 = []
				for channel in range(0,self.no_channels): # 64 channels
					data2 = []	
					for window in range(0,self.no_windows): # 19 windows
						data2.append(self.thinking_feats[0,trial][channel][window])
					data1.append(data2)
				self.data.append(data1)
		elif dim == '4d_features':
			for trial in self.prompts_index:  # selected trials
				data1 = []
				for feature in range(0,self.no_features,self.no_windows): # 63 features
					data2 = []
					for window in range(0,self.no_windows): # 19 windows
						data3 = []
						for channel in range(0,self.no_channels): # 64 channels
							data3.append(self.thinking_feats[0,trial][channel][window+feature])
						data2.append(data3)
					data1.append(data2)
				self.data.append(data1)
		else:
			pass

		self.batch[b'labels'] = self.prompts_binary
		self.batch[b'data'] = np.array(self.data)
		self.batch[b'prompts'] = [i.encode('utf8') for i in self.prompts]

	def pkl_data(self, filename = 'filename'):
		self.batch[b'batch_label'] = filename.encode('utf8')
		pkl_out = open(filename, 'wb')
		pkl.dump(self.batch, pkl_out)
		pkl_out.close()
		
	def make_batch(self, data, task, dim, batch_no = 0):
		self.load_data(data, task)
		self.batch_data(dim)
		self.pkl_data(filename = 'data_batch_{}'.format(batch_no))

def concatenate_batches():
	with open('data_batch_1', 'rb') as f:
		data = pkl.load(f, encoding='bytes')
	all_data = data[b'data']
	all_labels = data[b'labels']
	for i in range(2,len(folders)+1): # no. of data batches
		with open('data_batch_{}'.format(i), 'rb') as f:
			data = pkl.load(f, encoding='bytes')
		all_data = np.concatenate((all_data, data[b'data']), 0)
		all_labels = np.concatenate((all_labels, data[b'labels']), 0)		
	data = {b'data':all_data, b'labels':all_labels}
	print("Shape of data in outfile file 'data' is: ", np.shape(data[b'data']))
	with open('data', 'wb') as pkl_out:
		pkl.dump(data, pkl_out)
	for i in range(1,len(folders)+1):
		os.remove('data_batch_{}'.format(i))

def shortest_trial(folders): # Around ~4.8 seconds, but may only need ~1000ms
	i = []
	for j in folders:
		inds = scipy.io.loadmat('{}/epoch_inds.mat'.format(j))
		thinking_inds = inds['thinking_inds'][0]
		i.append(min([(trial[0,1]-trial[0,0]) for trial in thinking_inds]) + 1)
	return(min(i))

def make_data(folders, task, dim):
	batch_no = 1
	if 'features' in dim:
		for i in folders:
			BatchData().make_batch('{}/all_features_ICA.mat'.format(i), task, dim, batch_no = batch_no)
			print('done {} as data_batch_{}'.format(i,batch_no))
			batch_no += 1
	if 'raw' in dim:
		b = BatchData()
		for i in folders:
			b.shortest_trial = shortest_trial(folders)
			b.load_epoch_inds('{}/epoch_inds.mat'.format(i))
			b.load_raw_data(glob.glob('{}/*.set'.format(i))[0])
			b.make_batch('{}/all_features_ICA.mat'.format(i), task, dim, batch_no = batch_no)
			print('done {} as data_batch_{}'.format(i,batch_no))
			batch_no += 1
			b = BatchData()
	else:
		pass
	print('\nConcatenating batches...')
	concatenate_batches()

##################################################################

# Specify folders:

folders = [
'karaone/MM05',
'karaone/MM08',
'karaone/MM09',
'karaone/MM10',
'karaone/MM11',
'karaone/MM12',
'karaone/MM14',
'karaone/MM15',
'karaone/MM16',
'karaone/MM18',
'karaone/MM19',
'karaone/MM20',
'karaone/MM21',
'karaone/P02',
]

# Specify tasks:

binary_cv = [['/uw/','/iy/'],['/m/','/n/']]
binary_nasal = [['/tiy/','/piy/','pat','pot'],['/m/','/n/','knew','gnaw']]
binary_bilabial = [['/tiy/','/iy/','/n/'],['/piy/','pat','pot']]
binary_iy = [['/tiy/','/iy/','/piy/','/diy/'],['gnaw','pat','knew','pot']]
binary_uw = [['/uw/', 'knew'],['/iy/', 'gnaw']]
binary_backness = [['/uw/'],['/iy/']]
binary_voice = [['/tiy/'],['/diy/']]

# Specify dimensions:

raw_3d = '3d_raw'			# reformat into trials x channels x raw time. typically a 132x62x4839 numpy array
features_2d = '2d_features' # reformat into trials x features+time+channels. typically a 132x76608 numpy array
features_3d = '3d_features' # reformat into trials x features x time+channels. typically a 132x63x1216 numpy array
features_4d = '4d_features' # reformat into trials x features x time x channels. typically a 132x63x19x64 numpy array

##################################################################

make_data(folders, binary_cv, raw_3d)
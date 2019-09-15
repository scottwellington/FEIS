import os
import torch
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from torch import nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler

class Dataset(Dataset):
	
	def __init__(self, csv_path, transform=None):
		self.transform = transform
		self.data = pd.read_csv(csv_path, header=None)
		
	def __len__(self): # no. of samples
		return len(self.data)
	
	def __getitem__(self, index): # as ndarray
		data = self.data.iloc[index, 1:]
		label = self.data.iloc[index, 0]

		if self.transform:
			data = np.array(data, dtype=np.float32)
				# not really a transform; just a placeholder
		return data, label
	
class AutoEncoder(nn.Module):

	def __init__(self):
	
		super(AutoEncoder, self).__init__()
		
		self.encoder = nn.Sequential(
			nn.Linear(513+1+1+1, 512), # sp, ap, f0, vuv
			nn.LeakyReLU(),
			nn.Linear(512, 400),
			nn.LeakyReLU(),
			nn.Linear(400, 300),
			nn.LeakyReLU(),
			nn.Linear(300, 256),
			nn.Tanh()
		)		
		self.decoder_init = nn.Sequential(
			nn.Linear(256, 300),
			nn.LeakyReLU(),
			nn.Linear(300, 400),
			nn.LeakyReLU(),
			nn.Linear(400, 512),
			nn.LeakyReLU()
		)
		self.decoder_sp = nn.Sequential(
			nn.Linear(512, 512),
			nn.LeakyReLU(),
			nn.Linear(512, 513),
			nn.ReLU()
		)
		self.decoder_ap = nn.Sequential(
			nn.Linear(512, 32),
			nn.LeakyReLU(),
			nn.Linear(32, 8),
			nn.LeakyReLU(),
			nn.Linear(8, 1),
			nn.ReLU()
		)
		self.decoder_f0 = nn.Sequential(
			nn.Linear(512, 32),
			nn.LeakyReLU(),
			nn.Linear(32, 8),
			nn.LeakyReLU(),
			nn.Linear(8, 1),
			nn.ReLU()
		)
		self.decoder_vuv = nn.Sequential(
			nn.Linear(512, 32),
			nn.LeakyReLU(),
			nn.Linear(32, 8),
			nn.LeakyReLU(),
			nn.Linear(8, 1),
			nn.ReLU()
		)
		
	def forward(self, x):
	
	# =========================encode===========================
	# ===================bottleneck features====================
		if args.encode:
		
			x_encoded = self.encoder(x)
		
			return x_encoded
			
	# =========================decode===========================
	# ===================bottleneck features====================
		if args.decode:
		
			x_decoded = self.decoder_init(x)
			
			sp = self.decoder_sp(x_decoded)
			ap = self.decoder_ap(x_decoded)
			f0 = self.decoder_f0(x_decoded)
			vuv = self.decoder_vuv(x_decoded)

			x_decoded = torch.cat((sp, ap, f0, vuv), 1)
		
			return x_decoded
			
	# ==========================train===========================
	# =======================end-to-end=========================
		if args.train:
	
			x_encoded = self.encoder(x)
			
			# ======================noise=======================
			mu, sigma = 0, 1e-6 # additive Gaussian noise
			noise = np.random.normal(mu, sigma, [np.shape(x_encoded)[0],np.shape(x_encoded)[1]])
			noise = torch.tensor(np.array(noise, dtype=np.float32))
			x_encoded = torch.add(noise, x_encoded)
			# ==================================================
			
			x_decoded = self.decoder_init(x_encoded)
			
			sp = self.decoder_sp(x_decoded)
			ap = self.decoder_ap(x_decoded)
			f0 = self.decoder_f0(x_decoded)
			vuv = self.decoder_vuv(x_decoded)

			x_decoded = torch.cat((sp, ap, f0, vuv), 1)
			
			return x_encoded, x_decoded

class EarlyStopping:

	def __init__(self, patience=20):
		
		self.patience = patience
		self.patience_count = 0
		self.bar_for_val_loss = 1e10
		self.out_of_patience = False
		
	def __call__(self, current_val_loss, model):
		
		if current_val_loss <= self.bar_for_val_loss:
			torch.save(model.state_dict(), 'checkpoint.pth')
			self.bar_for_val_loss = current_val_loss
			self.patience_count = 0
		else:
			self.patience_count += 1
			if self.patience_count == self.patience:
				self.out_of_patience = True

def train_model():
	
	for epoch in range(num_epochs):
		
		train_loss = []		# store training loss over batch iterations
		val_loss = []		# store validation loss over batch iterations
		
		# =====================TRAINING=====================
		
		model.train() 											# Set model for training
		
		for data in train_dataloader:
			input, label = data
			input = Variable(input)
			if cuda:
				input.cuda()
			# =====================forward======================
			encoded, decoded = model(input)						# run batched input through the network
			loss = loss_function(decoded, input) 				# loss function defined below (MSE)
			# =====================backward=====================
			loss.backward() 									# backpropagate and compute new gradients
			optimizer.step() 									# apply updated gradients
			# =======================log========================
			train_loss.append(loss.item())						# store training loss
			optimizer.zero_grad() 								# reset gradients from earlier training step

		# ====================VALIDATION====================
		
		model.eval()											# Set model to evaluation for validation
		
		for data in val_dataloader:
			input, label = data
			input = Variable(input)
			if cuda:
				input.cuda()
			# =====================forward======================
			encoded, decoded = model(input)						# run batched input through the network
			loss = loss_function(decoded, input) 				# loss function defined below (MSE)
			# =======================log========================
			val_loss.append(loss.item())						# store validation loss
		
		# =====================PRINT LOG====================
		
		log = 'epoch [{}/{}]\ntraining loss: {:.10f}\nvalidation loss: {:.10f}\n'
		print(log.format(epoch + 1, num_epochs, np.average(train_loss), np.average(val_loss)))
		
		early_stopping(np.average(val_loss), model) 			# if validation loss decreases, save model and stop training
		if early_stopping.out_of_patience:
			print('early stopping: no validation improvement for {} epochs'.format(patience))
			break
		
		scheduler.step()										# prompt scheduler step
		
	model.load_state_dict(torch.load('checkpoint.pth'))			# if early stopping, load the last checkpoint
		
def encode_bn_reps():

	with open("bottleneck_reps.csv", 'w') as f:
		wr = csv.writer(f, lineterminator = '\n')
		for data in dataloader:
			input, label = data
			input = Variable(input)
			if cuda:
				input.cuda()
			# =====================forward======================
			bn_reps = model(input)								# get bottleneck representations
			# ======================write=======================
			for i in range(len(label)):
				wr.writerow([label[i]] + bn_reps[i].tolist())	# write to file
	
def decode_bn_reps():

	with open("world_params.csv", 'w') as f:
		wr = csv.writer(f, lineterminator = '\n')	
		for data in dataloader:
			input, label = data
			input = Variable(input)
			if cuda:
				input.cuda()
			# =====================forward======================
			decoded = model(input)								# get WORLD vocoder parameters
			# ======================write=======================
			for i in range(len(label)):
				wr.writerow([label[i]] + decoded[i].tolist())	# write to file

def plot_losses_over_time(): # adapted from https://github.com/Bjarten/early-stopping-pytorch
	
	# load saved training and validation losses from the trained autoencoder
	with open('loss_avgs.pkl', 'r') as losses:
		train_loss = pkl.load(losses)[0]
		val_loss = pkl.load(losses)[1]
		
	# visualize the loss as the network trained
	fig = plt.figure(figsize=(10,8))
	plt.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss')
	plt.plot(range(1,len(val_loss)+1),val_loss,label='Validation Loss')

	# find position of lowest validation loss
	minposs = val_loss.index(min(val_loss))+1 
	plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

	plt.xlabel('epochs')
	plt.ylabel('loss')
	plt.ylim(0, 0.5) # consistent scale
	plt.xlim(0, len(train_loss)+1) # consistent scale
	plt.grid(True)
	plt.legend()
	plt.tight_layout()
	plt.show()
	fig.savefig('loss_plot.png', bbox_inches='tight')

# ====================params========================
num_epochs = 1500
batch_size = 256
learning_rate = 1e-4
patience = 50
# ==================================================

# Set if you want to use GPU
cuda = True if torch.cuda.is_available() else False

# Initialize model
model = AutoEncoder()
if cuda:
		model.cuda()

# Define loss function:
loss_function = nn.MSELoss()
		
# Initialize optimizer & scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 20, gamma = 0.1)

# Initialize early stopping params
early_stopping = EarlyStopping(patience=patience)

if __name__ == "__main__":
	
	import argparse
	parser = argparse.ArgumentParser(description="AutoEncoder for WORLD vocoder features")
	mode = parser.add_mutually_exclusive_group()
	mode.add_argument("-t", "--train", action="store_true", help="train autoencoder")
	mode.add_argument("-e", "--encode", action="store_true", help="use encoder to store bottleneck features")
	mode.add_argument("-d", "--decode", action="store_true", help="use decoder on stored bottleneck features")
	parser.add_argument("-f", "--data", type=str, default='data', help="folder with .csv data to read/write from")
	parser.add_argument("-m", "--model", type=str, default='.', help="folder with trained autoencoder .pth file")
	args = parser.parse_args()

	# ======================data========================
	train, val = .9, .1
	data_dir = os.path.join(os.getcwd(), args.data)							# User-specified folder with .csv files
	model_dir = os.path.join(os.getcwd(), args.model, 'autoencoder.pth')	# User-specified folder to .pth file
	csv_files = [os.path.join(data_dir, i) for i in os.listdir(data_dir)]	# File paths to batched .csvs with data
	datasets = list(map(lambda x: Dataset(x, transform=True), csv_files))	# Compile list of datasets inits from .csvs
	dataset = ConcatDataset(datasets) 										# Compile concatenated dataset
	dataset_size = len(dataset)												# Get total number of dataset samples
	idx = list(range(dataset_size))											# Create list of indices for dataset
	np.random.shuffle(idx)													# Randomly shuffle
	split = int(np.floor(val * dataset_size))								# Get size of validation data
	train_idx, val_idx = idx[split:], idx[:split]							# Assign training and validation indices 
	train_sampler = SubsetRandomSampler(train_idx)							# Sampler for training batches
	val_sampler = SubsetRandomSampler(val_idx)								# Sampler for validation batches
	# ==================================================
	
	if args.train:
		avg_train_loss = []		# store average training loss per epoch for plotting
		avg_val_loss = []		# store average validation loss per epoch for plotting
		train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler = train_sampler) # Online read of sampled minibatched training data
		val_dataloader = DataLoader(dataset, batch_size=batch_size, sampler = val_sampler) # Online read of sampled minibatched validation data
		train_model()
		torch.save(model.state_dict(), model_dir) # Save model if number of epochs have run to zero
		with open('loss_avgs.pkl', 'wb') as losses:
			loss_avgs = pkl.dump([avg_train_loss, avg_val_loss], losses) # Save train/val losses for later plotting
		
	if args.encode or args.decode:
	
		# ======================data========================
		dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)	# Online read of unshuffled minibatched data
		model.load_state_dict(torch.load(model_dir))							# Load model from trained autoencoder
		model.eval()															# Set model to evaluation before running inference
		# ==================================================
		
		if args.encode:
			encode_bn_reps()
		
		if args.decode:
			decode_bn_reps()
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import _pickle as pkl

from braindecode.torch_ext.util import set_random_seeds
from braindecode.datautil.signal_target import SignalAndTarget
from braindecode.torch_ext.optimizers import AdamW

import logging
import importlib
importlib.reload(logging) # see https://stackoverflow.com/a/21475297/1469195
log = logging.getLogger()
log.setLevel('INFO')
import sys

logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
					 level=logging.INFO, stream=sys.stdout)

##########################################BEGIN MODEL###############################################

from torch.nn import init
from torch.nn.functional import elu

from braindecode.models.base import BaseModel
from braindecode.torch_ext.modules import Expression, AvgPool2dWithConv
from braindecode.torch_ext.functions import identity
from braindecode.torch_ext.util import np_to_var
from braindecode.torch_ext.util import var_to_np


class Deep4Net(BaseModel):
	"""
	Deep ConvNet model from [1]_.
	References
	----------
	.. [1] Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J., 
	   Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F. & Ball, T. (2017).
	   Deep learning with convolutional neural networks for EEG decoding and
	   visualization.
	   Human Brain Mapping , Aug. 2017. Online: http://dx.doi.org/10.1002/hbm.23730
	"""

	def __init__(
		self,
		in_chans,
		n_classes,
		input_time_length,
		final_conv_length,
		n_filters_time=25,
		n_filters_spat=25,
		filter_time_length=10,
		pool_time_length=3,
		pool_time_stride=3,
		n_filters_2=50,
		filter_length_2=10,
		n_filters_3=100,
		filter_length_3=10,
		n_filters_4=200,
		filter_length_4=10,
		first_nonlin=elu,
		first_pool_mode="max",
		first_pool_nonlin=identity,
		later_nonlin=elu,
		later_pool_mode="max",
		later_pool_nonlin=identity,
		drop_prob=0.5,
		double_time_convs=False,
		split_first_layer=True,
		batch_norm=True,
		batch_norm_alpha=0.1,
		stride_before_pool=False,
	):
		if final_conv_length == "auto":
			assert input_time_length is not None

		self.__dict__.update(locals())
		del self.self

	def create_network(self):
		if self.stride_before_pool:
			conv_stride = self.pool_time_stride
			pool_stride = 1
		else:
			conv_stride = 1
			pool_stride = self.pool_time_stride
		pool_class_dict = dict(max=nn.MaxPool2d, mean=AvgPool2dWithConv)
		first_pool_class = pool_class_dict[self.first_pool_mode]
		later_pool_class = pool_class_dict[self.later_pool_mode]
		model = nn.Sequential()
		if self.split_first_layer:
			model.add_module("dimshuffle", Expression(_transpose_time_to_spat))
			model.add_module(
				"conv_time",
				nn.Conv2d(
					1,
					self.n_filters_time,
					(self.filter_time_length, 1),
					stride=1,
				),
			)
			model.add_module(
				"conv_spat",
				nn.Conv2d(
					self.n_filters_time,
					self.n_filters_spat,
					(1, self.in_chans),
					stride=(conv_stride, 1),
					bias=not self.batch_norm,
				),
			)
			n_filters_conv = self.n_filters_spat
		else:
			model.add_module(
				"conv_time",
				nn.Conv2d(
					self.in_chans,
					self.n_filters_time,
					(self.filter_time_length, 1),
					stride=(conv_stride, 1),
					bias=not self.batch_norm,
				),
			)
			n_filters_conv = self.n_filters_time
		if self.batch_norm:
			model.add_module(
				"bnorm",
				nn.BatchNorm2d(
					n_filters_conv,
					momentum=self.batch_norm_alpha,
					affine=True,
					eps=1e-5,
				),
			)
		model.add_module("conv_nonlin", Expression(self.first_nonlin))
		model.add_module(
			"pool",
			first_pool_class(
				kernel_size=(self.pool_time_length, 1), stride=(pool_stride, 1)
			),
		)
		model.add_module("pool_nonlin", Expression(self.first_pool_nonlin))

		def add_conv_pool_block(
			model, n_filters_before, n_filters, filter_length, block_nr
		):
			suffix = "_{:d}".format(block_nr)
			model.add_module("drop" + suffix, nn.Dropout(p=self.drop_prob))
			model.add_module(
				"conv" + suffix,
				nn.Conv2d(
					n_filters_before,
					n_filters,
					(filter_length, 1),
					stride=(conv_stride, 1),
					bias=not self.batch_norm,
				),
			)
			if self.batch_norm:
				model.add_module(
					"bnorm" + suffix,
					nn.BatchNorm2d(
						n_filters,
						momentum=self.batch_norm_alpha,
						affine=True,
						eps=1e-5,
					),
				)
			model.add_module("nonlin" + suffix, Expression(self.later_nonlin))

			model.add_module(
				"pool" + suffix,
				later_pool_class(
					kernel_size=(self.pool_time_length, 1),
					stride=(pool_stride, 1),
				),
			)
			model.add_module(
				"pool_nonlin" + suffix, Expression(self.later_pool_nonlin)
			)

		add_conv_pool_block(
			model, n_filters_conv, self.n_filters_2, self.filter_length_2, 2
		)
		add_conv_pool_block(
			model, self.n_filters_2, self.n_filters_3, self.filter_length_3, 3
		)
		add_conv_pool_block(
			model, self.n_filters_3, self.n_filters_4, self.filter_length_4, 4
		)

		# model.add_module('drop_classifier', nn.Dropout(p=self.drop_prob))
		model.eval()
		if self.final_conv_length == "auto":
			out = model(
				np_to_var(
					np.ones(
						(1, self.in_chans, self.input_time_length, 1),
						dtype=np.float32,
					)
				)
			)
			n_out_time = out.cpu().data.numpy().shape[2]
			self.final_conv_length = n_out_time
		model.add_module(
			"conv_classifier",
			nn.Conv2d(
				self.n_filters_4,
				self.n_classes,
				(self.final_conv_length, 1),
				bias=True,
			),
		)
		model.add_module("softmax", nn.LogSoftmax(dim=1))
		model.add_module("squeeze", Expression(_squeeze_final_output))

		# Initialization, xavier is same as in our paper...
		# was default from lasagne
		init.xavier_uniform_(model.conv_time.weight, gain=1)
		# maybe no bias in case of no split layer and batch norm
		if self.split_first_layer or (not self.batch_norm):
			init.constant_(model.conv_time.bias, 0)
		if self.split_first_layer:
			init.xavier_uniform_(model.conv_spat.weight, gain=1)
			if not self.batch_norm:
				init.constant_(model.conv_spat.bias, 0)
		if self.batch_norm:
			init.constant_(model.bnorm.weight, 1)
			init.constant_(model.bnorm.bias, 0)
		param_dict = dict(list(model.named_parameters()))
		for block_nr in range(2, 5):
			conv_weight = param_dict["conv_{:d}.weight".format(block_nr)]
			init.xavier_uniform_(conv_weight, gain=1)
			if not self.batch_norm:
				conv_bias = param_dict["conv_{:d}.bias".format(block_nr)]
				init.constant_(conv_bias, 0)
			else:
				bnorm_weight = param_dict["bnorm_{:d}.weight".format(block_nr)]
				bnorm_bias = param_dict["bnorm_{:d}.bias".format(block_nr)]
				init.constant_(bnorm_weight, 1)
				init.constant_(bnorm_bias, 0)

		init.xavier_uniform_(model.conv_classifier.weight, gain=1)
		init.constant_(model.conv_classifier.bias, 0)

		# Start in eval mode
		model.eval()
		return model


# remove empty dim at end and potentially remove empty time dim
# do not just use squeeze as we never want to remove first dim
def _squeeze_final_output(x):
	assert x.size()[3] == 1
	x = x[:, :, :, 0]
	if x.size()[2] == 1:
		x = x[:, :, 0]
	return x


def _transpose_time_to_spat(x):
	return x.permute(0, 3, 2, 1)

############################################END MODEL###############################################

def unpkl(file):
	with open(file, 'rb') as foo:
		dict = pkl.load(foo, encoding='bytes')
	return dict

data = unpkl('data')
labels = data[b'labels']
data = data[b'data']
samples = np.shape(data)[0]

random_sampling = np.random.choice(samples,samples,replace=False)

train, test, val = 80, 10, 10

train = int(((samples/100) * train))
test = int(((samples/100) * test))
val = int(((samples/100) * val))

while train+test+val < samples:
	train = train + 1

X_train = np.array([data[i] for i in random_sampling[0:train]]).astype(np.float32)
X_test = np.array([data[i] for i in random_sampling[train:train+test]]).astype(np.float32)
X_val = np.array([data[i] for i in random_sampling[-val:]]).astype(np.float32)
y_train = np.array([labels[i] for i in random_sampling[0:train]]).astype(np.int64)
y_test = np.array([labels[i] for i in random_sampling[train:train+test]]).astype(np.int64)
y_val = np.array([labels[i] for i in random_sampling[-val:]]).astype(np.int64)

train_set = SignalAndTarget(X_train, y=y_train)
valid_set = SignalAndTarget(X_val, y=y_val)
test_set = SignalAndTarget(X_test, y=y_test)

# Set if you want to use GPU
cuda = True if torch.cuda.is_available() else False

set_random_seeds(seed=20170629, cuda=cuda)
in_chans = train_set.X.shape[1]
input_time_length = 450 # This will determine how many crops are processed in parallel.
# train_set.X.shape[0] = no. of trials
# train_set.X.shape[0] = no. of channels
# train_set.X.shape[2] = no. of samples (timesteps)
# train_set.X.shape is (557, 62, 4839) for raw KARA One data
# final_conv_length determines the size of the receptive field of the ConvNet
model = Deep4Net(in_chans=in_chans,
				 n_classes=2,
				 input_time_length=input_time_length,
				 final_conv_length=1,
				 n_filters_time=25,
				 n_filters_spat=25,
				 filter_time_length=10,
				 pool_time_length=3,
				 pool_time_stride=2,
				 n_filters_2=50,
				 filter_length_2=10,
				 n_filters_3=100,
				 filter_length_3=5,
				 n_filters_4=200,
				 filter_length_4=5,
				 first_nonlin=elu,
				 first_pool_mode="max",
				 first_pool_nonlin=identity,
				 later_nonlin=elu,
				 later_pool_mode="max",
				 later_pool_nonlin=identity,
				 drop_prob=0.5,
				 double_time_convs=False,
				 split_first_layer=True,
				 batch_norm=True,
				 batch_norm_alpha=0.1,
				 stride_before_pool=True)
if cuda:
	model.cuda()
	
optimizer = AdamW(model.parameters(), lr=1*0.01, weight_decay=0.5*0.001) # these are good values for the deep model
model.compile(loss=F.nll_loss, optimizer=optimizer,  iterator_seed=1, cropped=True)

model.fit(train_set.X, train_set.y, epochs=30, batch_size=64, scheduler='cosine',
		  input_time_length=input_time_length,
		  validation_data=(valid_set.X, valid_set.y),)
		  
###################COMPUTE CORRELATION: AMPLITUDE PURTURBATION - PREDICTION CHANGE###################

# Collect all batches and concatenate them into one array of examples:

from braindecode.datautil.iterators import CropsFromTrialsIterator

test_input = np_to_var(np.ones((2, in_chans, input_time_length, 1), dtype=np.float32))
if cuda:
	test_input = test_input.cuda()
out = model.network(test_input)
n_preds_per_input = out.cpu().data.numpy().shape[2]
iterator = CropsFromTrialsIterator(batch_size=32,input_time_length=input_time_length,
								  n_preds_per_input=n_preds_per_input)

train_batches = list(iterator.get_batches(train_set, shuffle=False))
train_X_batches = np.concatenate(list(zip(*train_batches))[0])

# Create a prediction function that wraps the model prediction function and returns the predictions as numpy arrays:
# We use the predition before the softmax, so we create a new module with all the layers of the old until before the softmax.

new_model = nn.Sequential()
for name, module in model.network.named_children():
	if name == 'softmax': break
	new_model.add_module(name, module)

new_model.eval();
pred_fn = lambda x: var_to_np(torch.mean(new_model(np_to_var(x).cuda() if cuda else np_to_var(x))[:,:,:,0], dim=2, keepdim=False))

from braindecode.visualization.perturbation import compute_amplitude_prediction_correlations

amp_pred_corrs = compute_amplitude_prediction_correlations(pred_fn, train_X_batches, n_iterations=12,
										 batch_size=30)

# Pick out one frequency range and mean correlations within that frequency range to make a scalp plot.
# Here use use the alpha frequency range.

print(amp_pred_corrs.shape)

fs = 1000.0 # Emotive EPOC+ = Sampling rate: 2048 internal downsampled to 128 SPS or 256 SPS (user configured)
freqs = np.fft.rfftfreq(train_X_batches.shape[2], d=1.0/fs)

delta = (0.5, 4)
theta = (4, 8)
alpha = (8, 13)
beta = (13, 30)
gamma = (30, 100)
start_freq = lambda x: x[0]
stop_freq = lambda x: x[1]

i_start = lambda x: np.searchsorted(freqs, start_freq(x))
i_stop = lambda x: np.searchsorted(freqs, stop_freq(x)) + 1

freq_corr = lambda x: np.mean(amp_pred_corrs[:,i_start(x):i_stop(x)], axis=1)

# Now get approximate positions of the channels in the 10-20 system:

from braindecode.datasets.sensor_positions import get_channelpos, CHANNEL_10_20_APPROX

ch_names_kara = [
'Fp1','Fpz','Fp2','Af3','Af4','F7','F5','F3','F1','Fz',
'F2','F4','F6','F8','Ft7','Fc5','Fc3','Fc1','Fcz','Fc2',
'Fc4','Fc6','Ft8','T7','C5','C3','C1','Cz','C2','C4',
'C6','T8','Tp7','Cp5','Cp3','Cp1','Cpz','Cp2','Cp4','Cp6',
'Tp8','P7','P5','P3','P1','Pz','P2','P4','P6','P8',
'Po7','Po5','Po3','Poz','Po4','Po6','Po8','P9','Oz','O2',
'P10','O1'] # Cb1 and Cb2 replaced with P9 and P10 respectively

ch_names_epoc = ['Af3','F7','F3','Fc5','T7','P7','O1','O2','P8','T8','Fc6','F4','F8','Af4']
                                                                                        
positions = lambda x: np.array([get_channelpos(name, CHANNEL_10_20_APPROX) for name in x])

# For plotting accuracies over time:

labels_per_trial_per_crop = model.predict_classes(test_set.X, individual_crops=True)
accs_per_crop = np.mean([l == y for l,y in zip(labels_per_trial_per_crop, test_set.y)], axis=0)
cropped_outs = model.predict_outs(test_set.X, individual_crops=True)

##################################RESULTS DISPLAY FOR PUBLICATION###################################

with open('Deep4Net.pkl', 'wb') as f:
	d = {}
	d['epochs_df'] = model.epochs_df # Display monitored values as pandas dataframe
	d['evaluate'] = model.evaluate(test_set.X, test_set.y) # Evaluate the accuracies to report in our publication by evaluating on the test set
	d['predict_classes'] = model.predict_classes(test_set.X) # Retrieve predicted labels per trial
	d['predict_outs'] = model.predict_outs(test_set.X) # Retrieve the probabilities per trial
	d['plot_data'] = {
	'accs_per_crop':accs_per_crop,
	'cropped_outs':cropped_outs,
	'test_set_labels':test_set.y,
	'freq_corr':{i:freq_corr(j) for i,j in [('delta',delta),('theta',theta),('alpha',alpha),('beta',beta),('gamma',gamma)]},
	'ch_names_kara':ch_names_kara,
	'ch_names_epoc':ch_names_epoc,
	'positions_kara':positions(ch_names_kara),
	'positions_epoc':positions(ch_names_epoc)
	}
	pkl.dump(d, f)
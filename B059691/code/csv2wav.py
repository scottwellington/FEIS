#!/usr/bin/env python
# -*- coding: utf-8 -*-
## Project: SCRIPT - February 2018 
## Contact: Oliver Watts - owatts@staffmail.ed.ac.uk

# world_features.py (below) from Oliver Watts (above) as-is, with minor changes on lines 57 & 63

import pandas as pd

################################### start world_features.py ########################################

import sys
import os

import numpy as np
from scipy import interpolate

import soundfile

import pyworld
import pysptk

#import librosa

import pylab
import scipy

from numpy import convolve
import matplotlib.pyplot as plt

# def librosa_features(wavefile):
#	 waveform, samplerate = soundfile.read(wavefile)
#	 linear = librosa.stft(y=waveform, n_fft=1024,
#						   hop_length=int(samplerate * 0.005),
#						   win_length=1024)

#	 # # magnitude spectrogram
#	 mag = np.abs(linear)  # (1+n_fft//2, T)
#	 return mag.T ## convert to (time, feats) orientation


def extract_world_features(wavefile):
	waveform, samplerate = soundfile.read(wavefile)
	f0, sp, ap = pyworld.wav2world(waveform, samplerate, frame_period=5) # use default options
	f0 = f0.reshape(-1,1)  
	return (f0, sp, ap, samplerate)


def sp2mgc(sp, dim, sr):
	return pysptk.mcep(sp, order=dim-1, alpha=get_world_alpha(sr), \
				maxiter=0, etype=1, eps=0.0, min_det=1e-06, itype=4)

def mgc2sp(mgc, sr):
	log_spec = pysptk.mgc2sp(mgc, alpha=get_world_alpha(sr), gamma=0, \
			fftlen=get_world_fftlen(sr)).real
	return np.exp(log_spec * 2)


### functions to get values dependent on sample rate consistent with world
def get_world_fftlen(sr):
	return {16000: 1024, 22050: 1024, 44100: 2048, 48000: 2048}[sr]

def get_world_alpha(sr):
	return {16000: 0.58, 22050: 0.65, 44100: 0.76, 48000: 0.77}[sr]


### functions to mimic's world's internal conversion between coarse 
### aperiodicity and fine aperiodicity spectrum
def get_world_freq_axis(sr, fft_size=1024):
	frequency_axis = []
	for i in range(int((fft_size / 2) + 1)): # modified to have int()
		frequency_axis.append(float(i) * sr / fft_size)
	return frequency_axis

def get_world_coarse_freq_axis(sr):
	kFrequencyInterval = 3000 ## const
	number_of_aperiodicities = {16000: 1, 22050:1, 44100:5, 48000:5}[sr] # included 22050:1
	coarse_frequency_axis = []
	for i in range(number_of_aperiodicities+1):
		coarse_frequency_axis.append( i * kFrequencyInterval )
	coarse_frequency_axis.append(sr / 2)
	return coarse_frequency_axis

def ap2coarse(ap, sr):
	m,n = ap.shape
	fft_size = (n-1) * 2
	coarse_axis = get_world_coarse_freq_axis(sr)
	freq_axis = get_world_freq_axis(sr, fft_size)
	f = interpolate.interp1d(freq_axis, ap)  ## this seems like overkill...
	coarse_ap = f(coarse_axis)
	return coarse_ap[:, 1:-1] ## throw out constant values in first and last bins

def coarse2ap(coarse, sr, fft_size):
	coarse_axis = get_world_coarse_freq_axis(sr)
	freq_axis = get_world_freq_axis(sr, fft_size)

	m,n = coarse.shape
	# add back in constant values in first and last bins
	padded_coarse = np.zeros((m,n+2))
	padded_coarse[:,1:-1] = coarse
	padded_coarse[:,0] = -60.0   ## hardcoded in world
	kMySafeGuardMinimum = 0.000000000001  ## constant in world code
	padded_coarse[:,-1] = -kMySafeGuardMinimum   ## hardcoded in world

	f = interpolate.interp1d(coarse_axis, padded_coarse)  ## this seems like overkill, but...
	ap = f(freq_axis)
	return ap

empty_array = np.zeros((0,0)) # TODO: const
def interpolate_through_unvoiced(data, vuv=empty_array):

	assert len(data.shape) == 2, 'interpolate_through_unvoiced only accepts 2D arrays'
	if vuv.size == empty_array.size:
		assert data.shape[1] == 1, 'To find voicing from the data itself, use data with only a single channel'
		voiced_ix = np.where( data > 0.0 )[0]  ## equiv to np.nonzero(y)
	else:
		voiced_ix = np.where( vuv > 0.0 )[0]
	mean_voiced = data[voiced_ix, ...].mean(axis=0)  ## using fill_value='extrapolate' creates very extreme values where there are long initial/final silences
		   ### TODO: this seems to affect denormalisation rather than training, look at extracintg stats and even training without regard to interpolated values?
	interpolator = scipy.interpolate.interp1d(voiced_ix, data[voiced_ix, ...], kind='linear', \
												axis=0, bounds_error=False, fill_value=mean_voiced)
	data_interpolated = interpolator(np.arange(data.shape[0])) # .reshape((-1,1)))

	voicing_flag = np.zeros((data.shape[0], 1))
	voicing_flag[voiced_ix] = 1.0

	return (data_interpolated, voicing_flag)

##################################### end world_features.py ########################################

sr = 16000

csvs = [os.path.splitext(i)[0] for i in os.listdir('.') if os.path.splitext(i)[1] == '.csv']

for i in csvs:

	a = pd.read_csv('{}.csv'.format(i), header=None)

	bap = np.ascontiguousarray(a.iloc[:,514].values).astype(np.float)
	bap = bap[..., np.newaxis]
	ap = coarse2ap(bap, sr, get_world_fftlen(sr))
	ap = np.ascontiguousarray(ap)

	f0 = np.exp(a.iloc[:,515].values).astype(np.float)
	#vuv = a.iloc[:,516].values
	#f0[a.iloc[:,516].values < 0.5] = 0.0
	f0 = f0.flatten()

	sp = np.ascontiguousarray(a.iloc[:,1:514]).astype(np.float)

	y = pyworld.synthesize(f0, sp, ap, sr, 5)
	soundfile.write('{}.wav'.format(i), y, sr)
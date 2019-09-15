#!/usr/bin/env python
# -*- coding: utf-8 -*-
## Project: SCRIPT - February 2018 
## Contact: Oliver Watts - owatts@staffmail.ed.ac.uk

# world_features.py (below) from Oliver Watts (above) as-is, with minor changes on lines 57 & 63

################################### start world_features.py ########################################

import sys
import os
from os.path import isfile, join

import numpy as np
from scipy import interpolate

import soundfile

import pyworld
import pysptk

import pylab
import scipy

from numpy import convolve
import matplotlib.pyplot as plt

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

import csv
import warnings
warnings.filterwarnings("ignore") # ignore interpolate_through_unvoiced() errors

def wav2csv(wavs):
	size = len(wavs)
	count = 0
	samples = 0
	for wav in wavs:
		if os.path.getsize(wav) == 44: # Hardcode remove files that SoX deemed to be all silence (8)
			os.remove(wav)
		else:
			with open('{}.csv'.format(os.path.join(out_dir, os.path.basename(wav)[:-4])), 'w') as f:
				wr = csv.writer(f, lineterminator = '\n')
				f0, sp, ap, sr = extract_world_features(wav)
				try:
					interp_f0, vuv = interpolate_through_unvoiced(f0)
				except ValueError:
					vuv = np.zeros((f0.shape[0], 1))
				for i in range(0,len(sp)):
					label = np.ascontiguousarray(os.path.basename(wav)[:-4] + '_' + str(i))
					temp = np.ascontiguousarray(np.mean(ap[i]))  # Confirmed with Oliver that this is sensible
					wr.writerow(np.concatenate((label, sp[i], temp, f0[i], vuv[i])))
				count += 1
				samples = samples + len(sp)
			print("done {}.csv [{}/{}]".format(os.path.basename(wav)[:-4],count,size))
	print("\nProcessed {} samples of length 513+1+1+1 (sp,ap,f0,vuv) from {} sound files".format(samples, size))

in_dir = '.'
out_dir = '.'

wavs = [os.path.realpath(f) for f in os.listdir(in_dir) if os.path.splitext(f)[1] == '.wav']
wav2csv(wavs)
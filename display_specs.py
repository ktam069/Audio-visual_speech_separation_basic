# Test program - Displays some mixed spectrograms (i.e. data used for training)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, LSTM, Bidirectional
from keras.layers import BatchNormalization, Activation, Flatten, TimeDistributed, Reshape
from keras.callbacks import ModelCheckpoint

import os
# import sys

# ===== Settings =====

# Display options
PRINT_DATA = False
DISPLAY_GRAPHS = False

# Range of data to use for training (excludes END_ID)
START_ID = 0
END_ID = 5

# Sampling rate
SAMPLING_RATE = 16000

# Filepaths to the locations for saved data and models

path_to_data = "./data/"
path_to_models = "./saved_models/"

path_to_data_audio = "./data/audio/audio_train/"
path_to_data_video = "./data/video/face_input/"

# audio_data_name = "trim_audio_train%d.wav"
# dataset_filepath = "./data_train/"+"dataset_train_%d-%d.npy"%(START_ID,END_ID-1)

# ====================

def load_data():
	audio_dataset = []
	print("Loading data files %d to %d..."%(START_ID,END_ID-1))
		
	for data_num in range(START_ID, END_ID):
		print("\tLoading audio data file %d..."%data_num, end='\r', flush=True)
		
		# Load wav file into an array
		data = load_wav(data_num)
		if data is None:
			continue
		
		# Convert the array into a spectrogram - for visualisation only
		if DISPLAY_GRAPHS:
			data_tmp = wav_to_spectrogram(data)
			visualise_data(data_tmp)
		
		# Add data point to clean audio data set
		audio_dataset.append(data)
	
	print()
	return audio_dataset

def generate_dataset(clean_audio_dataset):
	# Generate a data set of mixed audio inputs (x) and corresponding clean audio outputs (y)
	dataset = []
	
	for i in range(len(clean_audio_dataset)):
		for j in range(i+1, len(clean_audio_dataset)):
			# Mix audio for 2 speakers (no noise)
			mixed_audio = clean_audio_dataset[i] + clean_audio_dataset[j]
			
			# Add data point to data set
			dataset.append([mixed_audio, clean_audio_dataset[i], clean_audio_dataset[j]])
	
	return np.array(dataset)

def dataset_to_spectrograms(dataset):
	dataset_specs = np.zeros((dataset.shape[0], 298, 257, 2, dataset.shape[1]))
	
	for i in range(len(dataset)):
		print("\tConverting audio data %d/%d to spectrogram..."%(i+1,len(dataset)), end='\r', flush=True)
		
		w = []
		# Convert the data array into a spectrogram
		for j in range(dataset.shape[1]):
			s = wav_to_spectrogram(dataset[i,j,:])
			w.append(s)
			dataset_specs[i,:,:,:,j] = s
		
		visualise_data_mixture([w[1], w[2]], w[0])
	
	print()
	return dataset_specs

def load_wav(data_num):
	try:
		if PRINT_DATA:
			print("\n\n=== Loading audio file "+str(data_num)+" ===")
		
		# Filepath of audio file to be loaded
		path = path_to_data_audio + "trim_audio_train%d.wav"%data_num
		
		# Load wav file into an array (an ndarray)
		data, _ = librosa.load(path, sr=SAMPLING_RATE)
		
		if PRINT_DATA:
			print("\nData after loading:")
			print(data.shape)
			print(data)
	except:
		return None
	
	return data

def wav_to_spectrogram(data):
	
	# == STFT ==

	# Data padding??...may not be needed (TODO)
	length = len(data)
	# new_power_base_2 = np.ceil(np.log(length)/np.log(2))
	# new_len = pow(2, int(new_power_base_2))
	new_len = 48192 - 512
	
	if new_len > length:
		new_data = np.zeros(new_len)
		new_data[:len(data)] = data
	else:
		new_data = np.zeros(new_len)
		new_data[:] = data[:new_len]
	data = new_data.astype('float32')
		
	# Calculations taken from https://github.com/tensorflow/tensorflow/issues/24620
	sample_rate = 16000 #16kHz
	window_size_ms = 25
	window_stride_ms = 10
	window_size_samples = int(sample_rate * window_size_ms / 1000)
	window_stride_samples = int(sample_rate * window_stride_ms / 1000)
	
	window_size_samples = 512
	
	if PRINT_DATA:
		print("segment_size_samples", len(data))
		print("window_size_samples", window_size_samples)
		print("window_stride_samples", window_stride_samples)
	
	data = tf.signal.stft(data, 
						   frame_length=window_size_samples,
						   frame_step=window_stride_samples,
						   fft_length=window_size_samples,
						   pad_end=True)
						   # pad_end=False)
	
	# data = tf.signal.stft(data, frame_length=512, frame_step=160, fft_length=512,
				# window_fn=tf.signal.hann_window, pad_end=False)
		
	data = tf.Session().run(data)
	
	if PRINT_DATA:
		print("\nData after STFT:")
		print(data.shape)
		print(data)
	
	if DISPLAY_GRAPHS:
		# Plot spectrogram data
		plt.pcolormesh(np.abs(data.T))
		plt.title('STFT Magnitude')
		plt.ylabel('Frequency [Hz]')
		plt.xlabel('Time [sec]')
		# plt.show()
	
	# == Complex to Re/Im ==
	
	data = convert_to_scalars(data)
	
	# == Power Law Compression ==
	
	# TODO: test this
	
	# data = power_law_encode(data)
	# data = power_law_decode(data)
	
	if PRINT_DATA:
		print("\nData after conversion:")
		print(data.shape)
		print(data)
	
	return data

def convert_to_scalars(data):
	# Convert complex arrays into scalar components for Re and Im parts
	# i.e. from [[ a+ib,... ]...] to [[ [a,b],... ]...]
	
	s = data.shape
	new_s = (s[0],s[1],2)
	new_data = np.zeros(new_s)
	
	new_data[:,:,0] = data.real
	new_data[:,:,1] = data.imag
	
	return new_data

def convert_to_complex(data):
	# Convert scalar components for Re and Im parts into complex arrays
	# i.e. from [[ [a,b],... ]...] to [[ a+ib,... ]...]
	
	s = data.shape
	new_s = (s[0],s[1])
	new_data = np.zeros(new_s, dtype=complex)
	
	new_data[:,:] = data[:,:,0] + data[:,:,1]*1j
	
	return new_data

def power_law_encode(data, power=0.3):
	# Retain the original +ve/-ve signs
	signs = np.sign(data)
	compressed_data = np.power(np.abs(data), power)
	return compressed_data * signs

def power_law_decode(data, power=0.3):
	return power_law_encode(data, power=1.0/power)

def visualise_data(data):
	# Visualise real and imaginary data components
	
	plt.close()
	
	# Plot display settings
	plt.figure(figsize=(20, 10))
	plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.2, hspace=0.2)

	x = lambda a: 'Real' if a==0 else 'Imaginary'
	for i in range(2):
		plt.subplot(1, 3, i+1)
		
		plt.pcolormesh(np.abs(data[:,:,i].T))
		plt.title(x(i))
		plt.ylabel('Frequency [Hz]')
		plt.xlabel('Time [sec]')
		
	plt.subplot(1, 3, 3)
	
	plt.pcolormesh(np.abs(convert_to_complex(data).T))
	plt.title('Complex')
	plt.ylabel('Frequency [Hz]')
	plt.xlabel('Time [sec]')
	plt.show()

	
def visualise_data_mixture(data_list, data_mixed):	
	plt.close()
	
	# Plot display settings
	plt.figure(figsize=(20, 10))
	plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.2, hspace=0.2)

	x = lambda a: 'Part 1' if a==0 else 'Part 2'
	for i in range(2):
		plt.subplot(1, 3, i+1)
		
		spect = convert_to_complex(data_list[i])
		plt.pcolormesh(np.abs(spect.T))
		plt.title(x(i))
		plt.ylabel('Frequency [Hz]')
		plt.xlabel('Time [sec]')
		
	plt.subplot(1, 3, 3)
	
	plt.pcolormesh(np.abs(convert_to_complex(data_mixed).T))
	plt.title('Mixed')
	plt.ylabel('Frequency [Hz]')
	plt.xlabel('Time [sec]')
	plt.show()

def main():
	
	audio_wav = load_data()
	dataset_wav = generate_dataset(audio_wav)
	dataset_train = dataset_to_spectrograms(dataset_wav)
	
	# for i in range(dataset_train.shape[0]):
		# visualise_data()
		
		
if __name__ == '__main__':
	main()


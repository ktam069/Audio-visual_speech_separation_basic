# Main program - Part IV Project 80 Basic Version

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, LSTM, Bidirectional
from keras.layers import BatchNormalization, Activation, Flatten, TimeDistributed, Reshape

# import os
# import sys

# ===== Settings =====

# Filepaths to the locations for saved data and models

path_to_data = "./data/"
path_to_models = "./saved_models/"

path_to_data_audio = "./data/audio/audio_train/"
path_to_data_video = "./data/video/face_input/"

# audio_data_name = "trim_audio_train%d.wav"

# Plot display settings
plt.figure(figsize=(20, 10))
plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.2, hspace=0.2)

# ====================

def load_wav():
	# File number of audio data to be loaded
	data_num = 1
	
	# Filepath of audio file to be loaded
	path = path_to_data_audio + "trim_audio_train%d.wav"%data_num
	
	# Sampling rate
	sr = 16000
	
	# Load wav file into an array (an ndarray)
	data, _ = librosa.load(path, sr=sr)
	
	print("\nData after loading:")
	print(data.shape)
	print(data)
	
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
	
	print("\nData after STFT:")
	print(data.shape)
	print(data)
	
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
	plt.clf()
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

def train_model(data, num_speakers=2):
	# Create a compiled model
	model = convolution_model(data)
	
	# Process training data
	pass
	# x_train, y_train, x_test, y_test = 
	
	# Train the model
	model.fit(x_train, y_train, batch_size=6, epochs=1)		# TODO: adjust the arguments used

def convolution_model(data, num_speakers=2):
	assert data.shape == (298, 257, 2), "Please check if the input shape is correct"
	
	# == Audio convolution layers ==
	
	model = Sequential()
	
	# # Implicit input layer
	# inputs = Input(shape=(298, 257, 2))
	# model.add(inputs)
	
	# Convolution layers
	conv1 = Conv2D(96, kernel_size=(1,7), padding='same', dilation_rate=(1,1), input_shape=(298, 257, 2))
	model.add(conv1)
	model.add(BatchNormalization())
	model.add(Activation("relu"))
	
	conv2 = Conv2D(96, kernel_size=(7,1), padding='same', dilation_rate=(1,1))
	model.add(conv2)
	model.add(BatchNormalization())
	model.add(Activation("relu"))
	
	conv3 = Conv2D(96, kernel_size=(5,5), padding='same', dilation_rate=(1,1))
	model.add(conv3)
	model.add(BatchNormalization())
	model.add(Activation("relu"))
	
	conv4 = Conv2D(96, kernel_size=(5,5), padding='same', dilation_rate=(2,1))
	model.add(conv4)
	model.add(BatchNormalization())
	model.add(Activation("relu"))
	
	conv5 = Conv2D(96, kernel_size=(5,5), padding='same', dilation_rate=(4,1))
	model.add(conv5)
	model.add(BatchNormalization())
	model.add(Activation("relu"))
	
	conv6 = Conv2D(96, kernel_size=(5,5), padding='same', dilation_rate=(8,1))
	model.add(conv6)
	model.add(BatchNormalization())
	model.add(Activation("relu"))
	
	conv7 = Conv2D(96, kernel_size=(5,5), padding='same', dilation_rate=(16,1))
	model.add(conv7)
	model.add(BatchNormalization())
	model.add(Activation("relu"))
	
	conv8 = Conv2D(96, kernel_size=(5,5), padding='same', dilation_rate=(32,1))
	model.add(conv8)
	model.add(BatchNormalization())
	model.add(Activation("relu"))
	
	conv9 = Conv2D(96, kernel_size=(5,5), padding='same', dilation_rate=(1,1))
	model.add(conv9)
	model.add(BatchNormalization())
	model.add(Activation("relu"))
	
	conv10 = Conv2D(96, kernel_size=(5,5), padding='same', dilation_rate=(2,2))
	model.add(conv10)
	model.add(BatchNormalization())
	model.add(Activation("relu"))
	
	conv11 = Conv2D(96, kernel_size=(5,5), padding='same', dilation_rate=(4,4))
	model.add(conv11)
	model.add(BatchNormalization())
	model.add(Activation("relu"))
	
	conv12 = Conv2D(96, kernel_size=(5,5), padding='same', dilation_rate=(8,8))
	model.add(conv12)
	model.add(BatchNormalization())
	model.add(Activation("relu"))
	
	conv13 = Conv2D(96, kernel_size=(5,5), padding='same', dilation_rate=(16,16))
	model.add(conv13)
	model.add(BatchNormalization())
	model.add(Activation("relu"))
	
	conv14 = Conv2D(96, kernel_size=(5,5), padding='same', dilation_rate=(32,32))
	model.add(conv14)
	model.add(BatchNormalization())
	model.add(Activation("relu"))
	
	conv15 = Conv2D(8, kernel_size=(1,1), padding='same', dilation_rate=(1,1))
	model.add(conv15)
	model.add(BatchNormalization())
	model.add(Activation("relu"))
	
	# AV fusion step(s)
	model.add(TimeDistributed(Flatten()))
	
	# BLSTM
	new_matrix_length = 400
	model.add(Bidirectional(LSTM(new_matrix_length//2, return_sequences=True, input_shape=(298, 257*8))))
	
	# Fully connected layers
	model.add(Dense(600, activation="relu"))
	model.add(Dense(600, activation="relu"))
	model.add(Dense(600, activation="relu"))
	
	# Output layer (i.e. complex masks)
	outputs = Dense(257*2*num_speakers, activation="relu")
	model.add(outputs)
	outputs_complex_masks = Reshape((298, 257, 2, num_speakers))
	model.add(outputs_complex_masks)
	
	# Print the output shapes of each model layer
	for layer in model.layers:
		name = layer.get_config()["name"]
		if "batch_normal" in name or "activation" in name:
			continue
		print(layer.output_shape, "\t", name)
	
	# Compile the model before training
	model.compile(optimizer='adam', loss='mse')
	# model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
	
	return model

def main():
	
	data = load_wav()
	data = wav_to_spectrogram(data)
	
	# visualise_data(data)
	
	model = convolution_model(data)
	
	# train_model(data)
	
if __name__ == '__main__':
	main()


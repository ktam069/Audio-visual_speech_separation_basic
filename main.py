# Main program - Part IV Project 80 Basic Version

# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.io import wavfile

import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Input, Dense, Conv2D, LSTM, Bidirectional
from keras.layers import BatchNormalization, Activation, Flatten, TimeDistributed, Reshape
from keras.callbacks import ModelCheckpoint

import os
# import sys
import glob
from datetime import datetime

# ===== Settings =====

# Display options
PRINT_DATA = False
DISPLAY_GRAPHS = False

# Data transformation/processing
NORMALISE_DATA = False
POWER_ENCODE = True

# Use float32 format for dataset ndarrays
USE_FLOAT32 = True

# Range of data to use for training (excludes END_ID)
START_ID = 0
END_ID = 21

# Sampling rate
SAMPLING_RATE = 16000

# Filepaths to the locations for input audio-visual files
# path_to_data = "./data/"
path_to_data_audio = "./data/audio/audio_train/"
path_to_data_video = "./data/video/face_input/"
# audio_data_name = "trim_audio_train%d.wav"

# Filepaths to the locations for saved data and models
path_to_models = "./saved_models/"
path_to_saved_datasets = "./dataset_npy/"
path_to_outputs = "./output_wavs/"

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
	dataset_spects = np.zeros((dataset.shape[0], 298, 257, 2, dataset.shape[1]))
	
	# for i in range(len(dataset)):
		# print("\tConverting audio data %d/%d to spectrogram..."%(i+1,len(dataset)), end='\r', flush=True)
		
		# Convert the data array into a spectrogram
		# for j in range(dataset.shape[1]):
			# dataset_spects[i,:,:,:,j] = wav_to_spectrogram(dataset[i,j,:])
		
		# print(dataset_spects.shape, dataset.shape)
	
	print("Converting audio data to spectrograms...")
	dataset_spects = wav_to_spectrogram(dataset[:,:,:])
	# print(dataset_spects.shape, dataset.shape)
	
	# print()
	return dataset_spects

def dataset_labels_to_cRMs(dataset_spects):
	dataset_cRMs = np.zeros(dataset_spects.shape)
	
	print("Converting outputs to cRMs...")
		
	# print(dataset_spects.shape)
	
	# TODO: complete
	dataset_cRMs[:,:,:,:,0] = dataset_spects[:,:,:,:,0]
	dataset_cRMs[:,:,:,:,1] = cRM_encode(dataset_spects[:,:,:,:,0], dataset_spects[:,:,:,:,1])
	dataset_cRMs[:,:,:,:,2] = cRM_encode(dataset_spects[:,:,:,:,0], dataset_spects[:,:,:,:,2])
	
	return dataset_cRMs

def normalise_dataset(data):
	if type(data) is not np.ndarray:
		data = np.array(data)

	if PRINT_DATA:
		print("\nData before normalisation:")	
		print(data)
		print(np.max(data))
	
	max = np.max(data)
	if max > 0:
		data = data / max
	
	return data

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
	assert len(data.shape) == 3, "Unexpected shape for data input, should be (n_samples, n_speakers+1, t_samples)"
	
	# == STFT ==

	# Data padding??...may not be needed (TODO)
	length = data.shape[2]
	# new_power_base_2 = np.ceil(np.log(length)/np.log(2))
	# new_len = pow(2, int(new_power_base_2))
	new_len = 48192 - 512
	shape = data.shape
	new_shape = shape[:len(shape)-1] + (new_len,)
	
	if new_len > length:
		new_data = np.zeros(new_shape)
		new_data[:,:,:new_len] = data
	else:
		new_data = np.zeros(new_shape)
		new_data[:,:,:] = data[:,:,:new_len]
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
	
	# print(data.shape)
	data = np.moveaxis(data, 1, -1)
	# print(data.shape)
	
	if PRINT_DATA:
		print("\nData after STFT:")
		print(data.shape)
		# print(data)
	
	if DISPLAY_GRAPHS:
		# Plot spectrogram data
		plt.pcolormesh(np.abs(data.T))
		plt.title('STFT Magnitude')
		plt.ylabel('Frequency [Hz]')
		plt.xlabel('Time [sec]')
		# plt.show()
	
	# == Complex to Re/Im ==
	
	data = convert_to_scalars_ndarray(data)
	
	# == Power Law Compression ==
	
	# TODO: test this
	
	if POWER_ENCODE: data = power_law_encode(data)
	# data = power_law_decode(data)
	
	if PRINT_DATA:
		print("\nData after conversion:")
		print(data.shape)
		# print(data)
	
	return data

def convert_to_scalars_ndarray(data):
	# Convert complex arrays into scalar components for Re and Im parts
	# Allows extra dimension for processing multiple samples (compared to convert_to_scalars)
	
	new_s = data.shape + (2,)
	new_data = np.zeros(new_s)
	
	new_data[:,:,:,:,0] = data.real
	new_data[:,:,:,:,1] = data.imag
	
	new_data = np.moveaxis(new_data, -2, -1)
	
	return new_data

def convert_to_complex_ndarray(data):
	# Convert scalar components for Re and Im parts into complex arrays
	# Allows extra dimension for processing multiple samples (compared to convert_to_complex)
	
	# TODO: might need fixing (last two axes were in the wrong order - might not be fixed)
	data = np.moveaxis(data, -2, -1)
	
	new_s = data.shape[:-1]
	new_data = np.zeros(new_s, dtype=complex)
	
	new_data[:,:] = data[:,:,0] + data[:,:,1]*1j
	
	return new_data

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

def cRM_encode(S_spect, Y_spect):
	# Create a complex (ideal) ratio mask of the spectrogram input
	# S and Y are the STFT of the clean and noisy signals respectively
	
	output_mask = np.zeros(S_spect.shape)
	
	# Note: spectrogram shape is (298, 257, 2) - storing real and complex components
	S_spect_re = S_spect[:,:,:,0]
	S_spect_im = S_spect[:,:,:,1]
	Y_spect_re = Y_spect[:,:,:,0]
	Y_spect_im = Y_spect[:,:,:,1]
	
	# Use a small term (ep) to avoid division by zero
	ep = 1e-8
	denominator    = (Y_spect_re * Y_spect_re + Y_spect_im * Y_spect_im) + ep
	
	output_mask_re = (Y_spect_re * S_spect_re + Y_spect_im * S_spect_im) / denominator
	output_mask_im = (Y_spect_re * S_spect_im - Y_spect_im * S_spect_re) / denominator
	
	output_mask[:,:,:,0] = output_mask_re
	output_mask[:,:,:,1] = output_mask_im
	
	return output_mask

def cRM_decode(cRM_output, Y_spect):
	# (Apply complex mask to input spectrogram)
	# Multiply the complex ratio mask by the original noisy spectrogram to get the clean spectrogram
	# Needs to decompress the ratio mask beforehand
	
	output_spect = np.zeros(cRM_output.shape)
	
	cRM_re = cRM_output[:,:,:,0]
	cRM_im = cRM_output[:,:,:,1]
	Y_spect_re = Y_spect[:,:,:,0]
	Y_spect_im = Y_spect[:,:,:,1]
	
	# Multiplication of complex numbers: (a1+i*b1)(a2+i*b2) = (a1*a2-b1*b2) + i(a1*b1+a2*b2)
	output_spect_re = Y_spect_re * cRM_re - Y_spect_im * cRM_im
	output_spect_im = Y_spect_re * cRM_im + Y_spect_im * cRM_re
	
	output_spect[:,:,:,0] = output_spect_re
	output_spect[:,:,:,1] = output_spect_im
		
	return output_spect
	
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

def visualise_model_output(output_list, input_mixed):	
	plt.close()
	
	# Plot display settings
	plt.figure(figsize=(20, 10))
	plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.2, hspace=0.2)
	
	plt.subplot(1, 3, 1)
	
	plt.pcolormesh(np.abs(convert_to_complex(input_mixed).T))
	plt.title('Mixed Input')
	plt.ylabel('Frequency [Hz]')
	plt.xlabel('Time [sec]')

	x = lambda a: 'Part 1 Out' if a==0 else 'Part 2 Out'
	for i in range(2):
		plt.subplot(1, 3, i+2)
		
		spect = convert_to_complex(output_list[i])
		plt.pcolormesh(np.abs(spect.T))
		plt.title(x(i))
		plt.ylabel('Frequency [Hz]')
		plt.xlabel('Time [sec]')
	
	plt.show()

def train_model(x_train, y_train, num_speakers=2):
	assert x_train[0].shape == (298, 257, 2), "Data shape is incorrect - expected (298, 257, 2), got "+str(x_train[0].shape)
	
	# Create a compiled model
	model = convolution_model()
	
	# Process training data
	pass
	# x_train, y_train, x_test, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
	
	t = datetime.now().strftime("%d_%m_%H%M%S")
	
	# Create checkpoints when training model (save models to file)
	filepath = path_to_models + "basic-ao-%s-{epoch:02d}-{loss:.2f}.hdf5"%t
	checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='auto')
	# filepath = path_to_models + "basic-ao-%s-{epoch:02d}-{val_loss:.2f}.hdf5"%t
	# checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
	# checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=False)
	callback_list = [checkpoint]
	
	# Train the model
	# model.fit(x_train, y_train, batch_size=6, epochs=30, callbacks=callback_list, verbose=1)		# TODO: adjust the arguments used
	# model.fit(x_train, y_train, batch_size=6, epochs=20, callbacks=callback_list, verbose=1)		# TODO: adjust the arguments used
	model.fit(x_train, y_train, batch_size=6, epochs=20, callbacks=callback_list, verbose=1, validation_split=0.02)		# TODO: adjust the arguments used

def convolution_model(num_speakers=2):

	# == Audio convolution layers ==
	
	model = Sequential()
	
	# # Implicit input layer
	# inputs = Input(shape=(298, 257, 2))
	# model.add(inputs)
	
	# Convolution layers
	conv1 = Conv2D(96, kernel_size=(1,7), padding='same', dilation_rate=(1,1), input_shape=(298, 257, 2), name="input_layer")
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
	
	# == AV fused neural network ==
	
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
	# outputs = Dense(257*2*num_speakers, activation="relu")
	outputs = Dense(257*2*num_speakers, activation="sigmoid")				# TODO: check if this is more correct (based on the paper)
	model.add(outputs)
	outputs_complex_masks = Reshape((298, 257, 2, num_speakers), name="output_layer")
	model.add(outputs_complex_masks)
	
	# Print the output shapes of each model layer
	for layer in model.layers:
		name = layer.get_config()["name"]
		if "batch_normal" in name or "activation" in name:
			continue
		print(layer.output_shape, "\t", name)
	
	# Alternatively, print the default keras model summary
	print(model.summary())
	
	# Compile the model before training
	# model.compile(optimizer='adam', loss='mse')
	model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
	
	return model

def test_model(x_test, y_test):
	# Test using the newest available model
	all_model_paths = glob.glob(path_to_models+'*')
	newest_model_path = max(all_model_paths, key=os.path.getctime)
	
	# newest_model_path
	
	# Create a compiled model
	model = convolution_model()
	
	# Load weights from the last saved model
	model.load_weights(newest_model_path)
	
	# TODO / Temp: Save model for mobile append
	t = datetime.now().strftime("%d_%m_%H%M%S")
	model_save_path = path_to_models + "saved_model_basic_%s.h5"%t
	model.save(model_save_path)
	
	# TODO / Temp: Checking that the model loads
	# new_model = load_model(model_save_path)
	new_model = load_model(newest_model_path)
	print("\n" + "="*60 + "\n")
	print("Using model loaded from:", newest_model_path)
	print("\nLoaded model summary:")
	print(new_model.summary())
	
	print("\nEvaluating on test data...")
	
	# print(x_test.shape, y_test.shape)
	
	loss = model.evaluate(x_test, y_test)
	for i in range(len(model.metrics_names)):
		print(model.metrics_names[i]+":", loss[i])
	
	output_spects = new_model.predict(x_test)
	# print(type(output_spects))
	print(output_spects.shape)
	# print(output_spects)
	
	# ==================================================================================
	# Visualise the spectrograms for the first entry of the test set - testing purposes
	
	mixed_spect = x_test[0]
	
	# Predicted output cRM / spectrograms
	p1_spect = cRM_decode(output_spects[:,:,:,0], x_test)[0]
	p2_spect = cRM_decode(output_spects[:,:,:,1], x_test)[0]
	
	if DISPLAY_GRAPHS:
		# Display mixed input and predicted outputs
		visualise_model_output([p1_spect, p2_spect], mixed_spect)
	
	# Display actual clean spectrograms
	p1_spect_actual = cRM_decode(y_test[:,:,:,0], x_test)[0]
	p2_spect_actual = cRM_decode(y_test[:,:,:,1], x_test)[0]
	
	if DISPLAY_GRAPHS:
		# Display mixed input and original clean audio wavs
		visualise_model_output([p1_spect_actual, p2_spect_actual], mixed_spect)
	
	# Display cRMs
	if DISPLAY_GRAPHS:
		print("Displaying cRMs")
		p1_cRM = output_spects[0][:,:,:,0]
		p2_cRM = output_spects[0][:,:,:,1]
		p1_cRM_actual = y_test[0][:,:,:,0]
		p2_cRM_actual = y_test[0][:,:,:,1]
		visualise_model_output([p1_cRM, p2_cRM], mixed_spect)
		visualise_model_output([p1_cRM_actual, p2_cRM_actual], mixed_spect)
		
	# ==================================================================================
	
	# Convert separated speech spectrograms to wav files
	# output_spects = spectrogram_to_wav(output_spects)
	
	# wavfile.write(path_to_outputs+"output_file_%s_p1.wav"%t, SAMPLING_RATE, p1_spect)
	# wavfile.write(path_to_outputs+"output_file_%s_p2.wav"%t, SAMPLING_RATE, p2_spect)
	
	# print("Output file saved")
	

def main():
	
	# Generate data set to use for training
	filepath = path_to_saved_datasets+"dataset_train_%d-%d.npy"%(START_ID,END_ID-1)
	if not os.path.exists(filepath):
		# Load the data set of AV data, mix, and convert to spectrograms
		audio_wav = load_data()
		dataset_wav = generate_dataset(audio_wav)
		dataset_train = dataset_to_spectrograms(dataset_wav)
		dataset_train = dataset_labels_to_cRMs(dataset_train)
		if NORMALISE_DATA: dataset_train = normalise_dataset(dataset_train)
		print("Finished converting dataset to spectrograms and cRMs\n")
		
		if USE_FLOAT32:
			dataset_train = dataset_train.astype('float32')
		
		np.save(filepath, dataset_train)
	else:
		# Use existing generated data set
		print("Loading data set from file:", filepath)
		dataset_train = np.load(filepath)
	
	# Split dataset into inputs and ground truths
	x_train = dataset_train[:,:,:,:,0 ]
	y_train = dataset_train[:,:,:,:,1:]
	
	# # Build and train the neural network
	train_model(x_train, y_train)
	# train_model(x_train[:1], y_train[:1])
	
	# Test model - temporarily just using the training data to test for errors
	test_model(x_train[0:1], y_train[0:1])
	

if __name__ == '__main__':
	# Create any missing folders for saving data and outputs
	if not os.path.exists(path_to_models):
		os.mkdir(path_to_models)
	if not os.path.exists(path_to_saved_datasets):
		os.mkdir(path_to_saved_datasets)
	if not os.path.exists(path_to_outputs):
		os.mkdir(path_to_outputs)
	
	# Run AV speech separation program
	main()


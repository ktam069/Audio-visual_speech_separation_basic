# Main program - Part IV Project 80 Basic Version

# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.io import wavfile

import tensorflow as tf
from keras.models import Sequential, load_model, Model
from keras.layers import Input, Dense, Conv2D, LSTM, Bidirectional
from keras.layers import BatchNormalization, Activation, Flatten, TimeDistributed, Reshape
from keras.callbacks import ModelCheckpoint
from keras.layers import concatenate
from keras.layers import Lambda

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

def convolution_model(num_speakers=2):

	# == Audio convolution layers ==
	
	# Input layer for functional model
	audio_inputs = Input(shape=(298, 257, 2))
	
	# Convolution layers
	model = Conv2D(96, kernel_size=(1,7), padding='same', dilation_rate=(1,1))(audio_inputs)
	model = BatchNormalization()(model)
	model = Activation("relu")(model)
	
	model = Conv2D(96, kernel_size=(7,1), padding='same', dilation_rate=(1,1))(model)
	model = BatchNormalization()(model)
	model = Activation("relu")(model)
	
	model = Conv2D(96, kernel_size=(5,5), padding='same', dilation_rate=(1,1))(model)
	model = BatchNormalization()(model)
	model = Activation("relu")(model)
	
	model = Conv2D(96, kernel_size=(5,5), padding='same', dilation_rate=(2,1))(model)
	model = BatchNormalization()(model)
	model = Activation("relu")(model)
	
	model = Conv2D(96, kernel_size=(5,5), padding='same', dilation_rate=(4,1))(model)
	model = BatchNormalization()(model)
	model = Activation("relu")(model)
	
	model = Conv2D(96, kernel_size=(5,5), padding='same', dilation_rate=(8,1))(model)
	model = BatchNormalization()(model)
	model = Activation("relu")(model)
	
	model = Conv2D(96, kernel_size=(5,5), padding='same', dilation_rate=(16,1))(model)
	model = BatchNormalization()(model)
	model = Activation("relu")(model)
	
	model = Conv2D(96, kernel_size=(5,5), padding='same', dilation_rate=(32,1))(model)
	model = BatchNormalization()(model)
	model = Activation("relu")(model)
	
	model = Conv2D(96, kernel_size=(5,5), padding='same', dilation_rate=(1,1))(model)
	model = BatchNormalization()(model)
	model = Activation("relu")(model)
	
	model = Conv2D(96, kernel_size=(5,5), padding='same', dilation_rate=(2,2))(model)
	model = BatchNormalization()(model)
	model = Activation("relu")(model)
	
	model = Conv2D(96, kernel_size=(5,5), padding='same', dilation_rate=(4,4))(model)
	model = BatchNormalization()(model)
	model = Activation("relu")(model)
	
	model = Conv2D(96, kernel_size=(5,5), padding='same', dilation_rate=(8,8))(model)
	model = BatchNormalization()(model)
	model = Activation("relu")(model)
	
	model = Conv2D(96, kernel_size=(5,5), padding='same', dilation_rate=(16,16))(model)
	model = BatchNormalization()(model)
	model = Activation("relu")(model)
	
	model = Conv2D(96, kernel_size=(5,5), padding='same', dilation_rate=(32,32))(model)
	model = BatchNormalization()(model)
	model = Activation("relu")(model)
	
	model = Conv2D(8, kernel_size=(1,1), padding='same', dilation_rate=(1,1))(model)
	model = BatchNormalization()(model)
	model = Activation("relu")(model)
	
	audio_outputs = Reshape((298, 8*257))(model)
	
	audio_model = Model(audio_inputs, audio_outputs)

	# == Visual convolution layers ==
	
	visual_inputs_list = []
	visual_outputs_list = []
	
	for i in range(num_speakers):
		# Input layer for functional model
		# TODO: check the correctness of the input layer
		visual_inputs = Input(shape=(75, 1, 1024))
		
		# Convolution layers
		model = Conv2D(256, kernel_size=(7,1), padding='same', dilation_rate=(1,1))(visual_inputs)
		model = BatchNormalization()(model)
		model = Activation("relu")(model)
		
		model = Conv2D(256, kernel_size=(5,1), padding='same', dilation_rate=(1,1))(model)
		model = BatchNormalization()(model)
		model = Activation("relu")(model)
		
		model = Conv2D(256, kernel_size=(5,1), padding='same', dilation_rate=(2,1))(model)
		model = BatchNormalization()(model)
		model = Activation("relu")(model)
		
		model = Conv2D(256, kernel_size=(5,1), padding='same', dilation_rate=(4,1))(model)
		model = BatchNormalization()(model)
		model = Activation("relu")(model)
		
		model = Conv2D(256, kernel_size=(5,1), padding='same', dilation_rate=(8,1))(model)
		model = BatchNormalization()(model)
		model = Activation("relu")(model)
		
		model = Conv2D(256, kernel_size=(5,1), padding='same', dilation_rate=(16,1))(model)
		model = BatchNormalization()(model)
		model = Activation("relu")(model)
		
		model = Reshape((75, 256, 1))(model)
		
		# TODO - upsampling code taken from other repo
		def UpSampling2DBilinear(size):
			return Lambda(lambda x: tf.image.resize_bilinear(x, size, align_corners=True))
		
		model = UpSampling2DBilinear((298, 256))(model)
		
		visual_outputs = Reshape((298, 256))(model)
		
		visual_model = Model(visual_inputs, visual_outputs)
		
		visual_model.summary()
	
		visual_inputs_list.append(visual_inputs)
		visual_outputs_list.append(visual_outputs)
	
	# == AV fused neural network ==
	
	fused_model = concatenate([audio_outputs]+visual_outputs_list, axis=2)
	
	# # TEMP
	# fused_model = audio_model
	
	# AV fusion step(s)
	fused_model = TimeDistributed(Flatten())(fused_model)
	
	# BLSTM
	new_matrix_length = 400
	# fused_model = Bidirectional(LSTM(new_matrix_length//2, return_sequences=True, input_shape=(298, 257*8)))(fused_model)
	fused_model = Bidirectional(LSTM(new_matrix_length//2, return_sequences=True, input_shape=(298, 257*8 + 256*num_speakers)))(fused_model)
	
	# Fully connected layers
	fused_model = Dense(600, activation="relu")(fused_model)
	fused_model = Dense(600, activation="relu")(fused_model)
	fused_model = Dense(600, activation="relu")(fused_model)
	
	# Output layer (i.e. complex masks)
	fused_model = Dense(257*2*num_speakers, activation="sigmoid")(fused_model)				# TODO: check if this is more correct (based on the paper)
	outputs_complex_masks = Reshape((298, 257, 2, num_speakers), name="output_layer")(fused_model)
	
	av_model = Model(inputs=[audio_inputs]+visual_inputs_list, outputs=outputs_complex_masks)
	
	model = av_model
	
	# Print the output shapes of each model layer
	for layer in model.layers:
		name = layer.get_config()["name"]
		if "batch_normal" in name or "activation" in name:
			continue
		print(layer.output_shape, "\t", name)
	
	# Alternatively, print the default keras model summary
	model.summary()
	
	# Compile the model before training
	model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
	
	return model


def main():
	
	model_suffix = ""
	
	'''Save a new model'''
	
	# Create a compiled model
	model = convolution_model()
	
	# TODO / Temp: Save model for mobile append
	t = datetime.now().strftime("%d_%m_%H%M%S")
	model_save_path = path_to_models + "saved_model_ablation_%s_%s.h5"%(model_suffix, t)
	model.save(model_save_path)
	

if __name__ == '__main__':
	# Create any missing folders for saving data and outputs
	if not os.path.exists(path_to_models):
		os.mkdir(path_to_models)
	
	# Run AV speech separation program
	main()


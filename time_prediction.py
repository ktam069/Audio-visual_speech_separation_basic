# Main program - Part IV Project 80 Basic Version

# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.io import wavfile

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Dense, Conv2D, LSTM, Bidirectional
from tensorflow.keras.layers import BatchNormalization, Activation, Flatten, TimeDistributed, Reshape
from tensorflow.keras.callbacks import ModelCheckpoint

import os
# import sys
import glob
from datetime import datetime
import time

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

dataset_train = np.load(path_to_saved_datasets+"dataset_train_0-20.npy")

# Split dataset into inputs and ground truths
x_train = dataset_train[:,:,:,:,0 ]
y_train = dataset_train[:,:,:,:,1:]

x_visual = np.random.rand(len(x_train), 75, 1, 1024)

'''Loading models with custom Lambda layers'''
'''https://stackoverflow.com/questions/54347963/tf-is-not-defined-on-load-model-using-lambda/54348035'''

# model = load_model("saved_model_ablation__26_09_232505.h5")
model = load_model("saved_model_ablation__26_09_232505.h5", custom_objects={'tf': tf})

times = []
num_tests = 30

for i in range(num_tests):
	'''Predict data'''
	start_time = time.time()
	# model.predict([x_train[i:i+1], x_visual[i:i+1]])
	model.predict([x_train[i:i+1], x_visual[i:i+1], x_visual[i:i+1]])
	end_time = time.time()
	runtime = end_time-start_time
	times.append(runtime)
	print("Time taken for run %d:"%(i+1), runtime)


avg = sum(times) / len(times)
print("Number of test:", len(times))
print("Test times:", times)
print("Average running time:", avg)

np.savetxt("Laptop_basic_times.csv", times, delimiter=",")



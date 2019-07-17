# Comparing different methods of performing STFT, with different settings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa

import tensorflow as tf
# from keras.layers import Input, Dense, Convolution2D, Deconvolution2D, Bidirectional, TimeDistributed
from keras.layers import Input, Dense, Convolution2D


import scipy
import math

import time

start_time = time.time()

# import os
# import sys

# sys.path.append("./audio")
# sys.path.append("./video")

# from build_audio_database_v2 import build_database

# ===== Settings =====

# Filepaths to the locations where information was saved
path_to_data = "./data/"
path_to_models = "./saved_models/"

path_to_data_audio = "./data/audio/AV_model_database/single/"
# path_to_data_audio = "./data/audio/audio_train/"
path_to_data_video = "./data/video/face_emb/"
# path_to_data_video = "./data/video/face_input/"


# ====================

def fast_fourier_example():
	Fs = 150.0;  # sampling rate
	Ts = 1.0/Fs; # sampling interval
	t = np.arange(0,1,Ts) # time vector

	ff = 5;   # frequency of the signal
	y = np.sin(2*np.pi*ff*t)

	n = len(y) # length of the signal
	k = np.arange(n)
	T = n/Fs
	frq = k/T # two sides frequency range
	frq = frq[list(range(int(n/2)))] # one side frequency range

	Y = np.fft.fft(y)/n # fft computing and normalization
	Y = Y[list(range(int(n/2)))]

	fig, ax = plt.subplots(2, 1)
	ax[0].plot(t,y)
	ax[0].set_xlabel('Time')
	ax[0].set_ylabel('Amplitude')
	ax[1].plot(frq,abs(Y),'r') # plotting the spectrum
	ax[1].set_xlabel('Freq (Hz)')
	ax[1].set_ylabel('|Y(freq)|')
	
	plt.show()
	
	# Display forever
	while True:
		time.sleep(10)

def load_data():
	
	fig_h = 1
	fig_w = 3
	fig_no = 1

	data_num = 1
	# with open()	
	
	# ==================================
	print("\n\n==================================\n*.npy\n")
	# file_path = path_to_data_audio+"single-%05d.npy"%data_num
	
	data = np.load(path_to_data_audio+"single-%05d.npy"%data_num)
	print(data.shape)
	# print(data)
	
	print("time:", time.time()-start_time)
	
	# ==================================
	print("\n\n==================================\nwav file\n")
	
	fix_sr = 16000
	path = "./data/audio/audio_train/" + "trim_audio_train%d.wav"%data_num
	data, _ = librosa.load(path, sr=fix_sr)
	
	print(data.shape)
	print(data)
	
	print("time:", time.time()-start_time)
	
	# ==================================
	print("\n\n==================================\nscipy stft\n")
	
	length = len(data)
	new_power_base_2 = np.ceil(np.log(length)/np.log(2))
	new_len = pow(2, int(new_power_base_2))
	
	new_len = length	# = 48192 ??
	new_len = 48192 - 512	# taken from example code
	
	print(length, new_power_base_2, new_len)
	
	if new_len > length:
		new_data = np.zeros(new_len)
		new_data[:len(data)] = data
	else:
		new_data = np.zeros(new_len)
		new_data[:] = data[:new_len]
	data = new_data.astype('float32')
		
	# Calculations taken from https://github.com/tensorflow/tensorflow/issues/24620
	sample_rate = 16000 #16kHz
	# segment_size = 3000 #ms
	window_size_ms = 25
	window_stride_ms = 10
	# segment_size_samples = int(sample_rate * segment_size / 1000)
	window_size_samples = int(sample_rate * window_size_ms / 1000)
	window_stride_samples = int(sample_rate * window_stride_ms / 1000)
	
	segment_size_samples = len(data)
	window_size_samples = 512
	
	print("segment_size_samples", segment_size_samples)
	print("window_size_samples", window_size_samples)
	print("window_stride_samples", window_stride_samples)
	
	# fs = 10e3
	# N = 1e5
	# amp = 2 * np.sqrt(2)
	# noise_power = 0.01 * fs / 2
	# t = np.arange(N) / float(fs)
	# mod = 500*np.cos(2*np.pi*0.25*t)
	# carrier = amp * np.sin(2*np.pi*3e3*t + mod)
	# noise = np.random.normal(scale=np.sqrt(noise_power),
							 # size=t.shape)
	# noise *= np.exp(-t/5)
	# x = carrier + noise

	# f, t, Zxx = scipy.signal.stft(x, fs, nperseg=1000)
	# plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=amp)
	# plt.title('STFT Magnitude')
	# plt.ylabel('Frequency [Hz]')
	# plt.xlabel('Time [sec]')
	# plt.show()
	
	plt.subplot(fig_h, fig_w, fig_no)
	fig_no += 1
	
	f, t, data = scipy.signal.stft(data, sample_rate, nperseg=window_size_samples)
	# plt.pcolormesh(t, f, np.abs(data), vmin=0, vmax=np.max(np.abs(data)))
	plt.pcolormesh(t, f, np.abs(data))
	# plt.pcolormesh(np.abs(data))
	plt.title('STFT Magnitude')
	plt.ylabel('Frequency [Hz]')
	plt.xlabel('Time [sec]')
	# plt.show()
	
	# data = tf.signal.stft(data, 
						   # frame_length=window_size_samples,
						   # frame_step=window_stride_samples,
						   # fft_length=window_size_samples,
						   # pad_end=True)
						   # # pad_end=False)
	
	# data = tf.signal.stft(data, frame_length=512, frame_step=512, fft_length=512,
				# window_fn=tf.signal.hann_window, pad_end=False, name=None)
	
	print(data.shape)
	print(data)
	
	print("time:", time.time()-start_time)
	
	
	# ==================================
	print("\n\n==================================\ntf stft\n")
	
	data, _ = librosa.load(path, sr=fix_sr)

	length = len(data)
	new_power_base_2 = np.ceil(np.log(length)/np.log(2))
	new_len = pow(2, int(new_power_base_2))
	
	new_len = length	# = 48192 ??
	new_len = 48192 - 512
	
	print(length, new_power_base_2, new_len)
	
	if new_len > length:
		new_data = np.zeros(new_len)
		new_data[:len(data)] = data
	else:
		new_data = np.zeros(new_len)
		new_data[:] = data[:new_len]
	data = new_data.astype('float32')
		
	# Calculations taken from https://github.com/tensorflow/tensorflow/issues/24620
	sample_rate = 16000 #16kHz
	# segment_size = 3000 #ms
	window_size_ms = 25
	window_stride_ms = 10
	# segment_size_samples = int(sample_rate * segment_size / 1000)
	window_size_samples = int(sample_rate * window_size_ms / 1000)
	window_stride_samples = int(sample_rate * window_stride_ms / 1000)
	
	segment_size_samples = len(data)
	window_size_samples = 512
	
	print("segment_size_samples", segment_size_samples)
	print("window_size_samples", window_size_samples)
	print("window_stride_samples", window_stride_samples)
	
	data = tf.signal.stft(data, 
						   frame_length=window_size_samples,
						   frame_step=window_stride_samples,
						   fft_length=window_size_samples,
						   pad_end=True)
						   # pad_end=False)
	
	# data = tf.signal.stft(data, frame_length=512, frame_step=512, fft_length=512,
				# window_fn=tf.signal.hann_window, pad_end=False, name=None)
	
	data = tf.Session().run(data)
	print(data.shape)
	print(data)
	
	print("time:", time.time()-start_time)
	
	plt.subplot(fig_h, fig_w, fig_no)
	fig_no += 1
	
	plt.pcolormesh(np.abs(data.T))
	plt.title('STFT Magnitude')
	plt.ylabel('Frequency [Hz]')
	plt.xlabel('Time [sec]')
	# plt.show()
	
	# ==================================
	print("\n\n==================================\ntf stft example\n")
	
	# Calculations taken from https://github.com/tensorflow/tensorflow/issues/24620
	sample_rate = 16000 #16kHz
	segment_size = 2000 #ms
	window_size_ms = 40
	window_stride_ms = 20
	segment_size_samples = int(sample_rate * segment_size / 1000)
	window_size_samples = int(sample_rate * window_size_ms / 1000)
	window_stride_samples = int(sample_rate * window_stride_ms / 1000)
	audio_sequence = tf.placeholder(tf.float32,
									[1,segment_size_samples],
									name = 'audio_sequence')
	stfts = tf.signal.stft(audio_sequence, 
						   frame_length=window_size_samples,
						   frame_step=window_stride_samples,
						   fft_length=window_size_samples,
						   pad_end=True,
						   name = 'stfts')
						   
	print("segment_size_samples", segment_size_samples)
	print("window_size_samples", window_size_samples)
	print("window_stride_samples", window_stride_samples)
	
	print(stfts.shape)
	
	print("time:", time.time()-start_time)
	
	# ==================================
	print("\n\n==================================\ngithub stft example\n")
	data, _ = librosa.load(path, sr=fix_sr)
	data = stft_git(data)
	
	print(data.shape)
	print(data)
	
	plt.subplot(fig_h, fig_w, fig_no)
	fig_no += 1
	
	plt.pcolormesh(np.abs(data.T))
	plt.title('STFT Magnitude')
	plt.ylabel('Frequency [Hz]')
	plt.xlabel('Time [sec]')
	plt.show()

	
def stft_git(data, fft_size=512, step_size=160,padding=True):
	# short time fourier transform
	if padding == True:
		# for 16K sample rate data, 48192-192 = 48000
		pad = np.zeros(192,)
		data = np.concatenate((data,pad),axis=0)
	# padding hanning window 512-400 = 112
	window = np.concatenate((np.zeros((56,)),np.hanning(fft_size-112),np.zeros((56,))),axis=0)
	win_num = (len(data) - fft_size) // step_size
	out = np.ndarray((win_num, fft_size), dtype=data.dtype)
	for i in range(win_num):
		left = int(i * step_size)
		right = int(left + fft_size)
		out[i] = data[left: right] * window
	F = np.fft.rfft(out, axis=1)
	print(len(data),fft_size,step_size)
	# print(F.shape)
	return F

def main():
	plt.figure(figsize=(20, 10))
	plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.2, hspace=0.2)
	
	print("time:", time.time()-start_time)
	load_data()
	print("time:", time.time()-start_time)
	
	# fast_fourier_example()

if __name__ == '__main__':
	main()


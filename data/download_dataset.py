# Automatically download and process audio and visual data from youtube vidoes listed in the AVSpeech csv files

import os
import sys

sys.path.append("./audio")
sys.path.append("./video")

from audio_downloader import download_from_training, download_from_testing
from audio_norm import norm
from video_download import download_video
from MTCNN_detect import mtcnn_detect
from frame_inspector import frame_inspect
# from build_audio_database_v2 import build_database

# ===== Settings =====

# Range of data to download from AVSpeech (excludes end_id - usually...)
start_id = 501
end_id = 1001

# Whether to download from the training set or the test set
dl_from_training = True

# Normalise volume for mixing(?) - output files will be in 'data/audio/norm_audio_train'
normalise_data = True

# ====================

def process_audio():
	# Download and trim audio wav files from youtube
	if dl_from_training:
		download_from_training(start_id, end_id)		
	else:
		download_from_testing(start_id, end_id)
	
	# Normalises audio data
	if normalise_data:
		norm(start_id, end_id)
	
	print("\n\n ===== Completed processing audio ===== \n")

def process_video():
	# Download video data from online
	download_video(start_idx=start_id, end_idx=end_id)
	
	# Crop frames to fit face
	mtcnn_detect(detect_range=(start_id,end_id))
	
	# Keep only valid frames
	frame_inspect(inspect_range=(start_id,end_id))
	
	print("\n\n ===== Completed processing video ===== \n")

def build_AV_databases():
	# TODO: Not sure what the exact condition is, but can fail without enough data
	assert end_id-start_id > 5, "Too few samples to generate database (probably...)"
	
	build_database(sample_range=(start_id,end_id))
	
	print("\n\n ===== Completed building databases ===== \n")

def main():
	# Download and process audio data from links
	os.chdir("./audio")
	process_audio()
	
	# Download and process video data from links
	os.chdir("../video")
	process_video()
	
	# # Generate database from audio and visual data
	# os.chdir("../audio")
	# build_AV_databases()

if __name__ == '__main__':
	main()


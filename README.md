# Description
Basic version of the implementation for Part IV Project #80. Commands are adpated to be run on windows.

# Dependencies

To run this project fully, the following libraries and tools are required.

### Download and install:

**Use pip install:**

keras

tensorflow

youtube-dl

pytest

sox

**Install manually:**

ffmpeg

cmake

### Ensure that the following are in your system path environment variables:

...\Python\Scripts

...\ffmpeg\bin

...\sox

# Instructions for Running

## Dataset

Follow the instructions on the [README](https://github.com/ktam069/Audio-visual_speech_separation_basic/tree/master/data) in the data folder.

The code used for downloading the data is taken from another [repository](https://github.com/bill9800/speech_separation).

## Running the Model

After having downloaded a range of data into the data folder, specify the range of data to use in the main.py script (by changing the START_ID and END_ID).

Run the following to train and/or test the model:

```
python main.py
```

(To be completed.)


# Part IV Project 80 - Audio-visual Analysis

## Description
Basic version of the implementation for Part IV Project #80 (2019). Commands are adpated to be run on windows.

## Dependencies

To run this project fully, the following libraries and tools are required.

### Language and version:

Python 3.6

### Download and install:

**Use pip install:**

* keras

* tensorflow

* librosa

* youtube-dl

* pytest

* sox

To install all the required libraries listed above, run the following command from the *ai_project* folder:

```
python -m pip install -r requirements.txt
```

**Install manually:**

* ffmpeg

* cmake

**Ensure that the following are in your system path environment variables:**

* ...\Python\Scripts

* ...\ffmpeg\bin

* ...\sox

## Instructions for Running

### Dataset

Follow the instructions on the [README](https://github.com/ktam069/Audio-visual_speech_separation_basic/tree/master/data) in the data folder.

The code used for downloading the data is taken from another [repository](https://github.com/bill9800/speech_separation).

### Running the Model

After having downloaded a range of data into the data folder, specify the range of data to use in the main.py script (by changing the START_ID and END_ID).

Run the following to train and/or test the model:

```python
python main.py
```

By default, the program will pre-process the dataset, train the model, and proceed to test the trained model.


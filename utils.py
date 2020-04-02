
# TODO: Clean these imports one day lol
import pydub
from pydub import AudioSegment
import pydub
from pydub.playback import play
from python_speech_features import mfcc
from python_speech_features import logfbank
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from time import sleep
import scipy.io.wavfile as wav
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch


def read_mp3(f, normalized=False):
    """MP3 to numpy array"""
    a = pydub.AudioSegment.from_mp3(f)
    y = np.array(a.get_array_of_samples())
    if a.channels == 2:
        y = y.reshape((-1, 2))
    if normalized:
        return a.frame_rate, np.float32(y) / 2**15
    else:
        return a.frame_rate, y


def write_mp3(f, sr, x, normalized=False):
    """numpy array to MP3"""
    channels = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1
    if normalized:  # normalized array - each item should be a float in [-1, 1)
        y = np.int16(x * 2 ** 15)
    else:
        y = np.int16(x)
    song = pydub.AudioSegment(y.tobytes(), frame_rate=sr, sample_width=2, channels=channels)
    song.export(f, format="mp3", bitrate="320k")


def detect_leading_silence(sound, silence_threshold=-50.0, chunk_size=10):
    '''
    sound is a pydub.AudioSegment
    silence_threshold in dB
    chunk_size in ms
    iterate over chunks until you find the first one with sound
    '''
    trim_ms = 0 # ms
    assert chunk_size > 0 # to avoid infinite loop
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
        trim_ms += chunk_size

    return trim_ms


def detect_leading_silence_filepath(filepath):
    sound = pydub.AudioSegment.from_mp3(filepath)
    return detect_leading_silence(sound)


def extract_mfcc(sound):
    chunk_size = 10
    first_noisy_idx = int(detect_leading_silence(sound, chunk_size=chunk_size))
    sound = sound[first_noisy_idx:]
    
    # TODO: Should we uncomment this? how to get rid of warning?
    # return mfcc(np.array(sound.get_array_of_samples()), samplerate=sound.frame_rate)
    return mfcc(np.array(sound.get_array_of_samples()))


def extract_mfcc_filepath(filepath):
    sound = pydub.AudioSegment.from_mp3(filepath)
    return extract_mfcc(sound)


def extract_mfb(sound):
    chunk_size = 10
    first_noisy_idx = int(detect_leading_silence(sound, chunk_size=chunk_size))
    sound = sound[first_noisy_idx:]
    
    # make signal go back to array
    samples = np.array(sound.get_array_of_samples()) 
    if len(samples) == 0:
        raise Exception('Silent clip in extractMFB()')
        
    # TODO: Should we uncomment this? how to get rid of warning?
    # fbank_feat = logfbank(samples, samplerate=sound.frame_rate)
    
    fbank_feat = logfbank(samples) 
    return fbank_feat


def extract_mfb_filepath(filepath):
    sound = pydub.AudioSegment.from_mp3(filepath)
    return extract_mfb(sound)


def length_of_file(filepath):
    sound = pydub.AudioSegment.from_mp3(filepath)
    return(len(sound))


def zero_pad_in_end(audio, max_length):
    length = max_length - len(audio)  
    output = audio + AudioSegment.silent(duration=length)
    return output
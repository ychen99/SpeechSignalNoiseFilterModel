#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tensorflow as tf
import soundfile as sf
import librosa
import numpy as np
import os
from DTLN_model import DTLN_model


def process_file(model, audio_file_name, out_file_name):
    # read audio file with librosa to handle resampling and enforce mono
    in_data, fs = librosa.core.load(audio_file_name, sr=16000, mono=True)
    # get length of file
    len_orig = len(in_data)
    # pad audio
    zero_pad = np.zeros(384)
    in_data = np.concatenate((zero_pad, in_data, zero_pad), axis=0)
    # predict audio with the model
    predicted = model.predict_on_batch(
        np.expand_dims(in_data, axis=0).astype(np.float32))
    # squeeze the batch dimension away
    predicted_speech = np.squeeze(predicted)
    predicted_speech = predicted_speech[384:384 + len_orig]
    print('processed successfully!')
    return predicted_speech




if __name__ == '__main__':
    model_weights = r'model.h5'
    input = '/path'
    output = '/path'
    modelClass = DTLN_model()
    modelClass.build_DTLN_model(norm_stft=True)
    modelClass.model.load_weights(model_weights)
    processed = process_file(modelClass.model, input, output)

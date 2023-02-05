import fnmatch
import os
from random import seed,shuffle
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.callbacks import CSVLogger, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.layers import Activation, Dense, LSTM, Dropout, Lambda, Input, Multiply, Layer, Conv1D
from tensorflow.python.keras.models import Model,load_model
from tensorflow.python.keras import optimizers, backend
from wavinfo import WavInfoReader
import soundfile as sf
import librosa
import librosa.display
import matplotlib

class InstantLayerNormalization(Layer):
    '''
    Class implementing instant layer normalization. It can also be called
    channel-wise layer normalization and was proposed by
    Luo & Mesgarani (https://arxiv.org/abs/1809.07454v2)
    '''

    def __init__(self, **kwargs):
        '''
            Constructor
        '''
        super(InstantLayerNormalization, self).__init__(**kwargs)
        self.epsilon = 1e-7
        self.gamma = None
        self.beta = None

    def build(self, input_shape):
        '''
        Method to build the weights.
        '''
        shape = input_shape[-1:]
        # initialize gamma
        self.gamma = self.add_weight(shape=shape,
                             initializer='ones',
                             trainable=True,
                             name='gamma')
        # initialize beta
        self.beta = self.add_weight(shape=shape,
                             initializer='zeros',
                             trainable=True,
                             name='beta')

    def call(self, inputs):
        '''
        Method to call the Layer. All processing is done here.
        '''

        # calculate mean of each frame
        mean = tf.math.reduce_mean(inputs, axis=[-1], keepdims=True)
        # calculate variance of each frame
        variance = tf.math.reduce_mean(tf.math.square(inputs - mean),
                                       axis=[-1], keepdims=True)
        # calculate standard deviation
        std = tf.math.sqrt(variance + self.epsilon)
        # normalize each frame independently
        outputs = (inputs - mean) / std
        # scale with gamma
        outputs = outputs * self.gamma
        # add the bias beta
        outputs = outputs + self.beta
        # return output
        return outputs

def build_model(norm_stft=False):
    fs = 16000
    batchsize = 10
    len_samples = 1
    activation = 'sigmoid'
    numUnits = 128
    numLayer = 2
    blockLen = 512
    block_shift = 128
    dropout = 0.25
    lr = 1e-3
    max_epochs = 200
    encoder_size = 256
    eps = 1e-7
    # input layer for time signal
    mag_dat = Input(batch_shape=(None,None, 257))
    # (batch_shape=(None, int(np.fix(self.fs * self.len_samples / self.block_shift)), 257))
    # normalizing log magnitude stfts to get more robust against level variations
    #if norm_stft:
    x = InstantLayerNormalization()(tf.math.log(mag_dat + 1e-7))

        # behaviour like in the paper
    #x = mag_dat
    # predicting mask with separation kernel
    for idx in range(2):
        x = LSTM(numUnits, return_sequences=True, stateful=False)(x)
        # using dropout between the LSTM layer for regularization
        # if idx < (numLayer - 1):
        #     x = Dropout(dropout)(x)
        # creating the mask with a Dense and an Activation layer
    x = Dense(257)(x)
    estimated_sig = Activation(activation)(x)
    # create the model
    return Model(inputs=mag_dat, outputs=estimated_sig)


# create instance of the DTLN model class
model = build_model()
# build the model
print(model.summary())

model.load_weights('models_DTLN_model/DTLN_model.h5')
audio_cleanWithNoisy ,sr=librosa.load(os.path.join('NoisySpeech_test' ,'noisy1_SNRdb_0.0_clnsp1.wav'),sr=16000)
STFT_clean_abs = np.abs(librosa.stft(audio_cleanWithNoisy,n_fft=512,hop_length=128,center=False))
STFT_clean_abs=STFT_clean_abs.T
STFT_clean_abs= np.expand_dims(STFT_clean_abs,axis=0)
predict=model(STFT_clean_abs)
print(np.sum(predict))
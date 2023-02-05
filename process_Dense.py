import fnmatch
import os
from random import seed,shuffle
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.callbacks import CSVLogger, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.layers import Activation, Dense,LeakyReLU, LSTM, Dropout, Lambda, Input, Multiply, Layer, Conv1D
from tensorflow.python.keras.models import Model,load_model
from tensorflow.python.keras import optimizers, backend
from wavinfo import WavInfoReader
import soundfile as sf
import librosa
import librosa.display
import matplotlib


def stftLayer( x):
    '''
    Method for an STFT helper layer used with a Lambda layer. The layer
    calculates the STFT on the last dimension and returns the magnitude and
    phase of the STFT.
    '''
    blockLen =512
    block_shift =256
    # creating frames from the continuous waveform
    frames = tf.signal.frame(x, blockLen, block_shift)
    # calculating the fft over the time frames. rfft returns NFFT/2+1 bins.
    stft_dat = tf.signal.rfft(frames)
    # calculating magnitude and phase from the complex signal
    mag = tf.abs(stft_dat)
    phase = tf.math.angle(stft_dat)
    # returning magnitude and phase as list
    return [mag, phase]

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

def build_model():
    fs = 16000
    batchsize = 10

    activation = 'sigmoid'
    blockLen = 512
    block_shift = 128
    dropout = 0.25
    lr = 1e-3
    max_epochs = 200
    encoder_size = 256
    eps = 1e-7
    norm_stft = True
    # input layer for time signal
    mag_dat = Input(batch_shape=(None, 257 * 6))
    if norm_stft:
        x = InstantLayerNormalization()(tf.math.log(mag_dat + 1e-7))
    else:
        # behaviour like in the paper
        x = mag_dat
    # predicting mask with separation kernel
    x = Dense(4 * 257)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dense(int(np.fix(2 * 257)))(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dense(257, activation=activation)(x)

    # create the model
    model = Model(inputs=mag_dat, outputs=x)
    # show the model summary
    print(model.summary())
    return Model(inputs=mag_dat, outputs=x)


# create instance of the DTLN model class
model = build_model()
# build the model
print(model.summary())

model.load_weights('models_Dense_nom_model/Dense_nom_model.h5')
NoisySpeech, fs_1 = sf.read(os.path.join('NoisySpeech_test', 'noisy3_SNRdb_20.0_clnsp3.wav'))
NoisySpeech = tf.convert_to_tensor(NoisySpeech)


NoisySpeech_noisy_abs, STFT_noisy_angle = stftLayer(NoisySpeech)
NoisySpeech_noisy_abs = NoisySpeech_noisy_abs.numpy()
num_samples = NoisySpeech_noisy_abs.shape[0] - (6 -1)
in_dat = np.zeros((num_samples,6*257))
for idx in range(num_samples):

    in_dat[idx,:] = NoisySpeech_noisy_abs[idx:idx+6,:].reshape(-1)
predict=model.predict(in_dat)
print(np.sum(predict))
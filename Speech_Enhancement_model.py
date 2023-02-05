import fnmatch

import os
from random import seed, shuffle

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.callbacks import CSVLogger, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.layers import Activation, Dense, LSTM, Dropout, Lambda, Input, Multiply, Layer, Conv1D
from tensorflow.python.keras.layers.recurrent import DropoutRNNCellMixin, AbstractRNNCell
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import optimizers, backend, losses, initializers, activations,constraints

from wavinfo import WavInfoReader
import soundfile as sf


class Garch_cell(AbstractRNNCell):
    def __init__(self, **kwargs):
        '''
            Constructor
        '''
        super(Garch_cell, self).__init__(**kwargs)
        self.epsilon = 1e-7
        self.pi = None
        self.Alpha = None
        self.beta = None
        self.gamma = None
        self.eta = None

    def build(self, input_shape):
        self.Alpha = self.add_weight(shape=(257,),
                                     initializer=initializers.initializers_v2.Constant(0.5),
                                     trainable=True,
                                     name='Alpha',constraint=MixMax(1,0))
        self.beta = self.add_weight(shape=(257,),
                                    initializer=initializers.initializers_v2.Constant(1.0),
                                    trainable=True,
                                    name='beta',constraint=constraints.NonNeg())
        self.gamma = self.add_weight(shape=(257,),
                                     initializer=initializers.initializers_v2.Constant(0.3),
                                     trainable=True,
                                     name='gamma',constraint=MixMax(1,0))
        self.eta = self.add_weight(shape=(257,),
                                   initializer=initializers.initializers_v2.Constant(0.6),
                                   trainable=True,
                                   name='eta',constraint=MixMax(1,0))
        self.pi = self.add_weight(shape=(257,),
                                           initializer=initializers.initializers_v2.Constant(1.0),
                                           trainable=True,
                                           name='pi',constraint=constraints.NonNeg())

    def call(self, inputs, states):
        mag = inputs[:, 0:inputs.shape[1] // 2]
        probability = inputs[:, inputs.shape[1] // 2:]
        mag_square = tf.square(mag)
        #nosiy_square = mag_square * self.Alpha + probability * self.beta * states[1] + self.gamma * states[0]
        nosiy_square=(self.Alpha*states[0]+(1-self.Alpha)*mag_square)*(1-probability)+(self.beta+self.gamma*states[1]+self.eta*states[0])*probability
        clean = mag - tf.sqrt(nosiy_square+self.epsilon)
        clean = activations.elu(clean, alpha=0)

        # nosiyspeech_sum_ban1 = tf.math.reduce_sum(mag_square[:, 0:32], axis=1)
        # last_nosiyspeech_sum_ban1 = tf.math.reduce_sum(vor_mag_square[:, 0:32], axis=1)
        # nach_nosiyspeech_sum_ban1 = nosiyspeech_sum_ban1 * 0.5 + last_nosiyspeech_sum_ban1 * 0.5
        # speech_sum_ban1 = tf.math.reduce_sum(tf.math.square(clean[:, 0:32]), axis=1)
        # SSR_ban1 = 10 * (tf.math.log(nach_nosiyspeech_sum_ban1 / speech_sum_ban1) / tf.math.log(
        #     tf.constant(10, dtype=tf.float32)))
        # SSR_ban1 = tf.repeat(tf.expand_dims(SSR_ban1, axis=1), 32, axis=1)
        #
        # nosiyspeech_sum_ban2 = tf.math.reduce_sum(mag_square[:, 32:64], axis=1)
        # last_nosiyspeech_sum_ban2 = tf.math.reduce_sum(vor_mag_square[:, 32:64], axis=1)
        # nach_nosiyspeech_sum_ban2 = nosiyspeech_sum_ban2 * 0.5 + last_nosiyspeech_sum_ban2 * 0.5
        # speech_sum_ban2 = tf.math.reduce_sum(tf.math.square(clean[:, 32:64]), axis=1)
        # SSR_ban2 = 10 * (tf.math.log(nach_nosiyspeech_sum_ban2 / speech_sum_ban2) / tf.math.log(
        #     tf.constant(10, dtype=tf.float32)))
        # SSR_ban2 = tf.repeat(tf.expand_dims(SSR_ban2, axis=1), 32, axis=1)
        #
        # nosiyspeech_sum_ban3 = tf.math.reduce_sum(mag_square[:, 64:], axis=1)
        # last_nosiyspeech_sum_ban3 = tf.math.reduce_sum(vor_mag_square[:, 64:], axis=1)
        # nach_nosiyspeech_sum_ban3 = nosiyspeech_sum_ban3 * 0.5 + last_nosiyspeech_sum_ban3 * 0.5
        # speech_sum_ban3 = tf.math.reduce_sum(tf.math.square(clean[:, 64:]), axis=1)
        # SSR_ban3 = 10 * (tf.math.log(nach_nosiyspeech_sum_ban3 / speech_sum_ban3) / tf.math.log(
        #     tf.constant(10, dtype=tf.float32)))
        # SSR_ban3 = tf.repeat(tf.expand_dims(SSR_ban3, axis=1), 257 - 64, axis=1)
        #
        # SSR = tf.concat([SSR_ban1, SSR_ban2, SSR_ban3], axis=1)
        # Koffie_band = SSR * self.band_koffie + self.bais
        # Koffie_band = activations.elu(Koffie_band - 1, alpha=0) + 1
        # Koffie_band = -1 * (activations.elu(-1 * Koffie_band + 3.5, alpha=0) - 3.5)

        ceta_e = mag_square - self.pi * tf.square(clean)
        ceta_e = activations.elu(ceta_e, alpha=0)
        return clean, [nosiy_square, ceta_e]

    def get_config(self):
        config = super(Garch_cell, self).get_config()
        config.update({
            'Alpha':
                self.Alpha,
            'beta':
                self.beta,
            'gamma':
                self.gamma,
            'eta':
                self.eta,
            'pi':
                self.pi})

        return config

class Garch_model(Layer):

    def __init__(self, **kwargs):
        '''
            Constructor
        '''
        super(Garch_model, self).__init__(**kwargs)
        self.cell = None

    def build(self, input_shape):
        self.cell = Garch_cell()
        if isinstance(self.cell, Layer) and not self.cell.built:
            self.cell.build(input_shape[-1:])
            self.cell.built = True

    def call(self, inputs, *args, **kwargs):
        # def step(input_cell, state):
        #     output, new_states = self.cell.call(input_cell, state)
        #     return output, new_states

        init_states = [tf.square(inputs[:, 0, 0:inputs.shape[2] // 2] * (1 - inputs[:, 0, inputs.shape[2] // 2:])),
                       tf.square(inputs[:, 0, 0:inputs.shape[2] // 2] * (1 - inputs[:, 0, inputs.shape[2] // 2:]))]
        last_output, outputs, states = backend.rnn(self.cell.call, inputs, init_states, unroll=False)
        return outputs

class MixMax(constraints.Constraint):


  def __init__(self, max_value, min_value):
    self.max_value = max_value
    self.min_value = min_value

  def __call__(self, w):

    return backend.clip(w,self.min_value,self.max_value)

  def get_config(self):
      return {
          'min_value': self.min_value,
          'max_value': self.max_value,
      }

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

    def get_config(self):
        config = super(InstantLayerNormalization, self).get_config()
        config.update({
            'beta':
                self.beta,
            'gamma':
                self.gamma
        })

        return config

class audio_generator():
    '''
    Class to create a Tensorflow dataset based on an iterator from a large scale
    audio dataset. This audio generator only supports single channel audio files.
    '''

    def __init__(self, path_to_input, path_to_s1, len_of_samples, fs, train_flag=False):
        '''
        Constructor of the audio generator class.
        Inputs:
            path_to_input       path to the mixtures
            path_to_s1          path to the target source data
            len_of_samples      length of audio snippets in samples
            fs                  sampling rate
            train_flag          flag for activate shuffling of files
        '''
        # set inputs to properties
        self.path_to_input = path_to_input
        self.path_to_s1 = path_to_s1
        self.len_of_samples = len_of_samples
        self.fs = fs
        self.train_flag = train_flag
        # count the number of samples in your data set (depending on your disk,
        #                                               this can take some time)
        self.count_samples()
        # create iterable tf.data.Dataset object
        self.create_tf_data_obj()

    def count_samples(self):
        '''
        Method to list the data of the dataset and count the number of samples.
        '''

        # list .wav files in directory
        self.file_names = fnmatch.filter(os.listdir(self.path_to_input), '*.wav')
        # count the number of samples contained in the dataset
        self.total_samples = 0
        for file in self.file_names:
            info = WavInfoReader(os.path.join(self.path_to_input, file))
            self.total_samples = self.total_samples + \
                                 int(np.fix(info.data.frame_count / self.len_of_samples))

    def create_generator(self):
        '''
        Method to create the iterator.
        '''

        # check if training or validation
        if self.train_flag:
            shuffle(self.file_names)
            shuffle(self.file_names)
        # iterate over the files
        for file in self.file_names:


            # read the audio files
            noisy, fs_1 = sf.read(os.path.join(self.path_to_input, file))
            speech, fs_2 = sf.read(os.path.join(self.path_to_s1, file))
            # check if the sampling rates are matching the specifications
            if fs_1 != self.fs or fs_2 != self.fs:
                raise ValueError('Sampling rates do not match.')
            if noisy.ndim != 1 or speech.ndim != 1:
                raise ValueError('Too many audio channels. The DTLN audio_generator \
                                 only supports single channel audio data.')
            # count the number of samples in one file
            num_samples = int(np.fix(noisy.shape[0] / self.len_of_samples))
            # iterate over the number of samples
            for idx in range(num_samples):
                # cut the audio files in chunks
                in_dat = noisy[int(idx * self.len_of_samples):int((idx + 1) *
                                                                  self.len_of_samples)]
                tar_dat = speech[int(idx * self.len_of_samples):int((idx + 1) *
                                                                    self.len_of_samples)]
                # yield the chunks as float32 data
                yield in_dat.astype('float32'), tar_dat.astype('float32')

    def create_tf_data_obj(self):
        '''
        Method to to create the tf.data.Dataset.
        '''

        # creating the tf.data.Dataset from the iterator
        self.tf_data_set = tf.data.Dataset.from_generator(
            self.create_generator,
            (tf.float32, tf.float32),
            output_shapes=(tf.TensorShape([self.len_of_samples]), tf.TensorShape([self.len_of_samples])),
            args=None
        )


class ModelTrain():

    ## Class to create and train the DTLN model
    def __init__(self):
        # constructor
        self.cost_function = self.snr_cost
        self.model = Model()
        self.fs = 16000
        self.batchsize = 32
        self.len_samples = 12
        self.activation = 'sigmoid'
        self.numUnits = 128
        self.numLayer = 2
        self.blockLen = 512
        self.block_shift = 128
        self.dropout = 0.25
        self.lr = 1e-3
        self.max_epochs = 200
        self.encoder_size = 256
        self.eps = 1e-7
        # some line to correctly find some libraries in TF 2
        os.environ['PYTHONHASHSEED'] = str(233)
        #seed(233)
        np.random.seed(233)
        tf.random.set_seed(233)



        # physical_devices = tf.config.experimental.list_physical_devices('GPU')
        # if len(physical_devices) > 0:
        #     for device in physical_devices:
        #         tf.config.experimental.set_memory_growth(device, enable=True)

    @staticmethod
    def snr_cost(s_estimate, s_true):
        '''
        Static Method defining the cost function.
        The negative signal to noise ratio is calculated here. The loss is
        always calculated over the last dimension.
        '''

        # calculating the SNR
        snr = tf.reduce_mean(tf.math.square(s_true), axis=-1, keepdims=True) / \
              (tf.reduce_mean(tf.math.square(s_true - s_estimate), axis=-1, keepdims=True) + 1e-7)
        # using some more lines, because TF has no log10
        num = tf.math.log(snr)
        denom = tf.math.log(tf.constant(10, dtype=num.dtype))
        loss = -10 * (num / (denom))
        # returning the loss
        return loss

    def lossWrapper(self):
        '''
        A wrapper function which returns the loss function. This is done to
        to enable additional arguments to the loss function if necessary.
        '''

        def lossFunction(y_true, y_pred):
            # calculating loss and squeezing single dimensions away
            loss = tf.squeeze(self.cost_function(y_pred, y_true))
            # calculate mean over batches
            loss = tf.reduce_mean(loss)
            # return the loss
            return loss

        # returning the loss function as handle
        return lossFunction

    def stftLayer(self, x):
        '''
        Method for an STFT helper layer used with a Lambda layer. The layer
        calculates the STFT on the last dimension and returns the magnitude and
        phase of the STFT.
        '''

        # creating frames from the continuous waveform
        frames = tf.signal.frame(x, self.blockLen, self.block_shift)
        # calculating the fft over the time frames. rfft returns NFFT/2+1 bins.
        stft_dat = tf.signal.rfft(frames)
        # calculating magnitude and phase from the complex signal
        mag = tf.abs(stft_dat)
        phase = tf.math.angle(stft_dat)
        # returning magnitude and phase as list
        return [mag, phase]

    def ifftLayer(self, x):
        '''
        Method for an inverse FFT layer used with an Lambda layer. This layer
        calculates time domain frames from magnitude and phase information.
        As input x a list with [mag,phase] is required.
        '''

        # calculating the complex representation
        s1_stft = (tf.cast(x[0], tf.complex64) *
                   tf.exp((1j * tf.cast(x[1], tf.complex64))))
        # returning the time domain frames
        return tf.signal.irfft(s1_stft)

    def overlapAddLayer(self, x):
        '''
        Method for an overlap and add helper layer used with a Lambda layer.
        This layer reconstructs the waveform from a framed signal.
        '''

        # calculating and returning the reconstructed waveform
        return tf.signal.overlap_and_add(x, self.block_shift)

    def build_model(self, norm_stft=False):
        # input layer for time signal
        time_dat = Input(batch_shape=(None,None))
        # (batch_shape=(None, int(np.fix(self.fs * self.len_samples / self.block_shift)), 257))
        # normalizing log magnitude stfts to get more robust against level variations
        mag, angle = Lambda(self.stftLayer)(time_dat)

        # behaviour like in the paper
        mag_vor = mag
        # predicting mask with separation kernel
        for idx in range(self.numLayer):
            mag_vor = LSTM(self.numUnits, return_sequences=True, stateful=False)(mag_vor)
            # using dropout between the LSTM layer for regularization
            if idx < (self.numLayer - 1):
                mag_vor = Dropout(self.dropout)(mag_vor)
            # creating the mask with a Dense and an Activation layer
        presence_probability = Dense(257)(mag_vor)
        presence_probability = Activation(self.activation)(presence_probability)
        in_garch = tf.concat([mag, presence_probability], axis=2)
        estimated_mag = Garch_model()(in_garch)
        estimated_frames_1 = Lambda(self.ifftLayer)([estimated_mag, angle])
        encoded_frames = Conv1D(self.encoder_size, 1, strides=1, use_bias=False)(estimated_frames_1)
        encoded_frames_norm = InstantLayerNormalization()(encoded_frames)
        x = encoded_frames_norm

        for idx in range(self.numLayer):
            x = LSTM(self.numUnits, return_sequences=True, stateful=False)(x)
            # using dropout between the LSTM layer for regularization
            if idx < (self.numLayer - 1):
                x = Dropout(self.dropout)(x)

        x = Dense(256)(x)
        mask = Activation(self.activation)(x)

        estimated = Multiply()([encoded_frames, mask])
        # decode the frames back to time domain
        decoded_frames = Conv1D(self.blockLen, 1, padding='causal', use_bias=False)(estimated)
        # create waveform with overlap and add procedure
        estimated_sig = Lambda(self.overlapAddLayer)(decoded_frames)
        # create the model
        self.model = Model(inputs=time_dat, outputs=estimated_sig)
        # show the model summary
        print(self.model.summary())

    def compile_model(self):
        '''
        Method to compile the model for training
        '''

        # use the Adam optimizer with a clipnorm of 3
        optimizeradam = optimizers.adam_v2.Adam(learning_rate=self.lr, clipnorm=3)
        # compile model with loss function
        self.model.compile(loss=self.lossWrapper(), optimizer=optimizeradam)

    def train_model(self, runName, path_to_train_mix, path_to_train_speech, path_to_val_mix, path_to_val_speech):
        '''
        Method to train the DTLN model.
        '''

        # create save path if not existent
        savePath = './models_' + runName + '/'
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        # create log file writer
        csv_logger = CSVLogger(savePath + 'training_' + runName + '.log')
        # create callback for the adaptive learning rate
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                      patience=3, min_lr=10 ** (-10), cooldown=1)
        # create callback for early stopping
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0,
                                       patience=10, verbose=0, mode='auto', baseline=None)
        # create model check pointer to save the best model
        checkpointer = ModelCheckpoint(savePath + runName + '.h5',
                                       monitor='val_loss',
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=True,
                                       mode='auto',
                                       save_freq='epoch'
                                       )

        # calculate length of audio chunks in samples
        len_in_samples = int(np.fix(self.fs * self.len_samples /
                                    self.block_shift) * self.block_shift)
        # create data generator for training data
        generator_input = audio_generator(path_to_train_mix,
                                          path_to_train_speech,
                                          len_in_samples,
                                          self.fs, train_flag=True)
        dataset = generator_input.tf_data_set
        dataset = dataset.batch(self.batchsize, drop_remainder=True).repeat()
        # calculate number of training steps in one epoch
        steps_train = generator_input.total_samples // self.batchsize
        # create data generator for validation data
        generator_val = audio_generator(path_to_val_mix,
                                        path_to_val_speech,
                                        len_in_samples, self.fs)
        dataset_val = generator_val.tf_data_set
        dataset_val = dataset_val.batch(self.batchsize, drop_remainder=True).repeat()
        # calculate number of validation steps
        steps_val = generator_val.total_samples // self.batchsize
        # start the training of the model

        self.model.fit(
            x=dataset,
            batch_size=None,
            steps_per_epoch=steps_train,
            epochs=self.max_epochs,
            verbose=1,
            validation_data=dataset_val,
            validation_steps=steps_val,
            callbacks=[checkpointer, reduce_lr, csv_logger, early_stopping],
            max_queue_size=50,
            workers=4,
            use_multiprocessing=True)
        # clear out garbage
        backend.clear_session()

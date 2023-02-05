import fnmatch
import os
from random import seed,shuffle
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.callbacks import CSVLogger, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.layers import Activation, Dense, LSTM, Dropout, Lambda, Input, Multiply, Layer, Conv1D
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import optimizers, backend,losses
from tensorflow.python.keras.layers import CuDNNLSTM
from wavinfo import WavInfoReader
import soundfile as sf
import librosa
import librosa.display
import matplotlib


def lossFunction2(y_true, y_pred):
    # calculating loss and squeezing single dimensions away
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred , labels= y_true)
    # calculate mean over batches
    loss = tf.reduce_mean(loss)
    # return the loss
    return loss

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

class audio_generator():

    def __init__(self, path_to_nosiyspeech, path_to_clean,path_to_nosie, len_of_samples, fs, train_flag=False):
        self.path_to_nosiyspeech = path_to_nosiyspeech
        self.path_to_clean = path_to_clean
        self.path_to_nosie = path_to_nosie
        self.len_of_samples = len_of_samples
        self.fs = fs
        self.blockLen = 512
        self.block_shift = 256
        self.train_flag = train_flag
        self.count_samples()
        self.create_tf_data_obj()

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

    def count_samples(self):
        self.file_names = fnmatch.filter(os.listdir(self.path_to_nosiyspeech), '*.wav')
        self.total_samples = 0
        for file in self.file_names:
            info = WavInfoReader(os.path.join(self.path_to_nosiyspeech, file))
            self.total_samples = self.total_samples + int(np.fix(info.data.frame_count / (self.len_of_samples*256)))

    def create_generator(self):

     # check if training or validation
        if self.train_flag:
            shuffle(self.file_names)
        for file in self.file_names:
            i =0
            n =0
            file_clean = None
            file_noise = None
            for char in file:
                if char=='_':
                    n = n+1
                    if n ==3:
                        file_clean = file[i+1:]
                        file_noise = file[:i]+'.wav'
                i=i+1

            noisy, fs_1 = sf.read(os.path.join(self.path_to_nosie, file_noise))
            speech, fs_2 = sf.read(os.path.join(self.path_to_clean, file_clean))
            noisyspeech, fs_3 = sf.read(os.path.join(self.path_to_nosiyspeech, file))
            if fs_1 != self.fs or fs_2 != self.fs or fs_3 != self.fs:
                raise ValueError('Sampling rates do not match.')
            if noisy.ndim != 1 or speech.ndim != 1 or noisyspeech.ndim != 1:
                raise ValueError('Too many audio channels. The DTLN audio_generator \
                                     only supports single channel audio data.')

            noisy = tf.convert_to_tensor(noisy)
            speech = tf.convert_to_tensor(speech)
            noisyspeech = tf.convert_to_tensor(noisyspeech)
            STFT_noisyspeech_abs, STFT_noisyspeech_angle = self.stftLayer(noisyspeech)
            STFT_speech_abs, STFT_speech_angle = self.stftLayer(speech)
            STFT_noisy_abs, STFT_noisy_angle = self.stftLayer(noisy)
            STFT_noisy_abs = STFT_noisy_abs.numpy()
            STFT_speech_abs = STFT_speech_abs.numpy()
            STFT_noisyspeech_abs =STFT_noisyspeech_abs.numpy()
            # STFT_noisyspeech_abs = np.abs(librosa.stft(noisyspeech, n_fft=512, hop_length=128, center=False))
            # STFT_speech_abs = np.abs(librosa.stft(speech, n_fft=512, hop_length=128, center=False))
            # STFT_noisy_abs = np.abs(librosa.stft(noisy, n_fft=512, hop_length=128, center=False))
            tar = np.subtract(STFT_speech_abs, STFT_noisy_abs)

            for i in range(tar.shape[0]):
                for j in range(tar.shape[1]):
                    if tar[i, j] > 0.75:
                        tar[i, j] = 1
                    else: tar[i, j] = 0
            num_samples = int(np.fix(STFT_noisyspeech_abs.shape[0] / self.len_of_samples))

            for idx in range(num_samples):

                in_dat = STFT_noisyspeech_abs[int(idx * self.len_of_samples):int((idx + 1) * self.len_of_samples),:]
                tar_dat = tar[int(idx * self.len_of_samples):int((idx + 1) * self.len_of_samples),:]
                yield in_dat.astype('float32'), tar_dat.astype('float32')

    def create_tf_data_obj(self):
        self.tf_data_set = tf.data.Dataset.from_generator(
            self.create_generator,
            (tf.float32, tf.float32),
            output_shapes=(tf.TensorShape([self.len_of_samples,257]), tf.TensorShape([self.len_of_samples,257])),
            args=None
        )


class ModelTrain():

    ## Class to create and train the DTLN model
    def __init__(self):
        #constructor
        self.model = Model()
        self.fs = 16000
        self.batchsize= 64
        self.len_samples = 5
        self.activation = 'sigmoid'
        self.numUnits = 128
        self.numLayer = 2
        self.blockLen = 512
        self.block_shift = 256
        self.dropout = 0.25
        self.lr = 1e-5
        self.max_epochs = 200
        self.encoder_size = 256
        self.eps = 1e-7
        # some line to correctly find some libraries in TF 2
        os.environ['PYTHONHASHSEED'] = str(42)
        seed(42)
        np.random.seed(42)
        tf.random.set_seed(42)

        # physical_devices = tf.config.experimental.list_physical_devices('GPU')
        # if len(physical_devices) > 0:
        #     for device in physical_devices:
        #         tf.config.experimental.set_memory_growth(device, enable=True)



    def lossWrapper(self):
        '''
        A wrapper function which returns the loss function. This is done to
         enable additional arguments to the loss function if necessary.
        '''

        def lossFunction(y_true, y_pred):
            # calculating loss and squeezing single dimensions away
            loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred,labels= y_true)
            # calculate mean over batches
            loss = tf.reduce_mean(loss)
            # return the loss
            return loss

        # returning the loss function as handle
        return lossFunction



    def build_model(self, norm_stft = False):
        # input layer for time signal
        mag_dat = Input(batch_shape=(None,None,257))
        #(batch_shape=(None, int(np.fix(self.fs * self.len_samples / self.block_shift)), 257))
        # normalizing log magnitude stfts to get more robust against level variations
        if norm_stft:
            x = InstantLayerNormalization()(tf.math.log(mag_dat + 1e-7))
        else:
            # behaviour like in the paper
            x = mag_dat
        # predicting mask with separation kernel
        for idx in range(self.numLayer):
            x = LSTM(self.numUnits, return_sequences=True, stateful= False)(x)
            # using dropout between the LSTM layer for regularization
            if idx < (self.numLayer - 1):
                x = Dropout(self.dropout)(x)
            # creating the mask with a Dense and an Activation layer
        x = Dense(257)(x)


        # create the model
        self.model = Model(inputs=mag_dat, outputs=x)
        # show the model summary
        print(self.model.summary())

    def compile_model(self):
        '''
        Method to compile the model for training
        '''

        # use the Adam optimizer with a clipnorm of 3
        optimizeradam = optimizers.adam_v2.Adam(learning_rate=self.lr,clipnorm=3.0)
        # compile model with loss function
        self.model.compile(loss=self.lossWrapper(), optimizer=optimizeradam)

    def train_model(self, runName, path_to_train_mix, path_to_train_speech,path_to_train_nosiy, path_to_val_mix,
                    path_to_val_nosiy, path_to_val_speech):
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
                                    self.block_shift))
        # create data generator for training data
        generator_input = audio_generator(path_to_train_mix,
                                          path_to_train_speech,
                                          path_to_train_nosiy,
                                          len_in_samples,
                                          self.fs, train_flag=True)
        dataset = generator_input.tf_data_set
        dataset = dataset.batch(self.batchsize, drop_remainder=True).repeat()
        # calculate number of training steps in one epoch
        steps_train = generator_input.total_samples // self.batchsize
        # create data generator for validation data
        generator_val = audio_generator(path_to_val_mix,
                                        path_to_val_speech,
                                        path_to_val_nosiy,
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
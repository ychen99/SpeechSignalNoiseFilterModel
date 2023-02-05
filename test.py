import Garch_model_port
import soundfile as sf
import tensorflow as tf
import numpy as np


def snr_cost(s_estimate, s_true):

    # calculating the SNR
    snr = np.mean(np.square(s_true), axis=-1, keepdims=True) / \
          (np.mean(np.square(s_true - s_estimate), axis=-1, keepdims=True) + 1e-7)
    # using some more lines, because TF has no log10
    num = np.log10(snr)
    loss = 10 * num
    # returning the loss
    return loss

Garch = Garch_model_port.Garch_model_predict()
print("load_over")
noisy, fs_1 = sf.read("run_set/noisy/fileid_6.wav")
clean, fs = sf.read("run_set/clean/fileid_6.wav")
predict=Garch.predict(noisy)
print(snr_cost(predict,clean))
print(snr_cost(noisy,clean))
sf.write("clean_test_fileid_6.wav",predict,fs_1)



from Speech_Enhancement_model import ModelTrain
import os
import tensorflow as tf
import sys

path_to_train_mix = 'training_set/train/noisy'
path_to_train_speech = 'training_set/train/clean'

path_to_val_mix = 'training_set/val/noisy'
path_to_val_speech = 'training_set/val/clean'

sys.setrecursionlimit(3000)
print(sys.getrecursionlimit())
# name your training run
runName = 'Enhancement3'
#create instance of the DTLN model class
my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')
os.environ["OMP_NUM_THREADS"] = "1" # 1为一个核，设置为5的时候，系统显示用了10个核，不太清楚之间的具体数量关系
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


modelTrainer = ModelTrain()
# build the model
modelTrainer.build_model()
# compile it with optimizer and cost function for training
modelTrainer.compile_model()
modelTrainer.model.load_weights("models_Enhancement3/Enhancement3.h5")
# train the model
modelTrainer.train_model(runName, path_to_train_mix, path_to_train_speech, path_to_val_mix, path_to_val_speech)
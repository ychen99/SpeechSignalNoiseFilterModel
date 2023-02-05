from model_nur_Dense import ModelTrain

path_to_train_mix = 'NoisySpeech_training'
path_to_train_speech = 'CleanSpeech_training'
path_to_train_nosiy = 'Noise_training'
path_to_val_mix = 'NoisySpeech_test'
path_to_val_speech = 'CleanSpeech_test'
path_to_val_nosiy = 'Noisy_test'

# name your training run
runName = 'Dense_nom_model'
# create instance of the DTLN model class
modelTrainer = ModelTrain()
# build the model
modelTrainer.build_model(norm_stft=True)
# compile it with optimizer and cost function for training
modelTrainer.compile_model()
modelTrainer.model.load_weights('models_Dense_nom_model/Dense_nom_model.h5')
# train the model
modelTrainer.train_model(runName, path_to_train_mix, path_to_train_speech, path_to_train_nosiy,path_to_val_mix,
                          path_to_val_nosiy ,path_to_val_speech)
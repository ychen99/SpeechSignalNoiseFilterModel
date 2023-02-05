from Speech_Enhancement_model import ModelTrain
import numpy as np


#先创建class Garch_model_predict()，降噪时直接调用predict就行
class Garch_model_predict():
    def __init__(self):
        self.Garch_model = ModelTrain()
        # build the model
        self.Garch_model.build_model()
        self.Garch_model.model.load_weights("models_Enhancement3/Enhancement3.h5")

    # 输入为numpy数组，输出为numpy数组
    # 输入数组的采样频率为16k 如不是请用重采样，例如使用librosa.resample，然后再调用本程序

    def predict(self,noisy):
        noisy = np.expand_dims(noisy, axis=0)
        output = self.Garch_model.model.predict(noisy)
        return output[0, :]



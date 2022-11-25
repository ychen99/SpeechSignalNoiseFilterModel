import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import math
import soundfile
from arch import  arch_model

#检测信噪比的函数
def SNR(s,sn):
    s=s-np.mean(s)
    s=s/np.max(np.abs(s))
    mean_s=(np.sum(s))/(len(s))
    ps=np.sum((s-mean_s)*(s-mean_s))
    pn=np.sum((s-sn)*(s-sn))
    return 10*math.log((ps/pn),10)

#文件读取 clnsp1 无噪音音频     noisy1 纯噪音    noisy1+clnsp1 合成的含噪音频
noisy, sr=librosa.load('./noisy1.wav',sr=8000)
audio_cleanWithNoisy ,sr=librosa.load('./noisy1+clnsp1.wav',sr=8000)
clean, sr=librosa.load('clnsp1.wav',sr=8000)
#sftf傅里叶变换 取幅值和相位
STFT_clean_abs = np.abs(librosa.stft(clean,n_fft=512,center=False))
STFT_clean_angle = np.angle(librosa.stft(clean,n_fft=512,center=False))
STFT_noisy_abs = np.abs(librosa.stft(noisy,n_fft=512,center=False))
STFT_noisy_angle= np.angle(librosa.stft(noisy,n_fft=512,center=False))
STFT_cleanWithNoisy_abs = np.abs(librosa.stft(audio_cleanWithNoisy,n_fft=512,center=False))
STFT_cleanWithNoisy_angle = np.angle(librosa.stft(audio_cleanWithNoisy,n_fft=512,center=False))
#转换某些数据类型方便后面使用
STFT_clean_abs = STFT_clean_abs.astype(dtype='float64',order="C")
STFT_noisy_angle=STFT_noisy_angle.astype(dtype='float64',order="C")
STFT_noisy_abs=STFT_noisy_abs.astype(dtype='float64',order="C")
#取幅值的平方
STFT_cleanWithNoisy_abs_square=np.square(STFT_cleanWithNoisy_abs)
STFT_cleanWithNoisy_abs_square=STFT_cleanWithNoisy_abs_square.astype(dtype='float64',order="C")
#申明一些列表方便后面使用
omega_square=np.zeros(STFT_cleanWithNoisy_abs_square.shape)
omega_square_noisy=np.zeros(STFT_cleanWithNoisy_abs_square.shape)
omega_square_clean=np.zeros(STFT_cleanWithNoisy_abs_square.shape)
STFT_filtered = np.zeros(STFT_cleanWithNoisy_abs.shape,dtype=complex)


if STFT_cleanWithNoisy_abs_square.shape[0]==(STFT_noisy_abs.shape[0]):
    print("LIST IS OK")


for i in range (STFT_noisy_abs.shape[0]):#分别计算每个频率的 arch模型参数
    noisy_arch_res = arch_model(STFT_noisy_abs[i],mean="Zero",dist="normal",rescale=False).fit(update_freq=5)
    noisy_arch_res_np = noisy_arch_res.params.to_numpy()#计算i频率下的纯噪音的arch参数 Ω α β
    for j in range (STFT_cleanWithNoisy_abs_square[i].shape[0]):#使用上一部算出来的参数带入含噪音频的arch模型中计算噪音的方差
        if j==0:
            omega_square[i,j]= noisy_arch_res_np[0]
        else: omega_square[i,j]= noisy_arch_res_np[0] + noisy_arch_res_np[1]*STFT_cleanWithNoisy_abs_square[i,j-1] + noisy_arch_res_np[2]*omega_square[i,j-1]

STFT_filtered_abs_squre= STFT_cleanWithNoisy_abs_square - omega_square #含噪音频直接减去预测的噪音方差
STFT_filtered_abs_squre=np.maximum(STFT_filtered_abs_squre,0)#去除非零值
STFT_filtered_abs=np.sqrt(STFT_filtered_abs_squre)
STFT_filtered.real = STFT_filtered_abs*np.cos(STFT_cleanWithNoisy_angle)
STFT_filtered.imag = STFT_filtered_abs*np.sin(STFT_cleanWithNoisy_angle)
filtered=librosa.istft(STFT_filtered,n_fft=512,center=False)#还原
filtered=filtered*1.2#补偿一点音量
filtered=np.maximum(filtered,-1)#去除小于-1数
filtered=np.minimum(filtered,1)#去除大于1数
soundfile.write('./filtered clean1.wav',filtered,samplerate=sr,subtype='PCM_24')



print('原信噪比：')
print(SNR(clean[0:59700],audio_cleanWithNoisy[0:59700]))
print('处理之后')
print(SNR(filtered[0:59700],audio_cleanWithNoisy[0:59700]))

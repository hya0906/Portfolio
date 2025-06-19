import librosa
import numpy as np
import glob
import os
import pandas as pd
from tensorflow.python.client import device_lib
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from sklearn import preprocessing
import soundfile as sf
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import librosa.display
import statistics as st


os.environ['CUDA_VISIBLE_DEVICES'] = "1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
print(device_lib.list_local_devices())
print(tf.test.is_built_with_cuda())
tf.config.list_physical_devices('GPU')
#tf.test.is_gpu_available('GPU') 위의 함수로 바뀐다고 함 (2.4.0)
print(tf.sysconfig.get_build_info())


ll = []
def get_feature(file_path, mfcc_len=39, mean_signal_length=90000):
    """
    file_path: Speech signal folder
    mfcc_len: MFCC coefficient length
    mean_signal_length: MFCC feature average length
  	"""
    signal, sr = librosa.load(file_path, sr=None)
    signal = librosa.resample(signal, orig_sr=sr, target_sr=16000)
    s_len = len(signal)
    ll.append(s_len)

    if s_len < mean_signal_length:
        pad_len = mean_signal_length - s_len
        pad_rem = pad_len % 2
        pad_len //= 2
        signal = np.pad(signal, (pad_len, pad_len + pad_rem), 'constant', constant_values=0)
    else:
        pad_len = s_len - mean_signal_length
        pad_len //= 2
        signal = signal[pad_len:pad_len + mean_signal_length]

    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=mfcc_len)
    return mfcc.T


data = np.load("C:\\Users\\711_2\\Desktop\\Yuna_Hong\\speech_expression\\TIM-Net_SER-main\\Data\\RAVDE.npy",allow_pickle=True).item()
print(data["x"].shape)
RAVDE_CLASS_LABELS = ("neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised")#rav
#01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised
#emo_label = {Neutral, Happy, Sad, Anger, Fear, Disgust}
#emo_label = {"NEU": 0, "HAP": 1, "SAD": 2, "ANG": 3, "FEA": 4, "DIS": 5}
emo_label = {"NEU": 5, "HAP": 4, "SAD": 6, "ANG": 0, "FEA": 3, "DIS": 2, "a": 1, "b":7}


#RAVDESS
with tf.device('/device:GPU:0'):
    num = 39
    PATH = "C:\\Users\\711_2\\Desktop\\Yuna_Hong\\speech_expression\\TIM-Net_SER-main\\RAVDESS"
    FOLDER_PATH = os.listdir(PATH)
    ls = glob.glob(PATH)
    print(ls)

    features = []
    labels = []
    for folders in FOLDER_PATH:
        folder = os.path.join(PATH, folders)
        wav_files = os.listdir(folder)

        # print(len(wav_files))
        # sampleList = random.sample(wav_files, 6)
        # for i in range(6):
        #     a = wav_files.index(sampleList[i])
        #     del wav_files[a]
        # print(len(sampleList), len(wav_files))

        for wav_file in wav_files:
            wav_path = os.path.join(folder, wav_file)
            label = [0 for i in range(len(RAVDE_CLASS_LABELS))]
            label[int(wav_file.split('-')[2]) - 1] = float(1)
            labels.append(label)

            feature = get_feature(wav_path)
            features.append(feature)
            print(np.shape(np.array(features)))

    features = np.array(features)
    labels = np.array(labels)

    print("!!!!!!!", np.array(features).shape)
    features = np.array(features, dtype="float32")
    labels = np.array(labels, dtype="float32")
    data = {"x": features, "y": labels}

    print("min: ", min(ll), "max: ",max(ll))
    print(features.shape)

    plt.figure(figsize=(12, 4))
    librosa.display.specshow(features[1].T)
    plt.ylabel('MFCC coeffs')
    plt.xlabel('Time')
    plt.title('mel spectrogram (normalize -1~1)')
    plt.colorbar()
    plt.tight_layout()
    ## plot with default setting

    plt.show()

    plt.figure(figsize=(12, 4))
    librosa.display.specshow(features[2].T)
    #plt.ylabel()
    plt.xlabel('Time')
    plt.title('mel spectrogram to db (normalize -1~1)')
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))
    librosa.display.specshow(features[3].T)
    #plt.ylabel()
    plt.xlabel('Time')
    plt.title('mel spectrogram(trans) to db')
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    np.save(f'./data/RAVDE_autoencoder', data)
    x_save_load = np.load(f'./data/RAVDE_autoencoder.npy', allow_pickle=True)
    print(x_save_load)


#CREMA-D
#
# with tf.device('/device:GPU:0'):
#     num = 39
#     PATH = "C:\\Users\\711_2\\Desktop\\Yuna_Hong\\speech_expression\\TIM-Net_SER-main\\CREMA-D"
#     FOLDER_PATH = os.listdir(PATH)
#     ls = glob.glob(PATH)
#     print(ls)
#
#     features = []
#     labels = []
#     for idx, file_name in enumerate(FOLDER_PATH):
#         wav_path = os.path.join(PATH, file_name)
#         #print(file_name)
#         label = [0 for i in range(8)]
#         print(file_name)
#         label[int(emo_label[file_name.split('_')[-2]])] = float(1)
#         print(label)
#
#         #print(label)
#         feature = get_feature(wav_path, mfcc_len = num)
#         #print(feature)
#         features.append(feature)
#         labels.append(label)
#         if idx % 200 == 0:
#             print(np.shape(np.array(features)))
#             print(np.shape(np.array(labels)))
#
#     print(min(ll))
#     print(st.median(ll))
#     print(max(ll))
#     print("!!!!!!!", np.array(features).shape)
#     features = np.array(features, dtype="float32")
#     labels = np.array(labels, dtype="float32")
#     data = {"x": features, "y": labels}
#
#     plt.figure(figsize=(12, 4))
#     librosa.display.specshow(features[1].T)
#     plt.ylabel('MFCC coeffs')
#     plt.xlabel('Time')
#     plt.title('MFCC')
#     plt.colorbar()
#     plt.tight_layout()
#     plt.show()
#     plt.figure(figsize=(12, 4))
#     librosa.display.specshow(features[2].T)#[:, 1:].T)
#     plt.ylabel('MFCC coeffs')
#     plt.xlabel('Time')
#     plt.title('MFCC')
#     plt.colorbar()
#     plt.tight_layout()
#     plt.show()
#     '''
#     folder = os.path.join(PATH, folders)
#     wav_files = os.listdir(folder)
#     file_path = os.path.join(folder, wav_files[0])
#     mfcc_ = get_feature_(file_path)
#     print("original", mfcc_)
#     signal, fs = librosa.load(file_path)
#     a = signal[:81000].reshape(450,180)
#     print(a)
#     plt.matshow(a)
#     plt.show()
#     '''
#     # #
#     np.save(f'./data/CREMA_D_mfcc_60_110000_append8_stretch', data)
#     x_save_load = np.load(f'./data/CREMA_D_mfcc_60_110000_append8_stretch.npy', allow_pickle=True)
#     print(x_save_load)



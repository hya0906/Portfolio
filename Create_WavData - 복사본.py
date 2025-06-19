import librosa, soundfile
import numpy as np
import glob
import os
import pandas as pd
from tensorflow.python.client import device_lib
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import librosa.display
import numpy as np
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from tqdm import tqdm
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

print(device_lib.list_local_devices())
print(tf.test.is_built_with_cuda())
tf.config.list_physical_devices('GPU')
#tf.test.is_gpu_available('GPU') 위의 함수로 바뀐다고 함 (2.4.0)
print(tf.sysconfig.get_build_info())

# https://www.kaggle.com/code/kartik2khandelwal/data-augmentation
# NOISE
def noise(data, noise_factor=0.005):
    # a = np.amax(data)
    # b = np.random.uniform()
    # noise_amp = 0.035* b * a
    # data = data + noise_amp*np.random.normal(size=data.shape[0])
    wn = np.random.randn(len(data))
    augmented_data = data + noise_factor * wn
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data

# STRETCH
def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate =0.8)
# SHIFT
def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)
# PITCH
def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=pitch_factor)

def scaling_amplitude(data, amplitude_factor=5):
    data *= amplitude_factor
    return data

audio_aug = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    Shift(min_shift = -0.3, max_shift = 0.3, rollover=True, p=0.5),
])

ll = []
# def get_feature(file_path, sr, mfcc_len=39, mean_signal_length=90000, test = False):
#     """
#     file_path: Speech signal folder
#     mfcc_len: MFCC coefficient length
#     mean_signal_length: MFCC feature average length
#   	"""
#     signal, fs = librosa.load(file_path, sr=16000)
#     print(signal.shape, type(signal))
#
#     # librosa.display.waveshow(signal, fs, alpha=0.5)
#     # plt.xlabel("Time (s)")
#     # plt.ylabel("Amplitude")
#     # plt.title("Waveform")
#     # plt.show()
#     s_len = len(signal)
#     ll.append(s_len)
#
#     # if test is False:
#     #     if np.random.randint(2) == 0:
#     #         signal = scaling_amplitude(signal, amplitude_factor= random.uniform(0.8,1.2))
#     #     signal = audio_aug(signal, sample_rate=16000)
#     #     #print("augmentation")
#
#         # if np.random.randint(2) == 0:
#         #     signal = scaling_amplitude(signal, amplitude_factor= random.uniform(0.8,1.2))
#         # if np.random.randint(2) == 0:
#         #     signal = shift(signal)
#         # if np.random.randint(2) == 0:
#         #     signal = noise(signal, noise_factor=(np.random.randint(1, 10) * 0.001))
#         # if np.random.randint(2) == 0:
#         #     pitch_fac = random.random()
#         #     if pitch_fac == 0:
#         #         pitch_fac = random.random()
#         #     signal = pitch(signal, fs, pitch_factor=pitch_fac)
#     # else:
#     #     print("No augmentation")
#
#     if test is True:
#         if s_len < mean_signal_length:
#             pos = random.randint(0, mean_signal_length - s_len)
#             # 패칭개수 110000 - s_len - pos
#             signal = np.pad(signal, (pos, mean_signal_length - s_len - pos), 'constant', constant_values=0)
#         else:
#             pad_len = s_len - mean_signal_length
#             pad_len //= 2
#             signal = signal[pad_len:pad_len + mean_signal_length]
#
#     elif test is False:
#         if s_len < mean_signal_length:
#             #pos = random.randint(0, mean_signal_length - s_len)
#             # 패칭개수 110000 - s_len - pos
#             print("s_len: ", s_len)
#
#             pos_ = random.randint(16000, 32000)
#
#             # padding = signal.tolist()
#             # print(len(padding))
#             padding = np.array(signal[pos_:pos_ + (mean_signal_length - s_len)])
#             print("padding", padding.shape)
#             print("rest", mean_signal_length - s_len, "pos",pos_, "all", pos_ + (mean_signal_length - s_len))
#
#             # signal = np.pad(signal, (pos, mean_signal_length - s_len - pos), 'constant', constant_values=0)
#             print("")
#             if np.random.randint(2) == 0:
#                 #signal_.extend(padding) #뒤에추가
#                 signal = np.hstack((signal, padding))
#                 print("1",signal.shape)
#                 print("================")
#                 return signal
#             else:
#                 # padding.extend(signal_) #앞에추가
#                 signal_ = np.hstack((padding, signal))
#                 print("2",signal_.shape)
#                 print("================")
#                 return signal_
#
#
#         else:
#             pad_len = s_len - mean_signal_length
#             pad_len //= 2
#             signal = signal[pad_len:pad_len + mean_signal_length]
#             return signal
#
#     #mfcc = librosa.feature.mfcc(y=signal, sr=fs, n_mfcc=mfcc_len).T
#     #return signal #mfcc

def get_feature(file_path, sr, mfcc_len=39, mean_signal_length=90000, test = False):
    """
    file_path: Speech signal folder
    mfcc_len: MFCC coefficient length
    mean_signal_length: MFCC feature average length
  	"""
    signal, fs = librosa.load(file_path, sr=16000)
    s_len = len(signal)
    ll.append(s_len)

    if test is False:
        if np.random.randint(2) == 0:
            signal = scaling_amplitude(signal, amplitude_factor=random.uniform(0.8, 1.2))
        signal = audio_aug(signal, sample_rate=16000)
        # print("augmentation")

    if s_len < mean_signal_length:
        signal = np.tile(signal, int(mean_signal_length // s_len) + 1)
        pos = random.randint(0, ((mean_signal_length // s_len) + 1) * (s_len) - mean_signal_length)
        signal = signal[pos:pos + mean_signal_length]
    else:
        pad_len = s_len - mean_signal_length
        pad_len //= 2
        signal = signal[pad_len:pad_len + mean_signal_length]

    #mfcc = librosa.feature.mfcc(y=signal, sr=fs, n_mfcc=mfcc_len).T
    return signal #mfcc

def prepare_RAVDESS_DS(path_audios):
    """
    Generation of the dataframe with the information of the dataset. The dataframe has the following structure:
     ______________________________________________________________________________________________________________________________
    |             name            |                     path                                   |     emotion      |     actor     |
    ______________________________________________________________________________________________________________________________
    |  01-01-01-01-01-01-01.wav   |    <RAVDESS_dir>/audios_16kHz/01-01-01-01-01-01-01.wav     |     Neutral      |     1         |
    ______________________________________________________________________________________________________________________________
    ...

    :param path_audios: Path to the folder that contains all the audios in .wav format, 16kHz and single-channel(mono)
    """
    dict_emotions_ravdess = {
        0: 'Neutral',
        1: 'Calm',
        2: 'Happy',
        3: 'Sad',
        4: 'Angry',
        5: 'Fear',
        6: 'Disgust',
        7: 'Surprise'
    }
    data = []
    for path in tqdm(Path(path_audios).glob("**/*.wav")):
        name = str(path).split('\\')[-1].split('.')[0]
        # # except 'Surprise'
        # if int(name.split("-")[2]) - 1 == 7:
        #     continue
        label = dict_emotions_ravdess[int(name.split("-")[2]) - 1]  # Start emotions in 0
        actor = int(name.split("-")[-1])

        try:
            data.append({
                "name": name,
                "path": path,
                "emotion": label,
                "actor": actor
            })
        except Exception as e:
            # print(str(path), e)
            pass
    df = pd.DataFrame(data)
    return df

def generate_train_test(fold, df, save_path=""):
    """
    Divide the data in train and test in a subject-wise 5-CV way. The division is generated before running the training
    of each fold.

    :param fold:[int] Fold to create the train and test sets [ranging from 0 - 4]
    :param df:[DataFrame] Dataframe with the complete list of files generated by prepare_RAVDESS_DS(..) function
    :param save_path:[str] Path to save the train.csv and test.csv per fold
    """
    actors_per_fold = {
        0: [2, 5, 14, 15, 16],
        1: [3, 6, 7, 13, 18],
        2: [10, 11, 12, 19, 20],
        3: [8, 17, 21, 23, 24],
        4: [1, 4, 9, 22],
    }

    test_df = df.loc[df['actor'].isin(actors_per_fold[fold])]
    train_df = df.loc[~df['actor'].isin(actors_per_fold[fold])]

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    if(save_path!=""):
        train_df.to_csv(f"{save_path}/train.csv", sep="\t", encoding="utf-8", index=False)
        test_df.to_csv(f"{save_path}/test.csv", sep="\t", encoding="utf-8", index=False)
    return train_df, test_df

# def time_masking(feat, label, T = 10, freq_mask_num = 4): # (215,39)
#     feat_size = feat.shape[2]
#     seq_len = feat.shape[1]
#     print("feat/seq", feat_size, seq_len)
#     feat_ = copy.deepcopy(feat)
#     new_feat = []
#     new_label = []
#     #print(feat.shape)
#
#     for _ in range(3):
#         # freq mask
#         for _ in range(freq_mask_num):  # freq masking
#             # f = np.random.uniform(low=0, high=F)
#             # f = int(f)
#             f0 = random.randint(0, feat_size - T)
#             j[:, f0: f0 + T] = 10
#             print(feat_size - T)
#
#         for _ in range(freq_mask_num):  # time masking
#             # f = np.random.uniform(low=0, high=F)
#             # f = int(f)
#             t0 = random.randint(10, seq_len - T - 10)
#             j[t0: t0 + T, :] = 10
#
#         new_feat.append(j)
#         new_label.append(l)
#         j = copy.deepcopy(feat_[idx])
#
#     print("new_feat",len(new_feat), "new_label", len(new_label))
#     return np.array(new_feat), np.array(new_label)


RAVDE_CLASS_LABELS = ("angry", "calm", "disgust", "fear", "happy", "neutral","sad","surprise")#rav
#01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised

#RAVDESS
with tf.device('/device:GPU:0'):
    for fold in range(5):
        if fold == 0:
            continue
        print(f"==================fold {fold} =============")
        sr = 16000
        length = 58726 #84351  #
        num = 39
        PATH = r"C:\Users\711_2\Desktop\Yuna_Hong\백업파일\백업\dataset\speech\RAVDESS\audios_16kHz\total"
        save_path = "C:\\Users\\711_2\\Desktop\\Yuna_Hong\\speech_expression\\WAV2MFCC\\WAV2VEC\\distillation_231006\\FineTuningWav2Vec2_out"
        WAV_PATH = os.listdir(PATH)
        ls = glob.glob(PATH)
        print(ls)
        os.makedirs(f'./new_data/total_RAVDESS_aug_wav_{sr}_{length}_padding_x5_fold{fold}_append', exist_ok=True)
        now = datetime.now()
        now_time = datetime.strftime(now, '%Y-%m-%d_%H-%M-%S')

        df = prepare_RAVDESS_DS(PATH)
        print("df", df)
        _, _ = generate_train_test(fold, df, save_path)
        data_files = {
            "train": os.path.join(save_path, "train.csv"),
            "validation": os.path.join(save_path, "test.csv"),
        }

        # Load data
        dataset = load_dataset("csv", data_files=data_files, delimiter="\t")
        train_dataset = dataset["train"]
        eval_dataset = dataset["validation"]
        print("TRAIN: ", len(train_dataset["path"]))
        print("TEST: ", len(eval_dataset["path"]))

        speech_list = [path for path in train_dataset["path"]]
        print(len(speech_list))

        # Augment train data
        features = []
        labels = []
        for idx, wav_file in enumerate(tqdm(speech_list)):
            # 원본1개 + 변형 7개
            for a in range(5):
                label = [0 for i in range(len(RAVDE_CLASS_LABELS))]
                label[int(str(wav_file.split("\\")[-1]).split('-')[2]) - 1] = float(1)
                labels.append(label)
                name = str(wav_file.split("\\")[-1]).split('.')[0] + "-" + str(a)
                if a == 0:
                    feature = get_feature(wav_file, sr=sr, mean_signal_length=length, test=True)
                    # print("no aug - idx: ", idx, len(feature))
                else:
                    feature = get_feature(wav_file, sr=sr, mean_signal_length=length)

                features.append(feature)

                soundfile.write(
                    f'C:\\Users\\711_2\\Desktop\\Yuna_Hong\\speech_expression\\WAV2MFCC\\WAV2VEC\\new_data\\total_RAVDESS_aug_wav_{sr}_{length}_padding_x5_fold{fold}_append\\{name}.wav',
                    feature,
                    16000,
                    format='WAV')
            if (idx + 1) % 500 == 0:
                print(np.array(features).shape)

        # features = np.array(features, dtype="float32")
        # labels = np.array(labels, dtype="float32")
        # print(features.shape, labels.shape)
        #
        # train_data = {"x": features, "y": labels}
        #
        # name1 = "aug_wav_train_data_90000"
        # np.save(f'./new_data/{name1}_{now_time}', train_data)
        # x_save_load = np.load(f'./new_data/{name1}_{now_time}.npy', allow_pickle=True)
        # print(x_save_load)
        # print("train done")

        #################################
        # Augment test data
        speech_list = [path for path in eval_dataset["path"]]
        print(len(speech_list))

        features = []
        labels = []
        for idx, wav_file in enumerate(tqdm(speech_list)):
            label = [0 for i in range(len(RAVDE_CLASS_LABELS))]
            label[int(str(wav_file.split("\\")[-1]).split('-')[2]) - 1] = float(1)
            labels.append(label)
            name = str(wav_file.split("\\")[-1]).split('.')[0] + "-" + str(idx)
            feature = get_feature(wav_file, sr=sr, mean_signal_length=length, test=True)
            features.append(feature)
            soundfile.write(
                f"C:\\Users\\711_2\\Desktop\\Yuna_Hong\\speech_expression\\WAV2MFCC\\WAV2VEC\\new_data\\total_RAVDESS_aug_wav_{sr}_{length}_padding_x5_fold{fold}_append\\{name}.wav",
                feature,
                16000,
                format='WAV')
            if (idx + 1) % 100 == 0:
                print(np.array(features).shape)

        features = np.array(features, dtype="float32")
        labels = np.array(labels, dtype="float32")
        print(features.shape, labels.shape)

        test_data = {"x": features, "y": labels}

        name2 = "aug_wav_test_data_90000"
        np.save(f'./new_data/{name2}_{now_time}', test_data)
        x_save_load = np.load(f'./new_data/{name2}_{now_time}.npy', allow_pickle=True)
        print(x_save_load)
        print("test done")




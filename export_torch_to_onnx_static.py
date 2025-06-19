from onnxruntime.quantization import quantize
from transformers import Wav2Vec2ForCTC, BertModel
import torch
import argparse
from onnxruntime.quantization import quantize_dynamic, QuantType, quantize_static
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model
)
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from onnxruntime.quantization.calibrate import CalibrationDataReader
from functools import partial
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTQuantizer, ORTModelForFeatureExtraction, ORTModelForAudioClassification #ORTModelForSequenceClassification, 
from optimum.onnxruntime.configuration import AutoQuantizationConfig, AutoCalibrationConfig
from transformers import pipeline, AutoTokenizer, Wav2Vec2Processor, AutoProcessor, Wav2Vec2FeatureExtractor
from time import time
import torch
from datasets import load_dataset,  concatenate_datasets
import numpy as np
import onnxruntime as ort
import onnx
import torchaudio
from pathlib import Path
import warnings
import pandas as pd
from tqdm import tqdm
import os
import librosa

#https://github.com/ccoreilly/wav2vec2-service/blob/master/convert_torch_to_onnx.py
#https://github.com/microsoft/onnxruntime/issues/15888

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
        1: 'Happy',
        2: 'Sad',
        3: 'Angry',
        4: 'Fear',
        5: 'Disgust',
        6: 'Surprise'
    }
    data = []
    for path in tqdm(Path(path_audios).glob("**/*.wav")):
        name = str(path).split('\\')[-1].split('.')[0]
        if int(name.split("-")[2]) == 1:
            label = dict_emotions_ravdess[int(name.split("-")[2]) - 1]
        else:
            if int(name.split("-")[2]) == 2:
                label = dict_emotions_ravdess[0]
            elif int(name.split("-")[2]) != 2:
                label = dict_emotions_ravdess[int(name.split("-")[2]) - 2]
        actor = int(name.split("-")[6])
       
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


def prepare_CREMA_DS(path_audios):
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

    # dict_emotions_crema = {"NEU": 0,
    #                        "HAP": 1,
    #                        "SAD": 2,
    #                        "ANG": 3,
    #                        "FEA": 4,
    #                        "DIS": 5}
    dict_emotions_crema = {"NEU": "Neutral",
                           "HAP": "Happy",
                           "SAD": "Sad",
                           "ANG": "Angry",
                           "FEA": "Fear",
                           "DIS": "Disgust"}
    data = []
    for path in tqdm(Path(path_audios).glob("**/*.wav")):
        name = str(path).split('/')[-1].split('.')[0]
        label = dict_emotions_crema[name.split("_")[2]]
        actor = int(name.split("_")[0])
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


def rav_generate_train_test(fold, df, save_path=""):
    """
    Divide the data in train and test in a subject-wise 5-CV way. The division is generated before running the training
    of each fold.

    :param fold:[int] Fold to create the train and test sets [ranging from 0 - 4]
    :param df:[DataFrame] Dataframe with the complete list of files generated by prepare_RAVDESS_DS(..) function
    :param save_path:[str] Path to save the train.csv and test.csv per fold
    """
    RAVDESS_actors_per_fold = {
        0: [2, 5, 14, 15, 16],
        1: [3, 6, 7, 13, 18],
        2: [10, 11, 12, 19, 20],
        3: [8, 17, 21, 23, 24],
        4: [1, 4, 9, 22],
    }
    CREMA_actors_per_fold = {
        0: [1090, 1059, 1086, 1063, 1064, 1065, 1067, 1069, 1008, 1012, 1044, 1076, 1015, 1077, 1082, 1020, 1021,
            1054],  # 1469
        1: [1024, 1057, 1091, 1030, 1062, 1051, 1034, 1003, 1004, 1038, 1007, 1070, 1009, 1041, 1043, 1013, 1019,
            1022],  # 1464
        2: [1088, 1089, 1026, 1001, 1035, 1005, 1006, 1081, 1085, 1014, 1079, 1016, 1049, 1018, 1083, 1084, 1053,
            1055],  # 1476
        3: [1025, 1027, 1029, 1061, 1031, 1032, 1002, 1066, 1037, 1040, 1072, 1010, 1042, 1074, 1078, 1080, 1017,
            1087],  # 1475
        4: [1028, 1033, 1036, 1039, 1045, 1046, 1047, 1048, 1050, 1052, 1056, 1058, 1060, 1068, 1071, 1073, 1075,
            1011, 1023],  # 1558
    }

    test_df = df.loc[df['actor'].isin(RAVDESS_actors_per_fold[fold])]
    train_df = df.loc[~df['actor'].isin(RAVDESS_actors_per_fold[fold])]

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, test_df

def crema_generate_train_test(fold, df, save_path=""):
    """
    Divide the data in train and test in a subject-wise 5-CV way. The division is generated before running the training
    of each fold.

    :param fold:[int] Fold to create the train and test sets [ranging from 0 - 4]
    :param df:[DataFrame] Dataframe with the complete list of files generated by prepare_RAVDESS_DS(..) function
    :param save_path:[str] Path to save the train.csv and test.csv per fold
    """
    RAVDESS_actors_per_fold = {
        0: [2, 5, 14, 15, 16],
        1: [3, 6, 7, 13, 18],
        2: [10, 11, 12, 19, 20],
        3: [8, 17, 21, 23, 24],
        4: [1, 4, 9, 22],
    }
    CREMA_actors_per_fold = {
        0: [1090, 1059, 1086, 1063, 1064, 1065, 1067, 1069, 1008, 1012, 1044, 1076, 1015, 1077, 1082, 1020, 1021,
            1054],  # 1469
        1: [1024, 1057, 1091, 1030, 1062, 1051, 1034, 1003, 1004, 1038, 1007, 1070, 1009, 1041, 1043, 1013, 1019,
            1022],  # 1464
        2: [1088, 1089, 1026, 1001, 1035, 1005, 1006, 1081, 1085, 1014, 1079, 1016, 1049, 1018, 1083, 1084, 1053,
            1055],  # 1476
        3: [1025, 1027, 1029, 1061, 1031, 1032, 1002, 1066, 1037, 1040, 1072, 1010, 1042, 1074, 1078, 1080, 1017,
            1087],  # 1475
        4: [1028, 1033, 1036, 1039, 1045, 1046, 1047, 1048, 1050, 1052, 1056, 1058, 1060, 1068, 1071, 1073, 1075,
            1011, 1023],  # 1558
    }

    test_df = df.loc[df['actor'].isin(CREMA_actors_per_fold[fold])]
    train_df = df.loc[~df['actor'].isin(CREMA_actors_per_fold[fold])]

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, test_df

def speech_file_to_array_fn(path):
    """
    Loader of audio recordings. It loads the recordings and convert them to a specific sampling rate if required, and returns
    an array with the samples of the audio.

    :param path:[str] Path to the wav file.
    :param target_sampling_rate:[int] Global variable with the expected sampling rate of the model
    """
    #speech_array, sampling_rate = torchaudio.load(path)
    speech_array, sampling_rate = librosa.load(path, sr=16000)
    resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
    speech = resampler(speech_array)#.squeeze().numpy()
    return speech

def label_to_id(label, label_list):

    if len(label_list) > 0:
        return label_list.index(label) if label in label_list else -1

    return label

def preprocess_function(examples, input_column = "path", output_column = "emotion"):
    """
    Load the recordings with their labels.

    :param examples:[DataFrame]  with the samples of the training or test sets.
    :param input_column:[str]  Column that contain the paths to the recordings
    :param output_column:[str]  Column that contain the emotion associated to each recording
    :param target_sampling_rate:[int] Global variable with the expected sampling rate of the model
    """
  
    speech_list = [speech_file_to_array_fn(path) for path in examples[input_column]]
    target_list = [label_to_id(label, label_list) for label in examples[output_column]]

    result = processor(speech_list, sampling_rate=16000)
    result["labels"] = list(target_list)
    
    return result

def prepare_dataset(batch):
    audio_array = [torchaudio.load(i) for i in batch["path"]]
    audio_array = [torchaudio.functional.resample(torch.tensor(i), orig_freq=sr, new_freq=16000).numpy()[0] for (i,sr) in audio_array]
    batch = processor(audio_array, sampling_rate=16000,padding=True) #text가 있으면 padding안해도 되지만 text가 없으면 padding 있어야 함
    # print(batch["input_values"][0].shape, batch["input_values"][1].shape)
    batch["input_length"] = [len(i) / 16000 for i in audio_array]
    return batch

class CalibrationDataProvider(CalibrationDataReader):
    def __init__(self):
        super(CalibrationDataProvider, self).__init__()
        self.counter = 0

    def get_next(self):
        if self.counter > 2:
            return None
        else:
            self.counter += 1
            audio_len = 40000
            return {'input': torch.randn(1, audio_len).numpy().astype(np.float32)}
        

def convert_to_onnx(model_id_or_path, onnx_model_name):
    print(f"Converting {model_id_or_path} to onnx")
    model = Wav2Vec2Model.from_pretrained(model_id_or_path)
    # print(model._modules)
    # for i in model._modules:
    #     print("@@@",i, "\n")
    #     print(model._modules[i]._modules)
    audio_len = 40000

    x = torch.randn(1, audio_len, requires_grad=True)

    torch.onnx.export(model,                        # model being run
                    x,                              # model input (or a tuple for multiple inputs)
                    onnx_model_name,                # where to save the model (can be a file or file-like object)
                    export_params=True,             # store the trained parameter weights inside the model file
                    opset_version=11,               # the ONNX version to export the model to
                    do_constant_folding=True,       # whether to execute constant folding for optimization
                    input_names = ['input'],        # the model's input names
                    output_names = ['output'],      # the model's output names
                    dynamic_axes={'input' : {1 : 'audio_len'},    # variable length axes
                                'output' : {1 : 'audio_len'}})


def quantize_onnx_model(onnx_model_path, quantized_model_path, cal_dataset):
    # model = onnx.load(onnx_model_path)
    # nodes = model.graph.node
    # list_name = []
    # # print(list_name)
    # for x in nodes:
    #     if "Conv" in x.name:
    #         list_name.append(x.name)
    # print(list_name)
    print("Starting quantization...")
    quantize_static(onnx_model_path,
                     quantized_model_path,
                     calibration_data_reader = cal_dataset,
                     #nodes_to_quantize=list_name,
                     #nodes_to_quantize=['MatMul', 'Mul'],
                     nodes_to_quantize=['MatMul', 'Attention', 'LSTM', 'Gather', 'Transpose', 'EmbedLayerNormalization'],
                     #nodes_to_exclude=["Conv", "Softmax","Identity", "Matmul","Mul","Add", "Pow"], #있는게1번 없는게2번
                     weight_type=QuantType.QInt8)

    print(f"Quantized model saved to: {quantized_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="jonatasgrosman/wav2vec2-large-xlsr-53-english",
        #default = "bert-base-uncased",
        help="Model HuggingFace ID or path that will converted to ONNX",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Whether to use also quantize the model or not",
        default=True,
    )
    args = parser.parse_args()

    # processor = Wav2Vec2Processor.from_pretrained(args.model,)
    # rav_audios_dir = r"/home/yuna/code/RAVDESS"
    # crema_audios_dir = r"/home/yuna/code/CREMA-D"
    # save_path="./"
    # num_labels = 7
    # label_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    
    # df_rav = prepare_RAVDESS_DS(rav_audios_dir)
    # df_cre = prepare_CREMA_DS(crema_audios_dir)
    
    # train_df, test_df = rav_generate_train_test(0, df_rav, save_path)
    # train_df_, test_df_ = crema_generate_train_test(0, df_cre, save_path)
    # data_all = pd.concat([train_df, train_df_, test_df, test_df_])
    
    # if (save_path != ""):
    #     data_all.to_csv(f"{save_path}/data_all.csv", sep="\t", encoding="utf-8", index=False)
        
    #     data_files = {
    #         "data": os.path.join(save_path, "data_all.csv"),
    #         }

    # # Load data
    # dataset_all = load_dataset("csv", data_files=data_files, delimiter="\t")#, save_infos=True)
    # dataset = dataset_all["data"]
    # print(dataset)
    # dataset = dataset.map(
    #         preprocess_function,
    #         batch_size=100,
    #         batched=True,
    #         # num_proc=4,

    #     )

    cal = CalibrationDataProvider()
    model_id_or_path = args.model
    # onnx_model_name = model_id_or_path.split("/")[-1] + ".onnx"
    name = model_id_or_path.split("/")[-1]
    onnx_model_name = f"/home/yuna/code/saved_onnx/{name}.onnx"
    convert_to_onnx(model_id_or_path, onnx_model_name)
    if (args.quantize):
        # quantized_model_name = model_id_or_path.split("/")[-1] + ".quant.onnx"
        quantized_model_name = f"/home/yuna/code/saved_onnx/{name}.quant_static.onnx"
        quantize_onnx_model(onnx_model_name, quantized_model_name, cal)
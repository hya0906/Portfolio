# from ws4py.client.threadedclient import WebSocketClient
import time
import sys
import urllib
import json
import pyaudio
from array import array
from collections import deque
from queue import Queue, Full
from threading import Thread
import threading
from multiprocessing import Process, Queue
import pickle
from datetime import datetime
import numpy as np
import wave
import matplotlib.pyplot as plt
import speech_recognition as sr

import time

# Text
import argparse
import glob
import os
import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
import time
from transformers import (
    BloomForCausalLM,
    BloomTokenizerFast,
    Wav2Vec2Processor
)

# Speech
import torchaudio
import librosa
from Wav2VecAuxClasses import *

# Visual
# from VER_model import demo_model
import face_recognition
from FD_model.scrfd import SCRFD
import onnx
import onnxruntime
import cv2
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
# from VER_model.clip_inference import Net

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

########################## Fusion ###########################
weight_F = [1, 0.7, 0.3]
########################## Fusion ###########################

########################## Text ##########################
# labels = ['sadness', 'happy', 'disgust', 'anger', 'fear', 'surprise', 'neutral']
##label_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
labels = ['Surprise','Fear','Disgust','Happy','Sadness','Anger','Neutral']
#labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad','Surprised']
#labels = ['Sad', 'Happy', 'Disgust', 'Angry', 'Fear', 'Surprise','Neutral'] #speech
label2int = {
    "sadness": 0,
    "happy": 1,
    "disgust": 2,
    "anger": 3,
    "fear": 4,
    "surprise": 5,
    "neutral": 6
}


def top_k_index(logits, k):
    for i, idx in enumerate(logits.sort()[1][0]):
        if idx == k:
            return logits[0][i], i


args_dict = dict(
    data_dir="",  # path for data files
    output_dir="",  # path to save the checkpoints
    model_name_or_path='bigscience/bloom-560m',
    tokenizer_name_or_path='bigscience/bloom-560m',
    speech_model_path='SER_model/20221115_171626/checkpoint-4400',
    speech_model_id='jonatasgrosman/wav2vec2-large-xlsr-53-english',
    wav_file_path='./wave_file/wave.wav',
    max_seq_length=50,
    learning_rate=3e-6,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    train_batch_size=1,
    eval_batch_size=1,
    num_train_epochs=10,
    gradient_accumulation_steps=16,
    n_gpu=1,
    early_stop_callback=False,
    fp_16=False,  # if you want to enable 16-bit training then install apex and set this to true
    opt_level='O1',  # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
    max_grad_norm=1.0,  # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
    seed=42,
)
args = argparse.Namespace(**args_dict)


########################## Speech ##########################

processor = Wav2Vec2Processor.from_pretrained(args.speech_model_id)
# model_S = Wav2Vec2ForSpeechClassification.from_pretrained(args.speech_model_path).cuda()


# file_name = "C:\\Users\\yuna\\Desktop\BLOOM2-main\\BLOOM2_main\\Emotion_Fusion23_re\\SER_model\\wav2vec2-large-xlsr-53-english.quant_static.onnx"
file_name = ".\\SER_model\\wav2vec2-large-xlsr-53-english.quant_static.onnx"
session = onnxruntime.InferenceSession(file_name, providers=["CUDAExecutionProvider",])
def Speech_Emotion_Recognition(speechs):
    speech_array = np.concatenate(speechs) / (2 ** 15)

    ort_inputs = {session.get_inputs()[0].name: speech_array.reshape(1,-1).astype(np.float32)}
    ort_outs = session.run(None, ort_inputs)[0][0]
    pred = torch.tensor(ort_outs).softmax(dim=-1).cpu().numpy()

    prediction_s = np.zeros(pred.shape)
    #prediction_s = pred

    prediction_s[0] = pred[6]
    prediction_s[1] = pred[2]
    prediction_s[2] = pred[1]
    prediction_s[3] = pred[3]
    prediction_s[4] = pred[5]
    prediction_s[5] = pred[0]
    prediction_s[6] = pred[4]
    values = prediction_s

    return values


########################## Speech ##########################


########################## Visijon ##########################
mode = "image"#  "image" "video"
def Visual_Emotion_Recognition(face_video_frames):
    face_video_frames = face_video_frames[-16:]

    # image = cv2.cvtColor(face_video_frames, cv2.COLOR_BGR2RGB)
    # image = cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    face_video_frames = np.array(face_video_frames)

    arr = []
    for i in range(len(face_video_frames)):
        arr.append((face_video_frames[i]/ 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225])
    face_data_arr = np.array(arr)

    # tensor_face_data = face_data_arr.unsqueeze(0).cuda()#torch.from_numpy(face_data_arr).cuda()
    # tensor_face_data = np.expand_dims(face_data_arr, axis=0)
    # ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(tensor_face_data)}
    ort_inputs = {ort_session.get_inputs()[0].name: np.transpose(face_data_arr, (0, 3, 1, 2)).astype(np.float32)}

    # result = model_2D(tensor_face_data)

    start_time = time.time()

    result = ort_session.run(None, ort_inputs)
    inference_time = time.time() - start_time
    fps_MB = inference_time

    result = torch.tensor(result)[0]


    pred = result.softmax(dim=-1).cpu().detach().numpy()
    # print("visual_emotion_rec", pred)

    # labels = ['Surprise', 'Fear', 'Disgust', 'Happy', 'Sadness', 'Anger', 'Neutral']
    # happy: 3, sad:4, neutral: 6, anger:5, surprise:0, disgust:2
    # prediction_s = np.zeros(pred.shape)
    # prediction_s[0] = pred[0]
    # prediction_s[1] = pred[1]
    # prediction_s[2] = pred[1]
    # prediction_s[3] = pred[3]
    # prediction_s[4] = pred[5]
    # prediction_s[5] = pred[0]
    # prediction_s[6] = pred[4]
    # values = prediction_s
    values = pred

    return values

def FER(image, print_s=False):
    # image = image.transpose(1,2,0)#cv2.cvtColor(image.transpose(1,2,0), cv2.COLOR_BGR2RGB)
    # image = cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    face_data_arr = (image / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

    # tensor_face_data = face_data_arr.unsqueeze(0).cuda()#torch.from_numpy(face_data_arr).cuda()
    tensor_face_data = np.expand_dims(face_data_arr, axis=0)
    # ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(tensor_face_data)}
    ort_inputs = {ort_session.get_inputs()[0].name: np.transpose(tensor_face_data, (0, 3, 1, 2)).astype(np.float32)}

    # result = model_2D(tensor_face_data)

    start_time = time.time()

    result = ort_session.run(None, ort_inputs)
    if print_s == True:
        print(result)
    inference_time = time.time() - start_time
    fps_MB = inference_time

    pred = result[0][0]
    #label_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    #happy: 3, sad:4, neutral: 6, anger:5, surprise:0, disgust:2
    #surprise:1, happy:0, sad:5, neutral :6
    # prediction_s = np.zeros(pred.shape)
    # prediction_s[0] = pred[6]
    # prediction_s[1] = pred[2]
    # prediction_s[2] = pred[1]
    # prediction_s[3] = pred[3]
    # prediction_s[4] = pred[5]
    # prediction_s[5] = pred[0]
    # prediction_s[6] = pred[4]
    # values = prediction_s
    values = pred
    # print("FER", torch.tensor(values).softmax(dim=-1).cpu().detach().numpy())
    emotion_type_V = labels[np.argmax(values)]
    return emotion_type_V

def Fusion(speeches, face_video_frames):

    global emotion_flag, emotion_type_F, emotion_type_S, emotion_type_V

    try:
        Prediction_S = Speech_Emotion_Recognition(speeches)
        print("Speech Emotion: %s" % (labels[np.argmax(Prediction_S)]))
        emotion_type_S = labels[np.argmax(Prediction_S)]
        # print("FINAL_S", emotion_type_S)
    except:
        print("Serror")
        Prediction_S = [0, 0, 0, 0, 0, 0, 0]

    if len(face_video_frames) >= input_num:
        #####################################

        Prediction_V = Visual_Emotion_Recognition(face_video_frames)
        Prediction_V = np.mean(Prediction_V, 0)

        emotion_type_V = labels[np.argmax(Prediction_V)]
        # print("FINAL_V", emotion_type_V)
    else:
        Prediction_V = [0, 0, 0, 0, 0, 0, 0]

    try:
        Prediction_F = (weight_F[0] * Prediction_V + weight_F[1] * Prediction_S) / np.sum(weight_F)
        emotion_type_F = labels[np.argmax(Prediction_F)]
        # print( Prediction_V)
        # print( Prediction_S)
        # print( Prediction_T)

        # print("V- Sa:%.2f, Ha:%.2f, Di:%.2f, An:%.2f, Fe:%.2f, Su:%.2f, Ne:%.2f"
        #       % (Prediction_V[0], Prediction_V[1], Prediction_V[2], Prediction_V[3], Prediction_V[4], Prediction_V[5], Prediction_V[6]))
        # print("S- Sa:%.2f, Ha:%.2f, Di:%.2f, An:%.2f, Fe:%.2f, Su:%.2f, Ne:%.2f"
        #       % (Prediction_S[0], Prediction_S[1], Prediction_S[2], Prediction_S[3], Prediction_S[4], Prediction_S[5], Prediction_S[6]))
        # print("F- Sa:%.2f, Ha:%.2f, Di:%.2f, An:%.2f, Fe:%.2f, Su:%.2f, Ne:%.2f"
        #       % (Prediction_F[0], Prediction_F[1], Prediction_F[2], Prediction_F[3], Prediction_F[4], Prediction_F[5], Prediction_F[6]))

        ##################################
        #출력하려면 이 부분 주석 풀어야함. 아래쪽에 써놓은거는 아님
        # print("V- Su:%.2f, Fe:%.2f, Di:%.2f, Ha:%.2f, Sa:%.2f, An:%.2f, Ne:%.2f"
        #       % (Prediction_V[0], Prediction_V[1], Prediction_V[2], Prediction_V[3], Prediction_V[4], Prediction_V[5], Prediction_V[6]))
        # print("S- Su:%.2f, Fe:%.2f, Di:%.2f, Ha:%.2f, Sa:%.2f, An:%.2f, Ne:%.2f"
        #       % (Prediction_S[0], Prediction_S[1], Prediction_S[2], Prediction_S[3], Prediction_S[4], Prediction_S[5], Prediction_S[6]))
        # print("F- Su:%.2f, Fe:%.2f, Di:%.2f, Ha:%.2f, Sa:%.2f, An:%.2f, Ne:%.2f"
        #       % (Prediction_F[0], Prediction_F[1], Prediction_F[2], Prediction_F[3], Prediction_F[4], Prediction_F[5], Prediction_F[6]))
    except:
        print("Fusion error")
        emotion_type_F = None

    emotion_flag = 1

    frames = []
    speeches = []

    signal = [1, 0, 0]
    end = True



input_num = 8

#
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model_2D = Net.CLIPMLP('ViT-L/14', device)
# pretrained_dict = torch.load('./VER_model/clip_inference/saved/ViT-L/14', map_location='cuda:0')
# model_2D.load_state_dict(pretrained_dict)
# preprocess = model_2D.preprocessing("test")
# model_2D.eval()

# model_2D = demo_model.RN50()
# model_2D_MBv2 = demo_model.MobileNetV2()

# model_2D = torch.nn.DataParallel(model_2D).cuda()
# model_2D = model_2D.cuda()
# model_2D_MBv2 = model_2D_MBv2.cuda()
# onnx_model_name = "./VER_model/model/MBv2_RAF_AFF_GPU.onnx"
onnx_model_name = ".\\VER_model\\MobileNetV2_Demo.onnx"
onnx_model = onnx.load(onnx_model_name)
onnx.checker.check_model(onnx_model)
ort_session = onnxruntime.InferenceSession(onnx_model_name, providers=['CUDAExecutionProvider'])


face_size = 224

video_cap = cv2.VideoCapture(0)
video_frames = []
face_video_frames = []
max_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))  ### 영상 width(X)
max_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  ### 영상 height(Y)
emotion_flag = 0
emotion_type_V = None
emotion_type_S = None
emotion_type_F = None


# Face Detection

detector = SCRFD(model_file='.\\FD_model\\scrfd_500m_bnkps.onnx')

########################## Visijon ##########################


########################## ASR ##########################



def rate_limited(maxPerSecond):
    minInterval = 1.0 / float(maxPerSecond)

    def decorate(func):
        lastTimeCalled = [0.0]

        def rate_limited_function(*args, **kargs):
            elapsed = time.perf_counter() - lastTimeCalled[0]
            leftToWait = minInterval - elapsed
            if leftToWait > 0:
                time.sleep(leftToWait)
            ret = func(*args, **kargs)
            lastTimeCalled[0] = time.perf_counter()
            return ret

        return rate_limited_function

    return decorate


# const values for mic streaming
# CHUNK = 4080
CHUNK = int(5120 / 2)
BUFF = CHUNK * 10
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

# const valaues for silence detection
SILENCE_THREASHOLD = 3000
SILENCE_SECONDS = 1

end = False
start = True
frames = []
speeches = []
signal = [1, 0, 0]

SAD = False




def keyboard_listener():
    global start, end
    global signal


    if start == True and end == True:
        input("")
        print('start listening')
        start = True
        end = False
        signal = [1, 0, 0]
    else:
        input("")
        print("end listening")
        start = True
        end = True
        signal = [1, 1, 1]



class VideoRecorder():
    "Video class based on openCV"
    def __init__(self, cp, camindex=0, fps=30):

        self.open = True
        self.device_index = camindex
        self.fps = fps  # fps should be the minimum constant rate at which the camera can
        #self.video_cap = cv2.VideoCapture(self.device_index + cv2.CAP_DSHOW)
        self.frame_counts_w = 0  ### window frame count
        self.frame_counts_fw = 0  ### face window frame count
        self.frame_counts = 0  ### window frame count

        self.start_time = time.time()
        self.video_frames = []
        self.mask_video_frames = []
        self.face_video_frames = {}
        self.face_id_frames = {}
        self.full_index = 0
        self.cp = cp


        self.elapsed_time = 0
        self.capture_stop_index = 0

        self.emotion_flag = 0
        self.emotion_type = None
        self.emotion_type_3 = None


    def __exit__(self, type, value, traceback):
        # self.video_out.release()
        #self.video_cap.release()
        cv2.destroyAllWindows()


    def record(self):
        "Video starts being recorded"

        ############# visual input ##############
        global video_frames, face_video_frames, video_cap
        global emotion_flag, emotion_type_F, emotion_type_S, emotion_type_V

        video_frames = []
        face_video_frames = []

        while True:
            windowNotSet = True

            # if start == False and end == True:
            ret, v_frame = video_cap.read()
            if ret == 0:
                break
            video_frame = cv2.flip(v_frame, 1)  ###영상 반전
            m_video_frame = video_frame.copy()
            video_frames.append(video_frame)
            video_frames = video_frames[-29:]

            emoTx = 0
            emoTy = 25 * 5
            if emotion_type_V is not None:
                text_emo_V = 'Emotion_V : {}'.format(emotion_type_V)
                cv2.putText(video_frame, text_emo_V, (emoTx, emoTy - 25 * 2), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0), 1)

            if emotion_type_S is not None:
                text_emo_S = 'Emotion_S : {}'.format(emotion_type_S)
                cv2.putText(video_frame, text_emo_S, (emoTx, emoTy - 25 * 3), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1)

            # if emotion_flag != 0 and emotion_type_T is not None:
            #     text_emo_T = 'Emotion_T : {}'.format(emotion_type_T)
            #     cv2.putText(video_frame, text_emo_T, (emoTx, emoTy - 25 * 3), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1)

            if emotion_type_F is not None:
                text_emo_F = 'Emotion_F : {}'.format(emotion_type_F)
                cv2.putText(video_frame, text_emo_F, (emoTx, emoTy - 25 * 4), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)
                # emotion_flag = 0


            # cv2.imshow("Display Video", video_frame)
            video_frame_re = cv2.resize(video_frame, dsize=(0, 0), fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
            cv2.imshow("Display Video", video_frame_re)

            if windowNotSet is True:
                cv2.namedWindow("Display Video", cv2.WINDOW_NORMAL)

                windowNotSet = False

            # bbox = face_recognition.face_locations(video_frame)
            # start_time = time.time()
            bbox, kpss = detector.detect(video_frame, 0.5, input_size = (640, 640))
            # inference_time = time.time() - start_time
            # print(inference_time)

            if len(bbox) > 0:
                bbox = bbox[:, :4]
                ### face track init
                max_face_idx = 0
                max_face = 0
                if len(bbox) > 1:
                    for face_idx, (xs, ys, xe, ye) in enumerate(bbox):
                        if max_face < (xe - xs) * (ye - ys):
                            max_face = (xe - xs) * (ye - ys)
                            max_face_idx = face_idx
                            
                    
                for face_idx, (xs, ys, xe, ye) in enumerate(bbox):
                    # for (xs, ys, xe, ye), conf in zip(bbox, confs):
                    if xs >= 0 and ys >= 0 and xe <= max_width and ye <= max_height:
                        ys = int(ys)
                        xe = int(xe)
                        ye = int(ye)
                        xs = int(xs)
                        face_img_ori = video_frame[ys:ye, xs:xe]



                        face_img = cv2.cvtColor(face_img_ori, cv2.COLOR_BGR2RGB)
                        face_img = cv2.resize(face_img, (face_size, face_size))
                        # face_img = face_img.astype("float32") / 255.0
                        face_img = np.transpose(face_img, (2, 0, 1))
                        face_img = face_img.transpose(1, 2, 0)

                        center_x = int(xs + (xe - xs) / 2)
                        center_y = int(ys + (ye - ys) / 2)
                        center_xy = [center_x, center_y]

                        if max_face_idx == face_idx:
                            face_video_frames.append(face_img)
                            face_video_frames = face_video_frames[-29:]



                        cv2.rectangle(m_video_frame, (xs, ys), (xe, ye), (0, 0, 0), -1)

                        emoTx = xs
                        emoTy = ys

                        # emoTx = 0
                        # emoTy = 25 * 5
                        emotion_frame = FER(face_img)
                        text_emo_V = '{}'.format(emotion_frame)
                        text_size, _ = cv2.getTextSize(emotion_frame, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
                        cv2.rectangle(video_frame, (emoTx, emoTy - 25 ), (emoTx + text_size[0], emoTy - 25 + text_size[1]), (255, 255, 255), -1) #얼굴위감정표시
                        cv2.putText(video_frame, text_emo_V, (emoTx, emoTy - 25 + text_size[1]), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0), 1) #얼굴위감정표시

                        # if emotion_flag != 0 and emotion_type_S is not None:
                        #     text_emo_S = 'Emotion_S : {}'.format(emotion_type_S)
                        #     cv2.putText(video_frame, text_emo_S, (emoTx, emoTy - 25 *2), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1)

                        # if emotion_flag != 0 and emotion_type_T is not None:
                        #     text_emo_T = 'Emotion_T : {}'.format(emotion_type_T)
                        #
                        #     cv2.putText(video_frame, text_emo_T, (emoTx, emoTy - 25*3), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1)

                        # if emotion_flag != 0 and emotion_type_F is not None:
                        #     text_emo_F = 'Emotion_F : {}'.format(emotion_type_F)
                        #     cv2.putText(video_frame, text_emo_F, (emoTx, emoTy - 25*4), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)

                        cv2.rectangle(video_frame, (xs, ys), (xe, ye), (0, 255, 0), 5)

                video_frame_re = cv2.resize(video_frame, dsize=(0, 0), fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
                cv2.imshow("Display Video", video_frame_re)
                # cv2.imshow("Display Video", video_frame)

            k = cv2.waitKey(1) & 0xff




    def variable_init(self):
        "capture variable initialize"
        global video_frames, face_video_frames

        video_frames = video_frames[-16:]
        face_video_frames = face_video_frames[-16:]


    def stop(self):
        "Finishes the video recording therefore the thread too"
        if self.open:
            self.open=False
            self.video_cap.release()
            cv2.destroyAllWindows()


    def start(self):
        "Launches the video recording function using a thread"
        video_thread = threading.Thread(target=self.record)
        video_thread.start()


class AudioRecorder():
    "Audio class based on pyAudio and Wave"
    def __init__(self, cp, rate=16000, fpb=2**10, channels=1, audio_index=0):
        self.open = True
        self.rate = rate
        self.frames_per_buffer = fpb
        self.channels = channels
        self.format = pyaudio.paInt16
        self.audio = pyaudio.PyAudio()
        # self.stream = self.audio.open(format=self.format,
        #                               channels=self.channels,
        #                               rate=self.rate,
        #                               input=True,
        #                               input_device_index=audio_index,
        #                               frames_per_buffer=self.frames_per_buffer)

        self.frame_counts = 0  ### 절대 frame count
        self.frame_counts_w = 0  ### window frame count
        self.full_index = 0
        self.audio_frames = []
        self.cap_audio_frames = []

        self.elapsed_time = 0
        self.cp = cp
        self.sentence = ""


    def __exit__(self, type, value, traceback):
        self.stream.stop_stream()
        self.stream.close()

        self.closed = True
        self.audio.terminate()


    def record(self):


        audio = pyaudio.PyAudio()

        ############# audio input ################

        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            # input_device_index=input_device_id,
            frames_per_buffer=CHUNK
        )


        global start, end
        global frames, speeches, signal, face_video_frames
        global SAD
        global time_start, flag
        flag = 0
        while True:
            ################## audio ###########################################


            # print("listen")
            try:
                if start == True:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    # file_data = np.frombuffer(data[0:], dtype='int16')
                    # file_bytes = file_data.astype('float32')
                    # print("length=", len(data))
                    frames.append(data)
                    signal_data = np.fromstring(data, dtype=np.int16)
                    # peak = 20 * np.log10(np.average(np.abs(signal_data)) * 2)  # 데시벨 구하기
                    # print(int(peak), "decibel") #아무것도 안 말하면 25정도 나옴

                    if int(np.average(np.abs(signal_data))) >= 200 and SAD == False: #약 50데시벨 이상이어야 인식
                        print("\nstart listening")
                        SAD = True
                        flag = 1
                        time_start = time.time()
                    if SAD == True:
                        speeches.append(signal_data)
                        # print("SAD")
                    # print(signal_data.shape)


                    if SAD == True and np.average(np.abs(signal_data)) < 200 and (time.time() - time_start) > 1.5:
                        print("end listening")
                        end = True
                        flag = 0
                        # signal = [1, 1, 1]
                        Fusion(speeches, face_video_frames)
                        # print(time.time() - time_start)

                        ##########
                        now = datetime.now()
                        now_time = datetime.strftime(now, '%Y-%m-%d_%H-%M-%S')
                        WAVE_OUTPUT_FILENAME = f".\\record\\{now_time}.wav"
                        waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
                        waveFile.setnchannels(CHANNELS)
                        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
                        waveFile.setframerate(RATE)
                        waveFile.writeframes(b''.join(frames))
                        waveFile.close()

                        try:
                            r = sr.Recognizer()
                            korean_audio = sr.AudioFile(WAVE_OUTPUT_FILENAME)
                            # print(WAVE_OUTPUT_FILENAME)

                            with korean_audio as source:
                                self.cp.acquire()
                                # print('음성을 입력하세요.')
                                recorded_audio = r.record(source)
                                try:
                                    # print("!!!", audio.get_wav_data())
                                    self.sentence = r.recognize_google(recorded_audio, language='ko-EN')
                                    # print('음성변환 : ' + self.sentence)

                                    self.cp.notifyAll()
                                    self.cp.release()
                                    print("release audio")

                                except sr.UnknownValueError:
                                    print('오디오를 이해할 수 없습니다.')
                                except sr.RequestError as e:
                                    print(f'에러가 발생하였습니다. 에러원인 : {e}')
                            # os.remove(WAVE_OUTPUT_FILENAME)

                        except KeyboardInterrupt:
                            pass



            except Full:
                pass

            if end == True:
                self.variable_init()
                end = False
                SAD = False
                signal = [1, 0, 0]
                # print("EPD\n\n")



    def variable_init(self):
        "capture variable initialize"
        global frames, speeches

        frames = []
        speeches = []


    def stop(self):
        "Finishes the audio recording therefore the thread too"
        if self.open:
            self.open = False

    def start(self):
        "Launches the audio recording function using a thread"
        audio_thread = threading.Thread(target=self.record)
        audio_thread.start()


class Emotion_Fusion():
    "Video class based on openCV"
    def __init__(self):

        signal = []


    def record(self):
        "Video starts being recorded"

        ############# visual input ##############
        global video_frames, face_video_frames, video_cap, signal, speeches
        global emotion_flag, emotion_type_V, emotion_type_S, emotion_type_F
        global end, start
        global weight_F



        while True:

            if signal == [1, 0, 0]:#얼굴만?

                if len(face_video_frames) >= input_num:
                    #####################################
                    Prediction_V = Visual_Emotion_Recognition(face_video_frames)

                    emotion_flag = 1
                    emotion_type_V = labels[np.argmax(np.sum(Prediction_V,0))]

                # cv2.imshow("Display Video", video_frame)
                # # cv2.imshow("Capture Video", m_video_frame)
                # k = cv2.waitKey(1) & 0xff

            elif signal == [1, 1, 1]: #목소리,얼굴둘다

                try:
                    Prediction_S = Speech_Emotion_Recognition(speeches)
                    print("Speech Emotion: %s" % (labels[np.argmax(Prediction_S)]))
                    if text != "":
                        emotion_type_S = labels[np.argmax(Prediction_S)]
                except:
                    print("Serror")
                    Prediction_S = [0, 0, 0, 0, 0, 0, 0]


                if len(face_video_frames) >= input_num:
                    #####################################

                    Prediction_V = Visual_Emotion_Recognition(face_video_frames)

                    emotion_type_V = labels[np.argmax(np.sum(Prediction_V,0))]
                else:
                    Prediction_V = [0,0,0,0,0,0,0]

                try:
                    Prediction_F = (weight_F[0] * Prediction_V + weight_F[1] * Prediction_S) / np.sum(weight_F)
                    emotion_type_F = labels[np.argmax(Prediction_F)]
                    # print( Prediction_V)
                    # print( Prediction_S)
                    # print( Prediction_T)
                    # print("V- Sa:%.2f, Ha:%.2f, Di:%.2f, An:%.2f, Fe:%.2f, Su:%.2f, Ne:%.2f"
                    #       %(Prediction_V[0], Prediction_V[1], Prediction_V[2], Prediction_V[3], Prediction_V[4], Prediction_V[5], Prediction_V[6]))
                    # print("S- Sa:%.2f, Ha:%.2f, Di:%.2f, An:%.2f, Fe:%.2f, Su:%.2f, Ne:%.2f"
                    #       %(Prediction_S[0], Prediction_S[1], Prediction_S[2], Prediction_S[3], Prediction_S[4], Prediction_S[5], Prediction_S[6]))
                    # print("F- Sa:%.2f, Ha:%.2f, Di:%.2f, An:%.2f, Fe:%.2f, Su:%.2f, Ne:%.2f"
                    #       % (Prediction_F[0], Prediction_F[1], Prediction_F[2], Prediction_F[3], Prediction_F[4], Prediction_F[5], Prediction_F[6]))
                    
                    #############################################
                    #나중에 주석 풀면 출력나옴
                    # print("V- Su:%.2f, Fe:%.2f, Di:%.2f, Ha:%.2f, Sa:%.2f, An:%.2f, Ne:%.2f"
                    #       % (Prediction_V[0], Prediction_V[1], Prediction_V[2], Prediction_V[3], Prediction_V[4],
                    #          Prediction_V[5], Prediction_V[6]))
                    # print("S- Su:%.2f, Fe:%.2f, Di:%.2f, Ha:%.2f, Sa:%.2f, An:%.2f, Ne:%.2f"
                    #       % (Prediction_S[0], Prediction_S[1], Prediction_S[2], Prediction_S[3], Prediction_S[4],
                    #          Prediction_S[5], Prediction_S[6]))
                    # print("F- Su:%.2f, Fe:%.2f, Di:%.2f, Ha:%.2f, Sa:%.2f, An:%.2f, Ne:%.2f"
                    #       % (Prediction_F[0], Prediction_F[1], Prediction_F[2], Prediction_F[3], Prediction_F[4],
                    #          Prediction_F[5], Prediction_F[6]))
                except:
                    print("Fusion error")
                    emotion_type_F = None


                emotion_flag = 1


                frames = []
                speeches = []

                signal = [1, 0, 0]
                end = True


                # ws.send_data('EOS'.encode())



    def start(self):
        "Launches the video recording function using a thread"
        ER_thread = threading.Thread(target=self.record)
        ER_thread.start()


def start_AVrecording(camindex=0, fps=30, audio_index=0, sample_rate=16000):
    global video_thread
    global audio_thread


    video_thread = VideoRecorder(camindex=camindex, fps=fps)
    audio_thread = AudioRecorder(audio_index=audio_index, rate=sample_rate)
    # ER_thread = Emotion_Fusion()

    audio_thread.start()
    video_thread.start()
    # ER_thread.start()
    return


def stop_AVrecording():
    audio_thread.stop()
    video_thread.stop()


def AV_Capture(video_thread):

    if video_thread.full_index == 1 and video_thread.frame_counts != 0:
        ### video
        print('video_frm = ', video_thread.frame_counts, 'full_index =', video_thread.full_index, 'capture_frms =', len(video_thread.video_frames), 'elapsed_time=', video_thread.elapsed_time)

        video_thread.variable_init()



        ### Audio
        # print('audio_frm = ', audio_thread.frame_counts,  'capture_samples =', len(audio_thread.cap_audio_frames))
        #audio_data = audio_thread.cap_audio_frames
        #len_audio = len(audio_thread.cap_audio_frames)
        #audio_thread.variable_init()


def list_audio_devices(name_filter=None):
    pa = pyaudio.PyAudio()
    device_index = None
    sample_rate = None
    for x in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(x)
        print(pa.get_device_info_by_index(x))
        if name_filter is not None and name_filter in info['name']:
            device_index = info['index']
            sample_rate = int(info['defaultSampleRate'])
            break
    return device_index, sample_rate

#
def main():

    av_open = True
    start_AVrecording()

    while av_open:
        time.sleep(0)
        AV_Capture()

        if video_thread.capture_stop_index == 1:
            stop_AVrecording()
            break


# if __name__ == '__main__':
#    main()
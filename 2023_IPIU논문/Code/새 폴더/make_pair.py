#리스트의 사진 임베딩데이터추출 테스트
import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo.arcface_onnx import ArcFaceONNX
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import shutil
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from PIL import Image
import os

#integer_image_index \t label_index \t path_to_image

f= open("D:/insightface_folder/lab_test/label.txt", 'w')
folders = list(glob.iglob(os.path.join("C:/Users/IPCG/Desktop/Cropped_images/train", '*'))) #경로 뭉탱이를 리스트로
names = [os.path.basename(folder) for folder in folders]  # only name
c=0
for i, folder in enumerate(folders):
    name = names[i]
    videos = list(glob.iglob(os.path.join(folder, '*.*')))
    for j,img_path in enumerate(videos):
        print(img_path.split('\\')[-1])
        f.write(f"{img_path} \t {i}\n")
        #f.write(f"{c} \t {i} \t {img_path}\n")
        c+=1
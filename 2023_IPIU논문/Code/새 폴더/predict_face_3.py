#################################################################
# https://github.com/LeNguyenGiaBao/face_detection/blob/352b4fe2d1b33748aae0997e18d49b6c7f040bb1/retinaface/test_face_embedding.py
import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo.arcface_onnx import ArcFaceONNX
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import onnxruntime
import os

# from imgbeddings import imgbeddings

embs = []


def processing(imgs):
    print("~~~~~~~~~~~~~~~~~~~~~~~")
    c = 0
    for img in imgs:
        print(c)
        print(img)
        face = app.get(img)  # [i][0]
        emb = face.embedding
        embs.append(emb)
        c += 1

    return embs


# 얼굴 여러개를 인식해서 그 사진만 잘라서 sim을 구해서 데이터베이스에 있는 것과 비슷한게 있으면 표시
# 여러명 안되게/여러명이면 가까이 있는 사람만
img1 = cv2.imread('/content/drive/MyDrive/Colab Notebooks/files/gong.jpg')
print(img1)
print(type(img1))
img2 = cv2.imread('/content/drive/MyDrive/Colab Notebooks/files/dindin.jpg')
img3 = cv2.imread('/content/drive/MyDrive/Colab Notebooks/files/gong1.jpg')
img4 = cv2.imread('/content/drive/MyDrive/Colab Notebooks/files/gong2.jpg')
# img = cv2.imread('/content/dinyu.jpg')
img = cv2.imread('/content/gongdong2.jpeg')
##img = cv2.resize(img,(700,500))#이미지가 너무 작을경우 확대필요★
# img = cv2.imread('/content/gong.jpg')
print("SSSSSSSS", type(img))

model_name = 'buffalo_l'
app = FaceAnalysis(name=model_name, allowed_modules=['detection', 'recognition'])  # enable detection model only
app.prepare(ctx_id=0, det_size=(640, 640))  # 옆,기울어진얼굴 다 인식가능

handler = insightface.model_zoo.get_model('/content/drive/MyDrive/Colab Notebooks/files/w600k_r50.onnx')  # Arcface
handler.prepare(ctx_id=0)  # id는 기능없음


detector = insightface.model_zoo.get_model('/content/drive/MyDrive/Colab Notebooks/files/det_10g.onnx')  # retinaface
detector.prepare(ctx_id=0, det_size=(640, 640))
plt.imshow(img)

#====================================
bboxes, landmarks = detector.detect(img,(256,256),5,2) #img, input_size, max_num, metric #얼굴크면 img크기키워야함
print(bboxes,"/////////////")
print(landmarks,"/////////////")
name = os.path.basename("/content/drive/MyDrive/Colab Notebooks/files/gong.jpg")
print("name",name)
print(img.shape)
#face0 = cv2.imread('/content/face0.jpg')
#face1 = cv2.imread('/content/face1.jpg')
#imgs = [img1, img2, img3, img4] #gong,din,gong,gong,dinyu
#print(imgs)

ort_model = onnxruntime.InferenceSession('/content/drive/MyDrive/Colab Notebooks/emb_train_folder/model.onnx')
input = ort_model.get_inputs()[0].name
output = ort_model.get_outputs()[0].name

for i, box in enumerate(bboxes): #예측확률이 일정이상 넘어가지 않으면 unknown으로★
    # 좌표 추출
    x, y, w, h, _ = box
    print(x,y,w,h)
    # 경계 상자 그리기
    if x>img.shape[1]//2:#y축이 초과?할때도 필요할듯★
        person = img[int(0.5*y):int(1.1*h), int(0.9*x):int(1.1*w)]
    else:
        person = img[int(0.3*y):int(1.1*h), int(0.3*x):int(1.1*w)]
    #cv2.imwrite("face"+str(i)+".jpg",person) #image-y,x
    person = img[int(0.3*y):int(1.1*h), int(0.3*x):int(1.1*w)]
    person = np.array(person)
    face = app.get(person)[0] #얼굴이 사진의 일정비율을 차지하면 원본사진을 그대로 쓰는것추가★
    pic = np.array([face.embedding], dtype=np.float32)
    name = str(np.argmax(ort_model.run([output], {input : pic})))
    print(np.argmax(ort_model.run([output], {input : pic})))
    print("!!!!!!!",ort_model.run([output], {input : pic}))
    #오류구문추가/몇번이상 사람이 인식이 안될경우 크기조정기능필요★
    #rectangle그림따로 얼굴처리용이미지따로★
    print(img.shape[1]//2) #0 가로 1 세로
    print(img.shape)
    if x>img.shape[1]//2:
        cv2.rectangle(img, (int(0.8*x), int(0.3*y)), (int(w*1.1), int(h*1.1)), (0,0,255), thickness=2)
    else:
        cv2.rectangle(img, (int(0.5*x), int(0.3*y)), (int(w*1.2), int(h*1.1)), (0,0,0), thickness=2)
    cv2.circle(img, (int(x), int(y)), 2, (255,0,255), thickness=2)
    cv2.circle(img, list(map(int, tuple(landmarks[i][0]))), 2, (255,0,255), thickness=2)
    cv2.circle(img, list(map(int, tuple(landmarks[i][1]))), 2, (255,0,255), thickness=2)
    cv2.circle(img, list(map(int, tuple(landmarks[i][2]))), 2, (0,255,0), thickness=2)
    cv2.circle(img, list(map(int, tuple(landmarks[i][3]))), 2, (255,0,0), thickness=2)
    cv2.circle(img, list(map(int, tuple(landmarks[i][4]))), 2, (255,0,0), thickness=2)
    cv2.putText(img, name,(int(x), int(y)),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),1) #이름으로 바꾸기★

#embeddings = processing(imgs)
#print("\\\\\\\\\\\\\\\\\\\\")
#sims=[]
#for i in embeddings:
#    sim = handler.compute_sim(i, embeddings[-1])
#    sims.append(sim)
#    print(sim)
#print("\n")

plt.imshow(img)

#print(sim, sim2, sim3, sim4, sim5, "//", sim6, sim7)#, 1-sim3, 1-sim4)
#0jessi 1IU 2yuna_hong 3gongyu 4dongyuk 5barack
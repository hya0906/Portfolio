import insightface
import cv2
from insightface.app import FaceAnalysis
import onnxruntime
import numpy as np
from multiprocessing import Process
import matplotlib.pyplot as plt
import os

#0jessi 1IU 2yuna_hong 3gongyu 4dongyuk 5barack
#'ariana', 'barack', 'brendan', 'Chanhyuk', 'christopher', 'dongyuk', 'gongyu', 'IU', 'jessi', 'justin', 'suhyun', 'yuna_hong'
names = {0: 'ariana', 1: 'barack', 2: 'brendan', 3: 'Chanhyuk', 4: 'christopher', 5: 'dongyuk', 6: 'gongyu', 7: 'IU', 8: 'jessi', 9: 'justin', 10: 'suhyun', 11: 'yuna_hong'}
#buffalo_s는 인식안됨


model_name = 'buffalo_m'
app = FaceAnalysis(name=model_name, allowed_modules=['detection', 'recognition'])  # enable detection model only app.prepare(ctx_id=0, det_size=(640, 640))  # 옆,기울어진얼굴 다 인식가능 detector = insightface.model_zoo.get_model('D:\insightface_folder\lab_test\data\det_10g.onnx')  # retinaface
app.prepare(ctx_id=0, det_size=(640, 640))  # 옆,기울어진얼굴 다 인식가능

detector = insightface.model_zoo.get_model("C:/Users/711_2\Desktop\홍윤아\대학교코드\lab_test\data/det_10g.onnx") # retinaface
#D:\insightface_folder\lab_test\data\det_10g.onnx')
detector.prepare(ctx_id=0, det_size=(640, 640))

ort_model = onnxruntime.InferenceSession('C:/Users/711_2\Desktop\홍윤아\대학교코드\lab_test\data/model_12.onnx')
input = ort_model.get_inputs()[0].name
output = ort_model.get_outputs()[0].name
flag = False
name = ''
bboxes, landmarks = None, None


def who_are_you(pic):  # 일단 한명예측
    name = "Unknown"
    res = ort_model.run([output], {input: pic})
    if res[0][0][np.argmax(res)] > 0.95: #정확도가 0.97이상이면 해당 사람으로 인식
        name = names[np.argmax(res)]
        flag = True #인증된 사람이므로 mediapipe손 인식
    else:
        flag = False
    return name, flag


def recognize_faces(frame):
    global bboxes, name
    bboxes, landmarks = detector.detect(frame, (512, 512), 5, 2)  #이부분이 시간 오래 걸림
    for i, box in enumerate(bboxes):
        # 좌표 추출
        x, y, w, h, _ = map(int, box)
        try:
            #원래는 사람 얼굴 인식된 범위에서 크기 조금 더 크게 해서 얼굴인증하도록 함
            aa = np.array(frame[y-40:h+60, x-30:w+60])
            face = app.get(aa)[0]
            pic = np.array([face.embedding], dtype=np.float32)
            name, flag = who_are_you(pic)
        except:
            name = "Unknown"
            flag = False
        cv2.rectangle(frame, (x-30, y-30), (w+60, h+60), (255, 0, 255), thickness=2)
        cv2.rectangle(frame, (x, y), (w, h), (0, 0, 255), thickness=2)
        cv2.putText(frame, name, (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
    return flag, bboxes

def draw_faces(frame):
    x, y, w, h, _ = map(int, bboxes[0])
    cv2.rectangle(frame, (x, y), (w, h), (0, 0, 255), thickness=2)
    cv2.putText(frame, name, (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
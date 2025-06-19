# 3 리스트의 사진 임베딩데이터추출/학습
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
#'ariana', 'barack', 'brendan', 'Chanhyuk', 'christopher', 'dongyuk', 'gongyu', 'IU', 'jessi', 'justin', 'suhyun', 'yuna_hong'
# '''
embs = []
labels = []
model_name = 'buffalo_l'
# app = FaceAnalysis(name=model_name, allowed_modules=['detection', 'landmark_2d_106']) # enable detection model only
app = FaceAnalysis(name=model_name, allowed_modules=['detection', 'recognition'])  # enable detection model only
app.prepare(ctx_id=0, det_size=(640, 640))


folders = list(glob.iglob(os.path.join("..\Data\Face_images", '*')))  # 경로 뭉탱이를 리스트로
names = [os.path.basename(folder) for folder in folders]  # only name
names_label=[]
for i, folder in enumerate(folders):
    name = names[i]
    videos = list(glob.iglob(os.path.join(folder, '*.*')))
    # print('1', videos)
    names_label.append(name)
    for j, img_path in enumerate(videos):
        # print(img_path)
        img = cv2.imread(img_path)
        # print(img)
        face = app.get(img)[0]
        embs.append(np.array(face.embedding))
        # print(np.array(embs).shape)
        label = [0 for i in range(len(names))]  # len(categories))]
        label[i] = 1
        labels.append(label)
        print(j, name)
    print(np.array(embs).shape)
    print("labels", np.array(labels).shape)
print(names_label)

emb_X = np.array(embs, dtype=np.int64)
emb_Y = np.array(labels, dtype=np.int64)

np.save('embX', emb_X)
np.save('embY', emb_Y)

# '''
# 0yuna_hong 1jessi 2IU 3gongyu 4dongyuk 5barack
########
# 0jessi 1IU 2yuna_hong 3gongyu 4dongyuk 5barack
emb_X = np.load('embX.npy')
emb_Y = np.load('embY.npy')

# train/val/test 분리
trainX, testX, trainY, testY = train_test_split(emb_X, emb_Y, train_size=0.7,
                                                random_state=1, shuffle=True)

testX, valX, testY, valY = train_test_split(testX, testY, test_size=0.5,
                                            random_state=1, shuffle=True)

print("train", trainX.shape, trainY.shape, "test", testX.shape, testY.shape, "val", valX.shape, valY.shape)

# 모델생성
learning_rate = 0.0001
n_epochs = 30
n_class = 12


def create_model():
    model = keras.Sequential()
    model.add(keras.layers.Dense(16, input_dim=512, activation='relu'))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(n_class, activation="softmax"))
    return model


model = create_model()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss="categorical_crossentropy", metrics=['accuracy'])
model.summary()
# 학습
history = model.fit(trainX, trainY, epochs=n_epochs, validation_data=(valX, valY))

#img = cv2.imread("/content/testpic.jpg")
#face = app.get(img)[0]
#pic = np.array([face.embedding], dtype=np.int64)
#y_predict = model.predict(pic)
#print(y_predict)

ans = np.argmax(model.predict(testX), axis=-1)
print(ans)
print("testY", testY)
for i, an in enumerate(ans):
    if testY[i][an]==1:
        print("TRUE")
    else:
        print("FALSE")
# print([True if np.argmax(Y, axis=-1)==ans else False for Y in testY])

model.save('D:\insightface_folder\lab_test\data/tf_model_12', include_optimizer=False)  # 모델 저장

# 학습과정 그래프
fig, axes = plt.subplots(2, 1)
axes[0].plot(history.history['loss'], 'b-', label='loss')
axes[0].plot(history.history['val_loss'], 'g-', label='valloss')
axes[1].plot(history.history['accuracy'], 'r-', label='accuracy')
axes[1].plot(history.history['val_accuracy'], 'b-', label='valaccuracy')
plt.xlabel("Epoch")
plt.legend()
plt.show()
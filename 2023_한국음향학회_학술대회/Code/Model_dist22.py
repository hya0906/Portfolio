"""
@author: Jiaxin Ye
@contact: jiaxin-ye@foxmail.com
"""
import logging
import numpy as np
import tensorflow.keras.backend as K
from pathlib import Path
import librosa
from datasets import load_dataset
import os
import tensorflow as tf
from tensorflow.keras.optimizers import SGD,Adam
from tensorflow.keras import callbacks
from tensorflow.keras.layers import Layer,Dense,Input
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix, accuracy_score
from Common_Model import Common_Model
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import datetime
from preprocess_func import *
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import losses
from TIMNET_dist22 import TIMNET
from tqdm import tqdm
from transformers import AutoFeatureExtractor, TFWav2Vec2Model
from wav2vec2_classification import build_model
from datasets import load_from_disk
from einops import *
from numpy import *


global count
count = 0
batchsize = 60
epch = 1

# Only log error messages
tf.get_logger().setLevel(logging.ERROR)
# Set random seed
tf.keras.utils.set_random_seed(42)

# Maximum duration of the input audio file we feed to our Wav2Vec 2.0 model.
MAX_DURATION = 1
# Sampling rate is the number of samples of audio recorded every second
SAMPLING_RATE = 16000
BATCH_SIZE = 2 #32  # Batch-size for training and evaluating our model.
NUM_CLASSES = 8  # Number of classes our dataset will have (11 in our case).
HIDDEN_DIM = 768 #1024  # Dimension of our model output (768 in case of Wav2Vec 2.0 - Base).
MAX_SEQ_LENGTH = 84351  # Maximum length of the input audio file.
# Wav2Vec 2.0 results in an output frequency with a stride of about 20ms.
MAX_FRAMES = 249
MAX_EPOCHS = 10  # Maximum number of training epochs.
WAV_DATA_POINTS = 90000 #110000

MODEL_CHECKPOINT = "facebook/wav2vec2-base"  # Name of pretrained model from Hugging Face Model Hub
# feature_extractor = AutoFeatureExtractor.from_pretrained(
#     "facebook/wav2vec2-large-960h", return_attention_mask=True
# )
#
#
# processor = AutoProcessor.from_pretrained("facebook/wav2vec2-large-960h")
# pretrained_layer = TFWav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h", from_pt=True)
# target_sampling_rate = processor.feature_extractor.sampling_rate
# print(f"The target sampling rate: {target_sampling_rate}")

LEARNING_RATE = 5e-5
feature_extractor = AutoFeatureExtractor.from_pretrained(
    MODEL_CHECKPOINT, return_attention_mask=True
)

kl = tf.keras.losses.KLDivergence()


def kl_ce_loss(t_pred, alpha):
    def Huber_loss(x, y):
        return tf.reduce_mean(tf.where(tf.less_equal(tf.abs(x - y), 1.), tf.square(x - y) / 2, tf.abs(x - y) - 1 / 2))
    def Distance_wise_potential(x):
        x_square = tf.reduce_sum(tf.square(x), -1)
        prod = tf.matmul(x, x, transpose_b=True)
        distance = tf.sqrt(tf.maximum(tf.expand_dims(x_square, 1) + tf.expand_dims(x_square, 0) - 2 * prod, 1e-12))
        mu = tf.reduce_sum(distance) / tf.reduce_sum(tf.cast(distance > 0., tf.float32))
        return distance / (mu + 1e-8)
    def Angle_wise_potential(x):
        e = tf.expand_dims(x, 0) - tf.expand_dims(x, 1)
        e_norm = tf.nn.l2_normalize(e, 2)
        return tf.matmul(e_norm, e_norm, transpose_b=True)
    def loss(y_true, y_pred):
        global count, batchsize, epch
        print(y_pred.shape)

        # Extract the relevant batch indices for t_pred and y_pred
        t_pred_batch = t_pred[batchsize * count:batchsize * (count + 1)]

        # KL divergence 계산
        try:
            s = tf.nn.l2_normalize(y_pred, 1)
            t = tf.nn.l2_normalize(t_pred_batch, 1)
            distance_loss = Huber_loss(Distance_wise_potential(s), Distance_wise_potential(t))
            angle_loss = Huber_loss(Angle_wise_potential(s), Angle_wise_potential(t))
            rkd_loss = distance_loss + angle_loss * 2


            y_pred_soft = softmax(y_pred)  # temp3
            t_pred_soft = softmax(t_pred_batch)
            #kl_loss = tf.reduce_sum(t_pred_soft * tf.math.log(t_pred_soft / (y_pred_soft + tf.keras.backend.epsilon())),
            #                        axis=-1)
            kl_loss = kl(t_pred_soft, y_pred_soft)

            # Cross entropy 계산
            y_pred = tf.nn.softmax(y_pred) #temp1
            y_true = smooth_labels(y_true, 0.1)
            ce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)

        except Exception as e:
            print("error", e)
            print("t_pred_batch.shape", t_pred_batch.shape, "y_pred.shape", y_pred.shape)
            print(t_pred.shape)
        #print("4", batchsize * count, batchsize * (count + 1))
        # KL divergence와 Cross entropy를 합친 최종 loss 계산
        #total_loss = alpha * kl_loss + (1-alpha) * ce_loss
        total_loss = 0.2 * rkd_loss + (1 - alpha) * ce_loss+ alpha * kl_loss
        #print("5", batchsize * count, batchsize * (count + 1))

        # Update batch index
        count += 1
        if count % 20 == 0:
            count = 0
        #count %= (y_true.shape[0] // batchsize)
        return total_loss

    return loss


def smooth_labels(labels, factor=0.1):
    # smooth the labels
    labels *= (1 - factor)
    labels += (factor / labels.shape[1])
    return labels

class WeightLayer(Layer):
    def __init__(self, **kwargs):
        super(WeightLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1],1),
                                      initializer='uniform',
                                      trainable=True)
        super(WeightLayer, self).build(input_shape)

    def call(self, x):
        #print("===call===")
        #print("!",x.shape)
        tempx = tf.transpose(x,[0,2,1])  #(None, 39, 8) 그냥 transpose
        #print("!!", tempx.shape)
        x = K.dot(tempx,self.kernel)  #(None, 39, 1)
        #print("!!!",x.shape)
        x = tf.squeeze(x,axis=-1)  #(None, 39)
        #print("!!!!",x.shape)
        #print("===call end===")
        return  x

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[2])

def softmax(x, axis=-1):
    ex = K.exp((x - K.max(x, axis=axis, keepdims=True))/3)
    return ex/K.sum(ex, axis=axis, keepdims=True)

def softmax_with_temperature(z, T) :
    z = np.array(z)
    z = z / T
    max_z = np.max(z)
    exp_z = np.exp(z-max_z)
    sum_exp_z = np.sum(exp_z)
    y = exp_z / sum_exp_z
    return y

class ContrastiveLoss(losses.Loss):
  def __init__(self):
    super().__init__()

  def call(self, p, z):
    p = tf.math.l2_normalize(p, axis=1)
    z = tf.math.l2_normalize(z, axis=1)
    return - reduce(einsum(p, z, 'b d, b d -> b'), 'b -> 1', 'mean')

class TIMNET_Model(Common_Model): #tf.keras.Model로 변경가능
    def __init__(self, dilation_size, filter_size, args, input_shape, class_label, **params):
        super(TIMNET_Model,self).__init__(**params)
        self.args = args
        self.data_shape = input_shape
        self.num_classes = len(class_label)
        self.class_label = class_label
        # self.dilation_size = dilation_size
        # self.filter_size = filter_size
        self.matrix = []
        self.eva_matrix = []
        self.acc = 0
        #print("TIMNET MODEL SHAPE:",input_shape)
        self.create_model()
        #self.create_model2()

    def create_model(self):
        self.inputs=Input(shape = (self.data_shape[0],self.data_shape[1]))
        self.multi_decision = TIMNET(nb_filters=65,
                                kernel_size=self.args.kernel_size,
                                nb_stacks=self.args.stack_size,
                                dilations=8,
                                dropout_rate=self.args.dropout,
                                activation = self.args.activation,
                                return_sequences=True,
                                name='TIMNET')(self.inputs)

        self.decision = WeightLayer()(self.multi_decision)
        self.predictions = Dense(self.num_classes, activation=softmax)(self.decision)
        self.teacher_model = Model(inputs = self.inputs, outputs = self.predictions)

        self.teacher_model.compile(loss = "categorical_crossentropy",
                           optimizer =Adam(learning_rate=self.args.lr, beta_1=self.args.beta1, beta_2=self.args.beta2, epsilon=1e-8),
                           metrics = ['accuracy'])
        print("Temporal create succes!")

    def create_model2(self):
        print(self.args.filter_size, self.args.dilation_size, self.args.dropout)
        filter_size, dilation = self.args.filter_size, self.args.dilation_size
        self.inputs = Input(shape=(self.data_shape[0], self.data_shape[1]))
        self.multi_decision = TIMNET(nb_filters=filter_size,
                                     kernel_size=self.args.kernel_size,
                                     nb_stacks=self.args.stack_size,
                                     dilations=dilation,
                                     dropout_rate=self.args.dropout,
                                     activation=self.args.activation,
                                     return_sequences=True,
                                     name='TIMNET')(self.inputs)

        self.decision = WeightLayer()(self.multi_decision)
        self.predictions = Dense(self.num_classes)(self.decision)
        self.student_model = Model(inputs=self.inputs, outputs=self.predictions)
        print("Temporal create success!")


    def test(self,name, path, alpha, alpha_result_list):
        # https://github.com/huggingface/transformers/issues/16249 TFtrainer이제 사용못함.
        audios_dir = "C:\\Users\\711_2\\Desktop\\Yuna_Hong\\speech_expression\\WAV2MFCC\\WAV2VEC\\new_data\\RAVDESS_aug_wavfile_16000_padding84351_nopitch"
        data_path = 'C:\\Users\\711_2\\Desktop\\Yuna_Hong\\speech_expression\\WAV2MFCC\\WAV2VEC\\distillation - 복사본(2)_231014 - 복사본\\FineTuningWav2Vec2_out'
        #cache_dir = "C:\\Users\\yuna_hong\\Desktop\\Yuna_Hong\\WAV2MFCC\\WAV2VEC\\cache_dir"
        i=1
        scores = np.zeros(10)
        avg_accuracy = 0
        avg_loss = 0
        x_feats = []
        y_labels = []
        graphs = []
        result_list = []
        print("alpha: ", alpha)
        #alpha = 0.1
        input_column = "path"  # Name of the column that will contain the path of the recordings
        output_column = "emotion"
        now = datetime.datetime.now()
        now_time = datetime.datetime.strftime(now, '%Y-%m-%d_%H-%M-%S')
        #out_dir_models = os.path.join(data_path, "Models")  # out path to save trained models
        for fold in range(5):  # 5-CV strategy
            print(f"====fold {fold}====")

            # Call Student Model
            self.create_model2()

            out_dir_models_path = os.path.join(data_path, now_time, "fold" + str(fold))
            #save_path = os.path.join(data_path, now_time, "fold" + str(fold))
           # os.environ['TRANSFORMERS_CACHE'] = os.path.join(cache_dir, now_time, "fold" + str(fold))
           # os.environ['HF_DATASETS_CACHE'] = os.path.join(cache_dir, now_time, "fold" + str(fold))

           #  os.makedirs(out_dir_models_path, exist_ok=True)
           #  print("SAVING DATA IN: ", out_dir_models_path)
           #
           #  df = prepare_RAVDESS_DS(audios_dir)
           #  _, _ = generate_train_test(fold, df, out_dir_models_path)
           #
           #  data_files = {
           #      "train": os.path.join(out_dir_models_path, "train.csv"),
           #      "validation": os.path.join(out_dir_models_path, "test.csv"),
           #  }
           #
           #  # Load data
           #  dataset = load_dataset("csv", data_files=data_files, delimiter="\t")
           #  train_dataset = dataset["train"]
           #  eval_dataset = dataset["validation"]
           #  print("Processing fold: ", str(fold), " - actors in Train fold: ", set(train_dataset["actor"]))
           #  print("Processing fold: ", str(fold), " - actors in Eval fold: ", set(eval_dataset["actor"]))
           #  global label_list
           #  label_list = train_dataset.unique(output_column)
           #  label_list.sort()  # Let's sort it for determinism
           #  num_labels = len(label_list)
           #  print(f"A classification problem with {num_labels} classes: {label_list}")

            ##########mfcc데이터생성
            # Get Augmented Data
            train = np.load('../new_data/aug_MFCC_traindata_84351_7_2023-10-15_18-48-50.npy', allow_pickle=True).item()
            train_features, train_labels = train["x"], train["y"]
            test = np.load('../new_data/aug_MFCC_testdata_84351_7_2023-10-15_18-48-50.npy', allow_pickle=True).item()
            test_features, test_labels = test["x"], test["y"]
            print(train_features.shape, test_features.shape, train_labels.shape)


            # # Preprocess Wav2Vec2 Data
            # train_dataset = train_dataset.select(
            #     [i for i in range((len(train_dataset) // BATCH_SIZE) * BATCH_SIZE)]
            # )
            # eval_dataset = eval_dataset.select(
            #     [i for i in range((len(eval_dataset) // BATCH_SIZE) * BATCH_SIZE)]
            # )
            #
            # print("Generating test...")
            # eval_dataset = eval_dataset.map(
            #     preprocess_function,
            #     batch_size=100,
            #     batched=True,
            #     # num_proc=4
            # )
            # eval_dataset.save_to_disk("./huggingface_dataset")
            #
            # print("Generating training...")
            # train_dataset = train_dataset.map(
            #     preprocess_function,
            #     batch_size=100,
            #     batched=True,
            #     # num_proc=4
            # )
            # train_dataset.save_to_disk("./huggingface_dataset")
            # print("train dataset", train_dataset)

            train_dataset = load_from_disk("../wav2vec_huggingface_dataset/train")
            eval_dataset = load_from_disk("../wav2vec_huggingface_dataset/test")
            print(train_dataset, eval_dataset)

            train = train_dataset.shuffle(seed=42).with_format("numpy")[:]
            test = eval_dataset.shuffle(seed=42).with_format("numpy")[:]

            train_x = {x: y for x, y in train.items() if x != "label"}
            test_x = {x: y for x, y in test.items() if x != "label"}


            # Call pretraiend Teacher Model
            teacher_model = build_model()#self.create_model2()
            teacher_model.summary()

            #weight_path = f".\\RAVDESS_mfcc_60_46_main2_2023-06-03_00-53-18_92.847\\{str(self.args.split_fold)}-fold_weights_best_{str(i)}.hdf5"
            #weight_path = path + '/' + str(self.args.split_fold) + "-fold_weights_best_" + str(i) + ".hdf5"
            weight_path = f"C:\\Users\\711_2\\Desktop\\Yuna_Hong\\speech_expression\\WAV2MFCC\\WAV2VEC\\RAVDESS_RAVDESS_aug_wavfile_16000_padding84351_7class_2023-10-15_10-12-46\\10-fold_weights_best_0_0.84000.hdf5"
            teacher_model.load_weights(weight_path)

            #teacher_model.evaluate(train_x, train["labels"], batch_size= BATCH_SIZE)
            print("!!!!!!!!!!!!!!!!!!!!!!!!")

            # Make Teacher Soft Label
            y_train_kf = teacher_model.predict(train_x, batch_size= BATCH_SIZE)
            print(y_train_kf.shape, y_train_kf[0])
            y_train_kf = softmax_with_temperature(y_train_kf, 3)
            y_train_kf = smooth_labels(y_train_kf, 0.1)

            self.student_model.compile(loss= kl_ce_loss(y_train_kf, alpha),
                                       optimizer=Adam(learning_rate=self.args.lr, beta_1=self.args.beta1,
                                                      beta_2=self.args.beta2, epsilon=1e-8),
                                       metrics=['accuracy'])

            #train_labels = smooth_labels(train_labels, 0.1)

            os.makedirs(f'./new_weight/{name}_kl_ce{alpha}_{now_time}', exist_ok=True)
            weight_path_=f"./new_weight/{name}_kl_ce{alpha}_{now_time}/10-fold_weights_best_{str(i)}.hdf5"
            checkpoint = callbacks.ModelCheckpoint(weight_path_, monitor='val_accuracy', verbose=1,
                                                   save_weights_only=True, save_best_only=True, mode='max')


            h = self.student_model.fit(train_features, train_labels, validation_data=(test_features, test_labels), batch_size= 60, epochs=300, verbose=1,callbacks=[checkpoint])

            epochs = range(1, 300 + 1)
            history_dict = h.history
            loss = history_dict['loss']
            val_loss = history_dict['val_loss']
            acc = history_dict['accuracy']
            val_acc = history_dict['val_accuracy']

            plt.subplot(121)
            plt.plot(epochs, acc, 'g', label='Train acc')
            plt.plot(epochs, val_acc, 'b', label='val acc')
            plt.title('ACC')
            plt.xlabel('Epochs')
            plt.ylabel('ACC')
            plt.legend()

            plt.subplot(122)
            plt.plot(epochs, loss, 'g', label='Train loss')
            plt.plot(epochs, val_loss, 'b', label='val loss')
            plt.title('LOSS')
            plt.xlabel('Epochs')
            plt.ylabel('LOSS')
            plt.legend()
            #plt.show()
            plt.savefig(f"./graphs/{name}_{now_time}.jpg")
            plt.clf()

            # self.student_model.load_weights(weight_path_)
            # best_eva_list = self.student_model.evaluate(test_features, test_labels, batch_size=64)
            # alpha_result_list.append(best_eva_list[1])
            #
            # y_pred = self.student_model.predict(test_features, batch_size= 60)
            # print(y_pred.shape)
            # print("f1 score: ", f1_score(np.argmax(y_pred, axis=1), np.argmax(test_labels, axis=1), average='weighted'))
            if fold == 0:
                return x_feats, y_labels, alpha_result_list
        #     y_pred = self.student_model.predict(x_test, batch_size= self.args.batch_size)
        #     print(y_pred)
        #     y_pred = np.argmax(y_pred, axis=1)
        #     print(type(y_pred))
        #     print(y_pred)
        #     print(type(self.teacher_model.predict(x_test)))
        #     print(self.teacher_model.predict(x_test, batch_size= self.args.batch_size))
        #
        #     self.student_model.load_weights(weight_path_)
        #     best_eva_list = self.student_model.evaluate(x_test, y_test, batch_size= self.args.batch_size)
        #     avg_loss += best_eva_list[0]
        #     avg_accuracy += best_eva_list[1]
        #     result_list.append(best_eva_list[1])
        #     #print("avg_accuracy",avg_accuracy)
        #     print(str(i) + '_Model evaluation: ', best_eva_list, "   Now ACC:",str(round(avg_accuracy * 10000) / 100 / i))
        #
        #
        # alpha_result_list.append(result_list)
        # print(result_list,"\navg result",sum(result_list)/10)
        # print("alpha_result_list: ", alpha_result_list)
        return x_feats, y_labels, alpha_result_list


if "__name__"== "__main__":
    TM = TIMNET_Model()
    TM.test()

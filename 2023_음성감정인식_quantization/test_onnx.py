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
# from memory_profiler import profile
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


d = np.random.rand(1, 40000).astype(np.float32)
print(np.array(d))

# file_name = "/home/yuna/code/saved_onnx/wav2vec2-large-xlsr-53-english.onnx"
file_name = "/home/yuna/code/wav2vec2-large-xlsr-53-english.quant+.onnx"
# model = onnx.load(file_name)
# nodes = model.graph.node
# names = [x.name for x in nodes]
# all = []
# for i in names:mmm
#     all.append(i.split("_")[0])
# # print(set(all))
# print(['Conv', 'Gather', 'Mul', 'Slice', 'Constant', 'Div', 'Concat', 'Softmax', 'MatMul', 'Sqrt', 'Transpose', 'Reshape', 'Shape', 'Unsqueeze', 'Pow', 'ReduceMean', 'Identity', 'Add', 'Sub', 'Erf'])
# list_name = []
# for x in nodes:
#     if "MatMul" in x.name:
#         list_name.append(x.name)


#print("\n",list_name)
# session = ort.InferenceSession("/home/yuna/code/w2v_output_xlsr/model_quantized.onnx", providers=["CUDAExecutionProvider",])
#session = ort.InferenceSession("/home/yuna/code/wav2vec2-large-xlsr-53-english.quant+.onnx", providers=["CUDAExecutionProvider",])

session = ort.InferenceSession(file_name, providers=["CUDAExecutionProvider",])
# session = ort.InferenceSession("/home/yuna/code/wav2vec2-large-xlsr-53-english.quant_.onnx", providers=["CUDAExecutionProvider",])
ort_inputs = {session.get_inputs()[0].name: d}

all = []
for i in range(11):
    start = time()
    ort_outs = session.run(None, ort_inputs)
    end = time()
    all.append(end-start)

print(sum(all[1:])/10)

# from transformers import pipeline

# classifier = pipeline(model="superb/wav2vec2-base-superb-ks")
# classifier("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac")


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
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

model = ORTModelForFeatureExtraction.from_pretrained("/home/yuna/code/w2v_output_xlsr", file_name="model_quantized.onnx")
cls_pipeline = pipeline("feature-extraction", model=model)
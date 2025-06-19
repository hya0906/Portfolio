import argparse
import numpy as np
import torchaudio
import librosa
from transformers import Wav2Vec2Processor
import math
from Wav2VecAuxClasses import *
import pyaudio
import wave
from datetime import datetime
from transformers import AutoConfig
from tqdm import tqdm
import keyboard
import onnxruntime

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 3

audio = pyaudio.PyAudio()
stream = audio.open(format=pyaudio.paInt16,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=2,
                    frames_per_buffer=CHUNK)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Configuration of setup")
    parser.add_argument('--audio', type=str,
                        default='samples/01-01-03-02-01-01-02.wav',
                        help='Path audio data')
    parser.add_argument('-trainedModel', '--trained_model', type=str,
                        default=r'C:\Users\711_2\Desktop\Yuna_Hong\speech_expression\DEMO\checkpoint_2023-12-11_11-00-02_all_teacher0.7739\test_epoch13_0.7739.pth',
                        help='Path to the trained model')
    parser.add_argument('-model', '--model_id', type=str,
                        help='Model identificator in Hugging Face library [default: jonatasgrosman/wav2vec2-large-xlsr-53-english]',
                        default='jonatasgrosman/wav2vec2-large-xlsr-53-english')

    args = parser.parse_args()

    num_labels = 7
    label_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    config = AutoConfig.from_pretrained(
        args.model_id,  # path to the model of HuggingFace lib. that we will use as baseline to fine-tune.
        num_labels=num_labels,  # num classes
        label2id={label: i for i, label in enumerate(label_list)},  # dict that maps emotions -> numbers
        id2label={i: label for i, label in enumerate(label_list)},  # dict that maps numbers -> emotions
        finetuning_task="wav2vec2_clf",
    )
    pooling_mode = "mean"
    setattr(config, 'pooling_mode', pooling_mode)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = Wav2Vec2Processor.from_pretrained(args.model_id)
    model = Wav2Vec2ForSpeechClassification.from_pretrained(args.model_id, config=config).to(device)
    checkpoint = torch.load(args.trained_model)
    model.load_state_dict(checkpoint['model_state_dict'])

    label2emotion = {0: 'Angry',
                     1: 'Disgust',
                     2: 'Fear',
                     3: 'Happy',
                     4: 'Neutral',
                     5: 'Sad',
                     6: 'Surprised'}
    # label2emotion = {
    #     0: 'Neutral',
    #     1: 'Happy',
    #     2: 'Sad',
    #     3: 'Angry',
    #     4: 'Fear',
    #     5: 'Disgust',
    #     6: 'Surprise'
    # }

    stream = audio.open(format=pyaudio.paInt16,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        input_device_index=2,
                        frames_per_buffer=CHUNK)

    # start Recording
    now = datetime.now()
    now_time = datetime.strftime(now, '%Y-%m-%d_%H-%M-%S')
    WAVE_OUTPUT_FILENAME = f"./record/{now_time}.wav"
    #
    print("recording...")

    frames = []

    while True:
        data = stream.read(CHUNK)
        frames.append(data)
        if keyboard.is_pressed('space'):
            break

    print("finished recording")
    #this is my car - angry
    #I am so sad today - sad
    # is that you? - surprise
    #this is my house - neutral


     # stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()

    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

    # WAVE_OUTPUT_FILENAME = r".\rec\happy.wav"
    speech_array, sampling_rate = torchaudio.load(WAVE_OUTPUT_FILENAME)
    speech_array = speech_array.squeeze().numpy()
    speech_array = librosa.resample(np.asarray(speech_array), orig_sr=sampling_rate,
                                    target_sr=processor.feature_extractor.sampling_rate)

    file_name = "./SER_model/wav2vec2-large-xlsr-53-english.quant_static.onnx"
    session = onnxruntime.InferenceSession(file_name, providers=["CUDAExecutionProvider", ])
    ort_inputs = {session.get_inputs()[0].name: speech_array.reshape(1, -1).astype(np.float32)}
    ort_outs = session.run(None, ort_inputs)[0][0]
    values, indices = torch.tensor(ort_outs).softmax(dim=-1).topk(5)

    print("\nTop predictions:\n")
    for value, index in zip(values, indices):
        print(f"{label2emotion[int(index)]:>16s}: {100 * value.item():.2f}%")



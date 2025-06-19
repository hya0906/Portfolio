import pyaudio
import wave
from datetime import datetime

po = pyaudio.PyAudio()

for index in range(po.get_device_count()):
    desc = po.get_device_info_by_index(index)
    #if desc["name"] == "record":
    print("DEVICE: %s  INDEX:  %s  RATE:  %s " %  (desc["name"], index,  int(desc["defaultSampleRate"])))


FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 5

audio = pyaudio.PyAudio()
stream = audio.open(format=pyaudio.paInt16,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=2,
                    frames_per_buffer=CHUNK)
while True:

    # start Recording
    now = datetime.now()
    now_time = datetime.strftime(now, '%Y-%m-%d_%H-%M-%S')
    WAVE_OUTPUT_FILENAME = f"./record/{now_time}.wav"

    print("recording...")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("finished recording")

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

    x = input("계속? (yes/no)")
    if x is "no":
        break
    else:
        frames = []
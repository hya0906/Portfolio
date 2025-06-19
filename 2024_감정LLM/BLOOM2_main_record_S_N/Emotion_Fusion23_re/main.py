from Emotion_Fusion_VA import *

def start_AVrecording(camindex=0, fps=30, audio_index=0, sample_rate=16000):
    global video_thread
    global audio_thread

    condition = threading.Condition()
    video_thread = VideoRecorder(condition, camindex=camindex, fps=fps)
    audio_thread = AudioRecorder(condition, audio_index=audio_index, rate=sample_rate)
    # ER_thread = Emotion_Fusion()

    audio_thread.start()
    video_thread.start()
    # ER_thread.start()
    return

def stop_AVrecording():
    audio_thread.stop()
    video_thread.stop()

def main():

    av_open = True
    start_AVrecording()

    while av_open:
        time.sleep(0)
        AV_Capture(video_thread)

        if video_thread.capture_stop_index == 1:
            stop_AVrecording()
            break


if __name__ == "__main__":
    main()
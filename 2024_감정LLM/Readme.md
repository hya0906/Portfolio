**기간**: 2024년 2월 - 2024년 4월  
**제목**: 실시간으로 Facial expression과 Speech emotion recognition으로 사용자의 감정을 인식하여 감정적인 응답을 도출하는 LLM (모델미포함)
### 과제설명
2023_음성감정인식_quantization에서 했던 결과(quantized Wav2Vec2) + 박사가 한 얼굴감정인식<sup>[1]</sup> + LLM 합친버전    
 마이크로 말하면 STT로 변환 후 프롬프트에 얼굴 감정과 음성 감정, 문장의 감정 context가 모두 합쳐서 LLM에 넣어서 최종 결과 냄.
 
<p align="center" style="color:gray">
  <!-- 마진은 위아래만 조절하는 것이 정신건강에 좋을 듯 하다. 이미지가 커지면 깨지는 경우가 있는 듯 하다.-->
  <img src="https://github.com/user-attachments/assets/e76ca7de-c093-4253-b5bd-8af22936e3d8" />
  <strong>[Emotion LLM]</strong>
</p> 

### 파일설명
1. total_emotion.py - 전체버전 실행 (필요한 버전 간편하게 실행가능)
2. interact_gguf.py - 각 버전 main 실행  
3. mldata 폴더 - huggingface 에서 다운받은 LLM 파일 넣는 위치

<a name="각주 이름">1</a>: Lee, Bokyeung, et al. "Dropout Connects Transformers and CNNs: Transfer General Knowledge for Knowledge Distillation." 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). IEEE, 2025.

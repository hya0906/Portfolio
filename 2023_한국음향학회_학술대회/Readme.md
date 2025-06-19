**기간**: 2023년 4월 - 2023년 10월   
**제목**: WAV에서 MFCC로의 지식 증류를 이용한 음성감정인식   
**발표ppt**: 추계학술대회_11_02_수정본   

<p align="center" style="color:gray">
  <!-- 마진은 위아래만 조절하는 것이 정신건강에 좋을 듯 하다. 이미지가 커지면 깨지는 경우가 있는 듯 하다.-->
  <img src="https://github.com/user-attachments/assets/4978ca52-50fc-4c5a-92ac-017c5b6504e4" width="600"/>   
</p> 
<p align="center"><strong>[Overall Process]</strong></p>    

### 결과
2023.10.06   
Teacher - wav2vec2-base   
Student - Timnet

original Timnet (60/8/0/05) accuracy: 0.6900 (대부분 69 또는 71이 나옴 6,7번 돌리면 한번 72 나옴) / f1 score: 0.7015123137605948

![originalTIMNET_confusion_matirx](https://github.com/hya0906/ISPL_speech_recognition/assets/59861622/aae20315-025e-4a98-87c5-5cedbb96bfb4)

distilled Timnet (60/8/0.05/alpha0.5) accuracy:0.7867 / f1 score: 0.8010737926748237

![distilledTIMNET_confusion_matrix](https://github.com/hya0906/ISPL_speech_recognition/assets/59861622/6d0fafab-1558-4f77-93d8-9dc37be8b778)

+2025.06.12추가   
distilled Timnet RKD (60/8/0.05/alpha0.5, beta0.2) accuracy:0.81 / f1 score: 0.8148    

<p>
  <img src="https://github.com/user-attachments/assets/e3c8a028-77a8-446f-91d8-0d52571b8acd" width="600"/>
</p>

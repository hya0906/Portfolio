**기간**: 2023년 10월 - 2023년 12월   
**제목**: 음성감정인식모델 경량화 (Quantization)   
**설명**: ONNX dynamic quantization을 사용하여 finetuning한 Wav2Vec2-large 경량화


<table align="center">
    <p align="center"><strong>RAVDESS dataset 성능</strong></p>
    <tbody>
        <tr>
            <td><strong>Model</strong></td>
            <td><strong>5-CV Accuracy</strong></td> <!-- 첫 번째 행에 두 개의 셀 추가 -->
        </tr>
        <tr>
            <td rowspan="2">xlsr-Wav2Vec2.0 <br>(fine-tuning)</td>
            <td>81.23% (8 classes)</td>
        </tr>
        <tr>
            <td>83.83% (7 classes)</td>
        </tr>
    </tbody>
</table>   


<table align="center">
    <p align="center"><strong>RAVDESS + CREMA-D (Fold 0)</strong></p>
    <tbody>
        <tr>
            <td rowspan="2"><strong>Train</strong></td>
            <td rowspan="2"><strong>Test</strong></td>
            <td colspan="2"><strong>Fold 0</strong></td> <!-- 3열 -->
        </tr>
        <tr>
            <td><strong>5-CV <br>accuracy</strong></td>
            <td><strong>Weighted <br>F1-Score</strong></td> <!-- 3열 -->
        </tr>
        <tr>
            <td>RAVDESS(7)</td>
            <td>RAVDESS(7)</td>
            <td>88.00%</td>
            <td>87.74%</td>
        </tr>
        <tr>
            <td>CREMA-D</td>
            <td>CREMA-D</td>
            <td>74.47%</td>
            <td>74.14%</td>
        </tr>
        <tr>
            <td rowspan="2">RAVDESS(7)+CREMA-D</td>
            <td>RAVDESS(7)</td>
            <td>88.67%</td>
            <td>88.66%</td>
        </tr>
        <tr>
            <td>CREMA-D</td>
            <td>74.54%</td>
            <td>74.09%</td>
        </tr>
    </tbody>
</table>   

<p align="center" style="color:gray">
  <!-- 마진은 위아래만 조절하는 것이 정신건강에 좋을 듯 하다. 이미지가 커지면 깨지는 경우가 있는 듯 하다.-->
  <img src="https://github.com/user-attachments/assets/542a4c89-fa24-4644-86c2-c7a8b030157d"/>
  <strong>[Confusion matrix 비교]</strong>
</p> 

-	Inference time: RTX 2080ti 환경에서 2.5 seconds의 wav file 기준 0.02404 seconds 소요

**기간**: 2024년 3월 - 2025년 1월   
**제목**: 의료분야에서 사용자 맞춤형 조언을 제공하는 RAG기반 Private LLM 개발   
### 과제설명
사용자가 증상이나 질병을 말하면, 대화 내용과 관련된 개인의료정보를 바탕으로 조언하는 시스템.

(1) 모든 데이터를 저장하는 대신 필요한 정보만 압축적으로 저장함으로써 외부 장기 메모리의 필요 저장 용량을 줄임.   
(2) 인간의 기억 메커니즘과 유사하게 작동하는 기억 갱신 및 망각 프로세스를 포함.   
(3) 기존의 RAG 기반 시스템과 달리, 원래 쿼리에서 관련 검색 방향을 포괄하는 추가 쿼리를 생성하여 복잡한 쿼리를 처리가능.   
이 방법들을 통해 메모리 용량을 줄이고, 사용자의 과거 의료 내력와 연관지어 보다 적합한 답변을 제공할 수 있다.   

<p align="center" style="color:gray">
  <!-- 마진은 위아래만 조절하는 것이 정신건강에 좋을 듯 하다. 이미지가 커지면 깨지는 경우가 있는 듯 하다.-->
  <img src="https://github.com/user-attachments/assets/f32296f4-1236-4a9a-a009-d628262b91e5" />
  <strong>[Total Process]</strong>
</p> 

<p align="center" style="color:gray">
  <!-- 마진은 위아래만 조절하는 것이 정신건강에 좋을 듯 하다. 이미지가 커지면 깨지는 경우가 있는 듯 하다.-->
  <img src="https://github.com/user-attachments/assets/a590d63a-54c8-4438-9412-68fd5c41f90f" />
  <strong>[Comparison of answers with and without using derived queries]</strong>
</p> 

### 코드설명
1. fastapi를 이용한 코드 - client.py + server_v3.py
2. UI가 없고 client와 server을 하나의 코드로 만든 버전 - main_total_dialog_medi_realtime.py
3. mem0 - 사용한 라이브러리 폴더


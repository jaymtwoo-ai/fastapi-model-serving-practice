# FastAPI를 활용한 인공지능 모델 서빙 실습

## 1. 실습 목적

FastAPI를 활용하여 학습된 머신러닝 모델을 API 형태로 제공하는 모델 서빙 구조를 구현한다.

## 2. 실습 내용

- Iris 데이터셋을 활용한 머신러닝 모델 학습
- RandomForestClassifier 모델 생성
- 학습된 모델을 `model.pkl` 파일로 저장
- FastAPI에서 저장된 모델을 불러오기
- `/predict` API를 통해 입력값을 받고 예측 결과 반환
- Swagger UI에서 모델 예측 API 테스트

## 3. 파일 구성

```text
train_model.py      # 머신러닝 모델 학습 및 model.pkl 저장
model.pkl           # 학습된 모델 파일
main.py             # FastAPI 모델 서빙 API
requirements.txt    # 필요한 패키지 목록
README.md           # 실습 설명 문서
```

## 4. 실행 방법

### 1) 패키지 설치

```bash
pip install -r requirements.txt
```

### 2) 모델 학습 및 저장

```bash
python train_model.py
```

실행 후 `model.pkl` 파일이 생성된다.

### 3) FastAPI 서버 실행

```bash
uvicorn main:app --reload
```

## 5. Swagger 확인 주소

```text
http://localhost:8000/docs
```

## 6. API 테스트 예시

### GET /

응답 예시:

```json
{
  "message": "FastAPI ML Model Serving API"
}
```

### POST /predict

입력 예시:

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

출력 예시:

```json
{
  "prediction": 0,
  "prediction_name": "setosa"
}
```

## 7. 정리

이번 실습에서는 FastAPI를 활용하여 학습된 머신러닝 모델을 API로 서빙하는 구조를 구현하였다.  
모델을 학습한 뒤 `model.pkl`로 저장하고, FastAPI 서버에서 해당 모델을 불러와 `/predict` 엔드포인트를 통해 예측 결과를 반환하도록 구성하였다.

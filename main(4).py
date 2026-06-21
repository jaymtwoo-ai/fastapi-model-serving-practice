from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI(title="FastAPI ML Model Serving Practice")

# =========================
# 저장된 머신러닝 모델 불러오기
# =========================
with open("model.pkl", "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
target_names = model_data["target_names"]


# =========================
# API 동작 확인용 기본 엔드포인트
# =========================
@app.get("/")
def root():
    return {"message": "FastAPI ML Model Serving API"}


# =========================
# 예측 요청 데이터 형식 정의
# =========================
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# =========================
# ML 모델 예측 API
# =========================
@app.post("/predict")
def predict_iris(data: IrisInput):
    input_data = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    prediction = model.predict(input_data)[0]
    prediction_name = target_names[prediction]

    return {
        "prediction": int(prediction),
        "prediction_name": prediction_name
    }

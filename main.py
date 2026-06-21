from fastapi import FastAPI, Path, Query
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

app = FastAPI(title="FastAPI AI Model Serving Practice")

# =========================
# 1. 기본 API
# =========================
@app.get("/")
def root():
    return {"message": "Hello FastAPI"}


# =========================
# 2. Path Parameter 실습
# 요구사항: GET /items/{item_name}
# item_name: 최대 글자수 6
# =========================
@app.get("/items/{item_name}")
def get_item(item_name: str = Path(..., max_length=6)):
    return {"item_name": item_name}


# =========================
# 3. Query Parameter 실습
# 요구사항: GET /products/search?q=apple&limit=5
# =========================
@app.get("/products/search")
def product_search(
    q: str = Query(..., min_length=2, max_length=30),
    limit: int = Query(10, ge=1, le=100)
):
    return {"q": q, "limit": limit}


# =========================
# 4. 간단한 ML 모델 서빙 실습
# Iris 데이터를 사용한 RandomForest 예측 API
# =========================
iris = load_iris()
X = iris.data
y = iris.target

model = RandomForestClassifier(random_state=42)
model.fit(X, y)


class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.post("/predict")
def predict_iris(data: IrisInput):
    input_data = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    pred = model.predict(input_data)[0]
    pred_name = iris.target_names[pred]

    return {
        "prediction": int(pred),
        "prediction_name": pred_name
    }

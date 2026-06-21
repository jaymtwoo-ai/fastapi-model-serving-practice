from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle

# Iris 데이터셋 불러오기
iris = load_iris()
X = iris.data
y = iris.target

# 머신러닝 모델 학습
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# 모델과 클래스 이름 저장
model_data = {
    "model": model,
    "target_names": iris.target_names
}

with open("model.pkl", "wb") as f:
    pickle.dump(model_data, f)

print("모델 저장 완료: model.pkl")

FastAPI를 활용한 인공지능 모델 서빙 실습

1. 실습 목적
FastAPI를 활용하여 기본 API를 만들고, Path Parameter, Query Parameter, 간단한 머신러닝 모델 서빙 API를 구현한다.

2. 실습 내용
- FastAPI 기본 구조 작성
- GET / 기본 API 만들기
- Path Parameter 실습: /items/{item_name}
- Query Parameter 실습: /products/search?q=apple&limit=5
- RandomForestClassifier를 활용한 Iris 품종 예측 API 구현
- Swagger UI에서 API 테스트

3. 실행 방법

필요 패키지 설치:
pip install fastapi uvicorn scikit-learn

서버 실행:
uvicorn main:app --reload

또는 fastapi가 설치되어 있다면:
fastapi dev main.py

4. Swagger 확인 주소

http://localhost:8000/docs

5. 테스트 예시

GET /
결과:
{"message": "Hello FastAPI"}

GET /items/apple
결과:
{"item_name": "apple"}

GET /products/search?q=apple&limit=5
결과:
{"q": "apple", "limit": 5}

POST /predict
입력 예시:
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}

출력 예시:
{
  "prediction": 0,
  "prediction_name": "setosa"
}

6. 정리
이번 실습을 통해 FastAPI의 기본 구조와 라우팅 방식, Path Parameter와 Query Parameter 사용법을 익혔다.
또한 학습된 머신러닝 모델을 API 형태로 제공하는 간단한 모델 서빙 구조를 구현하였다.

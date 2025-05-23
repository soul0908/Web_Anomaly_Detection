# Web Anomaly Detection
이 프로젝트는 웹 트래픽 데이터를 기반으로 이상 탐지를 수행하는 다양한 딥러닝 및 머신러닝 모델을 구현한 것입니다. LSTM, GRU, DNN, XGBoost 등의 모델을 활용하여 웹 트래픽의 비정상적인 패턴을 식별하고자 합니다.


## 📁 프로젝트 구조
- LSTM/: LSTM(Long Short-Term Memory) 모델을 사용한 이상 탐지 구현 코드
- GRU.py: GRU(Gated Recurrent Unit) 모델을 사용한 이상 탐지 구현 코드
- DNN.py: 심층 신경망(Deep Neural Network)을 활용한 이상 탐지 구현 코드
- XGbost/: XGBoost 알고리즘을 사용한 이상 탐지 구현 코드
- anomlay.py: 이상 탐지의 핵심 로직을 담고 있는 스크립트
- README.md: 프로젝트에 대한 설명과 사용 방법을 담은 문서

## 🔍 모델 설명
- LSTM: 시간 순서가 중요한 데이터를 처리하는 데 효과적인 순환 신경망의 일종으로, 장기 의존성 문제를 해결합니다.
- GRU: LSTM의 변형으로, 더 간단한 구조를 가지면서도 유사한 성능을 제공합니다.
- DNN: 다층 퍼셉트론 구조를 가진 심층 신경망으로, 복잡한 패턴 인식에 활용됩니다.
- XGBoost: 결정 트리 기반의 앙상블 학습 기법으로, 높은 예측 성능과 효율성을 자랑합니다.

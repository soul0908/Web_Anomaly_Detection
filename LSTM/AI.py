import sqlite3
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Concatenate, Input
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Bidirectional
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve


# 정상 데이터와 비정상 데이터를 각각 다른 DB에서 불러오는 함수
def load_data_from_db(normal_db_path='normal_data.db', abnormal_db_path='abnormal_data.db'):
    # 정상 데이터 불러오기
    conn_normal = sqlite3.connect(normal_db_path)
    query_normal = 'SELECT * FROM normal_data'
    df_normal = pd.read_sql_query(query_normal, conn_normal)
    conn_normal.close()

    # 비정상 데이터 불러오기
    conn_abnormal = sqlite3.connect(abnormal_db_path)
    query_abnormal = 'SELECT * FROM abnormal_data'
    df_abnormal = pd.read_sql_query(query_abnormal, conn_abnormal)
    conn_abnormal.close()

    # 두 데이터프레임 결합 (정상 + 비정상)
    df = pd.concat([df_normal, df_abnormal], ignore_index=True)
    return df

# 데이터베이스 파일 경로 설정
normal_db_path = 'D:/Pay1oad 프로젝트/normal_data.db'
abnormal_db_path = 'D:/Pay1oad 프로젝트/abnormal_data.db'

# 데이터 로드
data = load_data_from_db(normal_db_path, abnormal_db_path)

method_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
user_agent_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

method_encoded = method_encoder.fit_transform(data[['Method']])
user_agent_encoded = user_agent_encoder.fit_transform(data[['User-Agent']])

# 'URL' 컬럼을 텍스트 데이터로 처리
MAXLEN = 25
VOCAB_SIZE = 5000
tokenizer = Tokenizer(num_words=VOCAB_SIZE, filters='', oov_token='<OOV>')
tokenizer.fit_on_texts(data['URL'])
url_sequences = tokenizer.texts_to_sequences(data['URL'])
url_padded = pad_sequences(url_sequences, maxlen=MAXLEN, padding='post')

# X는 URL, Method, User-Agent 피처들을 합쳐서 배열로 묶어줍니다.
X = pd.concat([pd.DataFrame(url_padded), pd.DataFrame(method_encoded), pd.DataFrame(user_agent_encoded)], axis=1)

# 레이블은 classification (정상: 0, 비정상: 1)
y = data['classification'].values

# X와 y의 크기 확인
print(f"X shape: {X.shape}")
print(f"y shape: {len(y)}")

# 데이터셋 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 모델 구축
def build_lstm_classifier(vocab_size, maxlen, input_shape_method, input_shape_user_agent):
    # URL 입력 (시퀀스)
    input_url = Input(shape=(maxlen,))
    embedding = Embedding(input_dim=vocab_size, output_dim=64, input_length=maxlen)(input_url)
    lstm = Bidirectional(LSTM(64, dropout=0.3))(embedding)  # ✅ Dropout + Bidirectional 추가

    # Method 입력
    input_method = Input(shape=(input_shape_method,))
    dense_method = Dense(32, activation='relu')(input_method)

    # User-Agent 입력
    input_user_agent = Input(shape=(input_shape_user_agent,))
    dense_user_agent = Dense(32, activation='relu')(input_user_agent)

    # 결합 및 출력
    concatenated = Concatenate()([lstm, dense_method, dense_user_agent])
    dense1 = Dense(32, activation='relu')(concatenated)
    output = Dense(1, activation='sigmoid')(dense1)

    model = Model(inputs=[input_url, input_method, input_user_agent], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 모델 학습
input_shape_method = method_encoded.shape[1]  # One-hot encoding된 method의 차원
input_shape_user_agent = user_agent_encoded.shape[1]  # One-hot encoding된 user_agent의 차원

model = build_lstm_classifier(VOCAB_SIZE, MAXLEN, input_shape_method, input_shape_user_agent)

# 모델 학습 (X_train의 각 피처를 분리하여 입력)
model.fit([X_train.iloc[:, :MAXLEN], X_train.iloc[:, MAXLEN:MAXLEN + input_shape_method], X_train.iloc[:, MAXLEN + input_shape_method:]], y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)

# 모델 평가
loss, acc = model.evaluate([X_test.iloc[:, :MAXLEN], X_test.iloc[:, MAXLEN:MAXLEN + input_shape_method], X_test.iloc[:, MAXLEN + input_shape_method:]], y_test, verbose=0)
print(f"Test Accuracy: {acc:.4f}")

# 예측값을 이진 클래스 (0: 정상, 1: 비정상)로 변환
y_pred = model.predict([X_test.iloc[:, :MAXLEN],
                        X_test.iloc[:, MAXLEN:MAXLEN + input_shape_method],
                        X_test.iloc[:, MAXLEN + input_shape_method:]
])

# ROC 기반 최적 threshold 탐색
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
j_scores = tpr - fpr
best_threshold = thresholds[np.argmax(j_scores)]

print(f"✅ Best Threshold based on ROC: {best_threshold:.4f}")

# 새로운 threshold 적용
y_pred_class = (y_pred > best_threshold).astype(int)

# 혼동 행렬
conf_matrix = confusion_matrix(y_test, y_pred_class)

# 혼동 행렬 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Abnormal'], yticklabels=['Normal', 'Abnormal'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# 🔍 정량적 성능 평가 지표 추가
TP = conf_matrix[1, 1]
TN = conf_matrix[0, 0]
FP = conf_matrix[0, 1]
FN = conf_matrix[1, 0]

precision = TP / (TP + FP + 1e-6)
recall = TP / (TP + FN + 1e-6)
f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

print(f"\n📊 Classification Metrics")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

# 모델 예측 실행
def predict_and_display(new_data):
    # URL 처리 (Tokenizer 사용)
    url_sequences = tokenizer.texts_to_sequences([entry['URL'] for entry in new_data])
    url_padded_new = pad_sequences(url_sequences, maxlen=MAXLEN, padding='post')

    # One-hot encoding (훈련 데이터에 대해 fit하고 예측 데이터에 대해 transform)
    method_encoded_new = method_encoder.transform(np.array([entry['Method'] for entry in new_data]).reshape(-1, 1))
    user_agent_encoded_new = user_agent_encoder.transform(np.array([entry['User-Agent'] for entry in new_data]).reshape(-1, 1))

    # 데이터를 모델에 넣을 형태로 변환
    X_new = [url_padded_new, method_encoded_new, user_agent_encoded_new]

    # 예측 수행
    predictions = model.predict(X_new)

    # ROC 기반 임계값 적용한 결과 출력
    for i, prediction in enumerate(predictions):
        label = 'Abnormal' if prediction > best_threshold else 'Normal'
        print(f"Sample {i + 1}: {new_data[i]['URL']}")
        print(f"Prediction: {label}")
        print(f"Prediction probability: {prediction[0]:.4f}\n")

# 기본적인 정상 및 비정상 로그 데이터 테스트
abnormal_data = [
    {
        'URL': 'http://localhost:8080/login?username=admin&password=1234',
        'Method': 'GET',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
    },
    {
        'URL': 'http://localhost:8080/admin?action=delete_all',
        'Method': 'POST',
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36'
    },
    {
        'URL': 'http://localhost:8080/delete.php?id=1',
        'Method': 'POST',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
    },
 {
        'URL': 'http://localhost:8080/search?query=<script>alert(1)</script>',
        'Method': 'GET',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
    },
    {
        'URL': 'http://localhost:8080/login.php?username=admin%27 OR 1=1--',
        'Method': 'POST',
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36'
    },
    {
        'URL': 'http://localhost:8080/admin?delete=true&user_id=1; DROP TABLE users;',
        'Method': 'POST',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
    },
    {
        'URL': 'http://localhost:8080/admin?id=1',
        'Method': 'POST',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
    }
]

normal_data = [
    {
        'URL': 'http://localhost:8080/products?id=1234',
        'Method': 'GET',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'
    },
    {
        'URL': 'http://localhost:8080/home',
        'Method': 'GET',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
    },
    {
        'URL': 'http://localhost:8080/products?id=123',
        'Method': 'GET',
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36'
    },
    {
        'URL': 'http://localhost:8080/about',
        'Method': 'GET',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'
    },
    {
        'URL': 'http://localhost:8080/product?id=1234',
        'Method': 'GET',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
    },
    {
        'URL': 'http://localhost:8080/product?id=1245',
        'Method': 'GET',
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36'
    },
    {
        'URL': 'http://localhost:8080/home',
        'Method': 'GET',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/113.0.0.0 Safari/537.36'
    },
    {
        'URL': 'http://localhost:8080/products/list?page=1',
        'Method': 'GET',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Safari/605.1.15'
    },
    {
        'URL': 'http://localhost:8080/login',
        'Method': 'POST',
        'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64) Gecko/20100101 Firefox/89.0'
    },
    {
        'URL': 'http://localhost:8080/register',
        'Method': 'POST',
        'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_4 like Mac OS X) AppleWebKit/605.1.15'
    },
    {
        'URL': 'http://localhost:8080/contact',
        'Method': 'GET',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/90.0.4430.93 Safari/537.36'
    },
    {
        'URL': 'http://localhost:8080/api/user/info',
        'Method': 'GET',
        'User-Agent': 'PostmanRuntime/7.28.4'
    },
    {
        'URL': 'http://localhost:8080/search?q=temperature+sensor',
        'Method': 'GET',
        'User-Agent': 'Mozilla/5.0 (Linux; Android 11; SM-G981N) AppleWebKit/537.36 Chrome/99.0.4844.84 Mobile Safari/537.36'
    },
    {
        'URL': 'http://localhost:8080/cart/view',
        'Method': 'GET',
        'User-Agent': 'Mozilla/5.0 (iPad; CPU OS 13_6_1 like Mac OS X) AppleWebKit/605.1.15'
    },
    {
        'URL': 'http://localhost:8080/checkout',
        'Method': 'POST',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/92.0.4515.107 Safari/537.36'
    },
    {
        'URL': 'http://localhost:8080/profile/edit',
        'Method': 'POST',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) Chrome/91.0.4472.114 Safari/537.36'
    }
]
# 예측 실행
print("\n🔴 [Abnormal Requests] 예측 결과")
predict_and_display(abnormal_data)

print("\n🟢 [Normal Requests] 예측 결과")
predict_and_display(normal_data)

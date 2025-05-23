import sqlite3
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout, Concatenate, Input, Bidirectional

# 데이터베이스 로드 함수
def load_data_from_db(normal_db_path='normal_data.db', abnormal_db_path='attack_data.db'):
    conn_normal = sqlite3.connect(normal_db_path)
    df_normal = pd.read_sql_query('SELECT * FROM normal_data', conn_normal)
    conn_normal.close()

    conn_abnormal = sqlite3.connect(abnormal_db_path)
    df_abnormal = pd.read_sql_query('SELECT * FROM abnormal_data', conn_abnormal)
    conn_abnormal.close()

    return pd.concat([df_normal, df_abnormal], ignore_index=True)

# DB 경로 설정 및 데이터 로드
normal_db_path = 'D:/Pay1oad 사업/정상/normal_data.db'
abnormal_db_path = 'D:/Pay1oad 사업/비정상/attack_data.db'
data = load_data_from_db(normal_db_path, abnormal_db_path)

# 인코딩 및 전처리
method_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
user_agent_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
method_encoded = method_encoder.fit_transform(data[['Method']])
user_agent_encoded = user_agent_encoder.fit_transform(data[['User-Agent']])

MAXLEN = 25
VOCAB_SIZE = 5000
tokenizer = Tokenizer(num_words=VOCAB_SIZE, filters='', oov_token='<OOV>')
tokenizer.fit_on_texts(data['URL'])
url_sequences = tokenizer.texts_to_sequences(data['URL'])
url_padded = pad_sequences(url_sequences, maxlen=MAXLEN, padding='post')

X = pd.concat([pd.DataFrame(url_padded), pd.DataFrame(method_encoded), pd.DataFrame(user_agent_encoded)], axis=1)
y = data['classification'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# GRU 모델 정의
def build_gru_classifier(vocab_size, maxlen, input_shape_method, input_shape_user_agent):
    input_url = Input(shape=(maxlen,))
    embedding = Embedding(input_dim=vocab_size, output_dim=128, input_length=maxlen)(input_url)
    gru = Bidirectional(GRU(64, dropout=0.3))(embedding)

    input_method = Input(shape=(input_shape_method,))
    dense_method = Dense(32, activation='relu')(input_method)
    dense_method = Dropout(0.3)(dense_method)

    input_user_agent = Input(shape=(input_shape_user_agent,))
    dense_user_agent = Dense(32, activation='relu')(input_user_agent)
    dense_user_agent = Dropout(0.3)(dense_user_agent)

    concatenated = Concatenate()([gru, dense_method, dense_user_agent])
    dense1 = Dense(32, activation='relu')(concatenated)
    output = Dense(1, activation='sigmoid')(dense1)

    model = Model(inputs=[input_url, input_method, input_user_agent], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 학습
input_shape_method = method_encoded.shape[1]
input_shape_user_agent = user_agent_encoded.shape[1]
model = build_gru_classifier(VOCAB_SIZE, MAXLEN, input_shape_method, input_shape_user_agent)
model.fit(
    [X_train.iloc[:, :MAXLEN],
     X_train.iloc[:, MAXLEN:MAXLEN + input_shape_method],
     X_train.iloc[:, MAXLEN + input_shape_method:]],
    y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1
)

# 평가 및 혼동 행렬
loss, acc = model.evaluate([
    X_test.iloc[:, :MAXLEN],
    X_test.iloc[:, MAXLEN:MAXLEN + input_shape_method],
    X_test.iloc[:, MAXLEN + input_shape_method:]], y_test, verbose=0)
print(f"Test Accuracy: {acc:.4f}")

pred = model.predict([
    X_test.iloc[:, :MAXLEN],
    X_test.iloc[:, MAXLEN:MAXLEN + input_shape_method],
    X_test.iloc[:, MAXLEN + input_shape_method:]])

pred_class = (pred > 0.5).astype(int)
conf_matrix = confusion_matrix(y_test, pred_class)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Abnormal'], yticklabels=['Normal', 'Abnormal'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (GRU Classifier)')
plt.show()

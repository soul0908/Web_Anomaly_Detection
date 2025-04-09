import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt

MODEL_PATH = "session_rf_model.pkl"
VEC_PATH = "session_tfidf_vectorizer.pkl"

# ✅ 모델 성능 평가 함수
def evaluate_model_performance(y_true, y_pred, y_proba=None):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_proba) if y_proba is not None else 0

    print("\n📊 모델 성능 평가 지표")
    print(f"✅ Accuracy  : {accuracy:.4f}")
    print(f"🎯 Precision : {precision:.4f}")
    print(f"🔍 Recall    : {recall:.4f}")
    print(f"⚖️  F1-score  : {f1:.4f}")
    print(f"🚦 ROC-AUC   : {roc_auc:.4f}\n")

    return f1 + recall

# ✅ 1. 학습 함수 (정상 + 공격 세션 포함)
def train_model(data_path):
    print("🔧 모델 학습 시작")

    df = pd.read_csv(data_path)
    df["text"] = df["Method"].fillna('') + " " + df["URL"].fillna('') + " " + df["content"].fillna('') + " " + df["User-Agent"].fillna('')
    df["session_id"] = df["cookie"].str.extract(r'JSESSIONID=([^;]+)').fillna("no_session")
    df["session_id"] = (df.index // 20).astype(str)

    session_data = df.groupby("session_id")["text"].apply(lambda x: " ".join(x)).reset_index()
    session_data = session_data.merge(
        df[["session_id", "classification"]].drop_duplicates("session_id"),
        on="session_id"
    )

    normal_sessions = [
        "GET /home HTTP/1.1", "GET /products HTTP/1.1", "GET /cart HTTP/1.1",
        "POST /checkout HTTP/1.1", "GET /confirmation HTTP/1.1", "GET /profile HTTP/1.1",
        "POST /update-profile HTTP/1.1", "GET /feedback HTTP/1.1", "GET /search?q=apple HTTP/1.1",
        "GET /shop/category/electronics HTTP/1.1", "GET /about HTTP/1.1", "GET /contact HTTP/1.1",
        "GET /cart/item?id=123 HTTP/1.1", "GET /checkout/summary HTTP/1.1",
        "GET /confirmation/orderid=987 HTTP/1.1", "GET /terms HTTP/1.1", "GET /faq HTTP/1.1",
        "GET /help HTTP/1.1", "GET /login HTTP/1.1", "GET /logout HTTP/1.1"
    ]
    normal_df = pd.DataFrame({
        "session_id": ["manual_" + str(i) for i in range(len(normal_sessions))],
        "text": normal_sessions,
        "classification": [0] * len(normal_sessions)
    })

    attack_sessions = [
        ["GET /login.jsp?user=admin&pass=admin HTTP/1.1", "GET /search.jsp?q=1' OR '1'='1 HTTP/1.1", "GET /products?id=10 OR 1=1 HTTP/1.1"],
        # ... 추가적인 공격 세션
    ]
    attack_df = pd.DataFrame({
        "session_id": [f"attack_{i}" for i in range(len(attack_sessions))],
        "text": [" ".join(sess if isinstance(sess, list) else [sess]) for sess in attack_sessions],
        "classification": [1] * len(attack_sessions)
    })

    session_data = pd.concat([session_data, normal_df, attack_df], ignore_index=True)
    X = session_data["text"]
    y = session_data["classification"]
    stratify = y if y.nunique() >= 2 and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=stratify, random_state=42)

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3), analyzer="char")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = XGBClassifier(eval_metric='logloss', random_state=42)

    # 하이퍼파라미터 튜닝을 위한 그리드 서치
    param_grid = {
        'max_depth': [3, 5, 7],
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.05, 0.1, 0.15],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'alpha': [0, 0.01, 0.1],
        'lambda': [0, 0.01, 0.1]
    }

    grid_search = GridSearchCV(clf, param_grid, cv=3, n_jobs=-1, verbose=1, scoring='f1')
    grid_search.fit(X_train_vec, y_train)

    print(f"🧪 최적 하이퍼파라미터: {grid_search.best_params_}")

    best_clf = grid_search.best_estimator_
    y_pred = best_clf.predict(X_test_vec)
    y_proba = best_clf.predict_proba(X_test_vec)[:, 1]

    new_score = evaluate_model_performance(y_test, y_pred, y_proba)

    # 모델 저장
    joblib.dump(best_clf, MODEL_PATH)
    joblib.dump(vectorizer, VEC_PATH)

    conf_matrix = confusion_matrix(y_test, y_pred)
    labels = unique_labels(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=labels)
    disp.plot(cmap='Blues', values_format='d')
    plt.title("Confusion Matrix (XGBoost + TF-IDF + GridSearch)")
    plt.show()

# ✅ 2. 검증 함수
def validate_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VEC_PATH):
        print("❌ 저장된 모델이 없습니다. 먼저 학습하세요.")
        return

    clf = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VEC_PATH)

    test_sessions = [
        ["GET /home HTTP/1.1", "GET /products HTTP/1.1", "GET /cart HTTP/1.1", "POST /checkout HTTP/1.1", "GET /products?id=1 OR 1=1 HTTP/1.1"],
        # ... 테스트 세션
    ]

    for i, session in enumerate(test_sessions, 1):
        print(f"\n🧪 테스트 세션 {i}")
        text = " ".join(session)
        vec = vectorizer.transform([text])
        proba = clf.predict_proba(vec)[0]
        if len(proba) == 2 and proba[1] > 0.7:
            print(f"🚨 공격 흐름 탐지! (공격 확률: {proba[1]:.2f})")
        else:
            print(f"✅ 정상 흐름 판단 (공격 확률: {proba[1] if len(proba)==2 else 'N/A'})")

# ✅ 3. 실행 선택 흐름
if __name__ == "__main__":
    print("무엇을 할까요?")
    print("1 - 모델 학습 (정상 + 공격 세션 포함)")
    print("2 - 모델 검증 (5개 테스트 세션 탐지)")
    choice = input("선택 (1 or 2): ").strip()

    if choice == "1":
        train_model("C:/Users/leeja/OneDrive/Desktop/anomaly/.venv/csic_database.csv")
    elif choice == "2":
        validate_model()
    else:
        print("❗ 잘못된 선택입니다.")

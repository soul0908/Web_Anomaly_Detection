import os, sqlite3, random, time, datetime
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ───────── 설정 ─────────
RANDOM_SEED     = 42
TFIDF_MAX_FEAT  = 4000
HIDDEN_UNITS    = [512, 128, 32]
DROPOUT         = 0.3
LR              = 3e-4
BATCH           = 512
EPOCHS          = 25
PATIENCE        = 5
NORMAL_DB, ATTACK_DB = "normal_data.db", "attack_data.db"
NORMAL_TABLE, ATTACK_TABLE = "normal_data", "attack_data"
COL_MAP = {"문장": "sentence", "유형": "ctype"}
# ───────────────────────

def log(msg): print(f"[{datetime.datetime.now():%H:%M:%S}] {msg}")
def set_seed(s): random.seed(s); np.random.seed(s); torch.manual_seed(s)
class LogDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y.values.astype(np.float32)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx):
        dense = torch.tensor(self.X[idx].toarray(), dtype=torch.float32).squeeze(0)
        return dense, torch.tensor([self.y[idx]], dtype=torch.float32)

class DenseNet(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        dims = [in_dim] + HIDDEN_UNITS
        layers = []
        for i in range(len(dims)-1):
            layers += [
                nn.Linear(dims[i], dims[i+1]),
                nn.ReLU(True),
                nn.BatchNorm1d(dims[i+1]),
                nn.Dropout(DROPOUT)
            ]
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

def run_epoch(model, loader, crit, optim=None):
    model.train(optim is not None)
    total, outs, ys = 0.0, [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        if optim: optim.zero_grad()
        out = model(X); loss = crit(out, y)
        if optim: loss.backward(); optim.step()
        total += loss.item() * len(X)
        outs.append(out.detach().cpu()); ys.append(y.detach().cpu())
    return total/len(loader.dataset), torch.cat(outs).numpy(), torch.cat(ys).numpy()

def metrics(y, p, threshold=0.3):
    p_sig = 1 / (1 + np.exp(-p))
    pred  = (p_sig >= threshold).astype(int)
    roc = roc_auc_score(y, p_sig)
    pr  = average_precision_score(y, p_sig)
    cm  = confusion_matrix(y, pred)
    rep = classification_report(y, pred, digits=3, zero_division=0)
    return roc, pr, cm, rep, p_sig

def build_preproc():
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["ctype"]),
        ("txt", TfidfVectorizer(analyzer="char", ngram_range=(3,5), max_features=TFIDF_MAX_FEAT), "sentence")
    ])

def load_sql(db, table):
    with sqlite3.connect(db) as c:
        df = pd.read_sql_query(f"SELECT * FROM {table}", c)
    return df.rename(columns=COL_MAP)[["sentence", "ctype"]]

# ───────── main ────────
if __name__ == "__main__":
    set_seed(RANDOM_SEED)
    base = os.path.dirname(os.path.abspath(__file__))
    normal = load_sql(os.path.join(base, NORMAL_DB), NORMAL_TABLE)
    attack = load_sql(os.path.join(base, ATTACK_DB), ATTACK_TABLE)
    normal["label"] = 0; attack["label"] = 1

    attack_train = attack.sample(frac=0.05, random_state=RANDOM_SEED)
    attack_test  = attack.drop(attack_train.index)
    tr_norm, te_norm = train_test_split(normal, test_size=0.3, random_state=RANDOM_SEED)

    train_df = pd.concat([tr_norm, attack_train], ignore_index=True)
    test_df  = pd.concat([te_norm, attack_test], ignore_index=True)

    y_tr, y_te = train_df.label, test_df.label
    X_tr_raw, X_te_raw = train_df.drop("label", axis=1), test_df.drop("label", axis=1)

    pre = build_preproc()
    X_tr = pre.fit_transform(X_tr_raw)
    X_te = pre.transform(X_te_raw)

    train_ds, test_ds = LogDataset(X_tr, y_tr), LogDataset(X_te, y_te)
    ld_tr = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    ld_te = DataLoader(test_ds, batch_size=BATCH, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = DenseNet(X_tr.shape[1]).to(device)
    pos_weight = torch.tensor((len(y_tr)-y_tr.sum())/y_tr.sum(), dtype=torch.float32).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=2)

    best, wait = 0, 0
    for ep in range(1, EPOCHS+1):
        st = time.time()
        tr_loss, _, _ = run_epoch(model, ld_tr, crit, opt)
        _, p_te, y_te_ = run_epoch(model, ld_te, crit)
        roc, pr, cm, _, _ = metrics(y_te_, p_te)
        log(f"[{ep:02}] loss {tr_loss:.4f} ROC {roc:.3f} PR {pr:.3f} TP {cm[1,1]} FN {cm[1,0]} FP {cm[0,1]} TN {cm[0,0]}")
        sched.step(pr)
        if pr > best: best = pr; wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                log("Early stop"); break

    _, p_te, y_te_ = run_epoch(model, ld_te, crit)
    roc, pr, cm, rep, prob = metrics(y_te_, p_te, threshold=0.3)
    print("\n=== 최종 평가 지표 (threshold=0.3) ===")
    print(f"ROC-AUC : {roc:.4f}\nPR-AUC  : {pr:.4f}")
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", rep)

    print("\n=== 예측 점수 분포 시각화 ===")
    plt.hist(prob[y_te_==1], bins=100, alpha=0.6, label="Anomalous")
    plt.hist(prob[y_te_==0], bins=100, alpha=0.6, label="Normal")
    plt.legend(); plt.title("Model Output Distribution"); plt.show()

    print("\n=== 샘플 로그 예측 ===")
    sample_logs = [
        "GET /tienda1/publico/vulnerable.jsp?id=1'or'1'='1 HTTP/1.1",
        "GET /index.jsp HTTP/1.1"
    ]
    sample_df = pd.DataFrame({"sentence": sample_logs, "ctype": ["SQLi?", "Normal?"]})
    X_sample = pre.transform(sample_df)
    with torch.no_grad():
        pred = torch.sigmoid(model(torch.tensor(X_sample.toarray(), dtype=torch.float32).to(device))).cpu().numpy().ravel()
    for s, p in zip(sample_logs, pred):
        lbl = "Anomalous" if p >= 0.3 else "Normal"
        print(f"{lbl:10}  {p:.3f}  |  {s}")


import numpy as np
import pandas as pd
import pywt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import precision_score, recall_score, f1_score


# =====================================================
# Step 1: Load window-based behavioral matrices
# =====================================================
def load_windowed_data(file, window_size=24, stride=24):
    df = pd.read_csv(file)
    feature_cols = [
        'device_count', 'email_count', 'file_count',
        'http_count', 'logon_count', 'total_behavior_count'
    ]

    df = df.sort_values(["date_only", "hour"]).reset_index(drop=True)
    X_list, y_list = [], []

    for start in range(0, len(df) - window_size + 1, stride):
        window = df.iloc[start:start + window_size]
        if len(window) != window_size:
            continue

        mat = window[feature_cols].T.values   # shape = (6, window_size)
        label = window["label"].max()        # union rule
        X_list.append(mat)
        y_list.append(label)

    X_all = np.array(X_list)
    y_all = np.array(y_list)

    print(f"[INFO] window={window_size}, samples={X_all.shape[0]}, "
          f"shape={X_all.shape[1:]}, anomaly_ratio={y_all.mean():.3f}")

    return X_all, y_all



# =====================================================
# Step 2: DWT decomposition
# =====================================================
def dwt_decompose(img, wavelet="db1", level=1):
    cA, (cH, cV, cD) = pywt.wavedec2(img, wavelet=wavelet, level=level)
    return cA, cH, cV, cD



# =====================================================
# Step 3: Class balancing (upsampling anomalies)
# =====================================================
def balance_data(X, y):
    df = pd.DataFrame({'label': y})
    df['data'] = list(X)

    df_major = df[df.label == 0]
    df_minor = df[df.label == 1]

    df_minor_up = resample(
        df_minor, replace=True,
        n_samples=len(df_major),
        random_state=42
    )

    df_bal = pd.concat([df_major, df_minor_up])
    X_bal = np.stack(df_bal['data'])
    y_bal = df_bal['label'].values

    return X_bal, y_bal



# =====================================================
# Step 4: Feature construction (Ablation support)
# =====================================================
def build_features(X, variant="full"):
    feats = []

    for img in X:

        # --- DWT decomposition ---
        if variant == "w/o DWT":
            feat = img.flatten()
        else:
            cA, cH, cV, cD = dwt_decompose(img)
            feat = np.concatenate([
                cA.flatten(),
                cH.flatten(),
                cV.flatten(),
                cD.flatten()
            ])

        # --- Simple attention (mean + max) ---
        if variant != "w/o Attention":
            feat = np.concatenate([feat, [feat.mean(), feat.max()]])

        # --- Deviation modulation ---
        if variant != "w/o Deviation Modulation":
            feat = feat / (np.std(feat) + 1e-6)

        feats.append(feat)

    return np.array(feats)



# =====================================================
# Step 5: PyTorch Dataset
# =====================================================
class LogDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]



# =====================================================
# Step 6: CNN Model
# =====================================================
class CNNModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)

        self.fc = nn.Sequential(
            nn.Linear((input_dim // 2) * 32, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.unsqueeze(1)               # (B,1,L)
        x = self.conv1(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.fc(x).squeeze()



# =====================================================
# Step 7: Transformer Model
# =====================================================
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2):
        super().__init__()

        self.embedding = nn.Linear(1, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.fc = nn.Sequential(
            nn.Linear(d_model * input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.unsqueeze(-1)       # (B, L, 1)
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.flatten(1)
        return self.fc(x).squeeze()



# =====================================================
# Step 8: Fusion evaluation (CNN + Transformer)
# =====================================================
def evaluate_models(cnn, trans, loader, alpha=0.5):
    y_true, y_cnn, y_trans, y_fusion = [], [], [], []

    cnn.eval()
    trans.eval()

    with torch.no_grad():
        for xb, yb in loader:
            y_true.extend(yb.tolist())

            p1 = cnn(xb).tolist()
            p2 = trans(xb).tolist()

            if not isinstance(p1, list): p1 = [p1]
            if not isinstance(p2, list): p2 = [p2]

            y_cnn.extend(p1)
            y_trans.extend(p2)
            y_fusion.extend([(alpha * a + (1 - alpha) * b) for a, b in zip(p1, p2)])

    def metrics(y_true, y_pred):
        y_bin = [1 if p >= 0.5 else 0 for p in y_pred]
        pre = precision_score(y_true, y_bin, zero_division=0)
        rec = recall_score(y_true, y_bin, zero_division=0)
        f1 = f1_score(y_true, y_bin, zero_division=0)
        return {"Precision": round(pre, 3), "Recall": round(rec, 3), "F1": round(f1, 3)}

    return {
        "CNN": metrics(y_true, y_cnn),
        "Transformer": metrics(y_true, y_trans),
        "Fusion": metrics(y_true, y_fusion)
    }



# =====================================================
# Step 9: Run Ablation
# =====================================================
def run_ablation(file, window_size=24, stride=24, epochs=10, batch_size=16):
    X_all, y_all = load_windowed_data(file, window_size, stride)
    X_bal, y_bal = balance_data(X_all, y_all)

    results = {}

    for variant in ["full"]:
        print(f"\n[RUNNING] Variant = {variant}")

        X_feat = build_features(X_bal, variant)

        X_train, X_test, y_train, y_test = train_test_split(
            X_feat, y_bal, test_size=0.2, stratify=y_bal, random_state=42
        )

        train_loader = DataLoader(LogDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(LogDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

        input_dim = X_train.shape[1]

        cnn = CNNModel(input_dim)
        trans = TransformerModel(input_dim)

        criterion = nn.BCELoss()
        opt1 = torch.optim.Adam(cnn.parameters(), lr=1e-3)
        opt2 = torch.optim.Adam(trans.parameters(), lr=1e-3)

        # ------------------- Train CNN and Transformer -------------------
        for epoch in range(epochs):
            cnn.train()
            trans.train()

            for xb, yb in train_loader:

                # ----- CNN -----
                opt1.zero_grad()
                loss1 = criterion(cnn(xb), yb)
                loss1.backward()
                opt1.step()

import numpy as np
import pandas as pd
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import precision_score, recall_score, f1_score



# ===================================================
# Step 1: Load window-based samples (support various window sizes)
# ===================================================
def load_windowed_data(file, window_size=24, stride=24):
    df = pd.read_csv(file)
    feature_cols = [
        'device_count', 'email_count', 'file_count',
        'http_count', 'logon_count', 'total_behavior_count'
    ]

    df = df.sort_values(["date_only", "hour"]).reset_index(drop=True)
    df["time_idx"] = np.arange(len(df))

    X_list, y_list = [], []
    for start in range(0, len(df) - window_size + 1, stride):
        win = df.iloc[start:start+window_size]
        if len(win) != window_size:
            continue

        mat = win[feature_cols].T.values       # shape = (6, window_size)
        label = win["label"].max()            # union label rule
        X_list.append(mat)
        y_list.append(label)

    X_all = np.array(X_list)
    y_all = np.array(y_list)

    print(f"[INFO] window={window_size}, samples={X_all.shape[0]}, "
          f"shape={X_all.shape[1:]}, anomaly_ratio={y_all.mean():.3f}")
    return X_all, y_all



# ===================================================
# Step 2: DWT decomposition
# ===================================================
def dwt_decompose(img, wavelet="db1", level=1):
    coeffs2 = pywt.wavedec2(img, wavelet=wavelet, level=level)
    cA, (cH, cV, cD) = coeffs2
    return cA, cH, cV, cD



# ===================================================
# Step 3: Class balancing
# ===================================================
def balance_data(X, y):
    df_all = pd.DataFrame({"label": y})
    df_all["data"] = list(X)

    df_major = df_all[df_all.label == 0]
    df_minor = df_all[df_all.label == 1]

    df_minor_up = resample(
        df_minor, replace=True,
        n_samples=len(df_major), random_state=42
    )

    df_bal = pd.concat([df_major, df_minor_up])
    X_bal = np.stack(df_bal["data"])
    y_bal = df_bal["label"].values

    return X_bal, y_bal



# ===================================================
# Step 4: Feature construction (Ablation modes)
# ===================================================
def build_features(X, variant="full"):
    feats = []

    for img in X:
        # 1) DWT decomposition
        if variant == "w/o DWT":
            feat = img.flatten()
        else:
            cA, cH, cV, cD = dwt_decompose(img)
            feat = np.concatenate([
                cA.flatten(), cH.flatten(),
                cV.flatten(), cD.flatten()
            ])

        # 2) Attention (simple mean/max)
        if variant != "w/o Attention":
            feat = np.concatenate([feat, [feat.mean(), feat.max()]])

        # 3) Deviation modulation
        if variant != "w/o Deviation Modulation":
            feat = feat / (np.std(feat) + 1e-6)

        feats.append(feat)

    return np.array(feats)



# ===================================================
# Step 5: Capsule Network (1D)
# ===================================================
class PrimaryCaps1D(nn.Module):
    def __init__(self, num_capsules=8, in_channels=32,
                 out_dim=8, kernel_size=9, stride=2):
        super().__init__()
        self.capsules = nn.Conv1d(
            in_channels, num_capsules * out_dim,
            kernel_size=kernel_size, stride=stride
        )
        self.num_capsules = num_capsules
        self.out_dim = out_dim

    def forward(self, x):
        B = x.size(0)
        u = self.capsules(x)
        u = u.view(B, self.num_capsules, self.out_dim, -1)
        u = u.permute(0, 1, 3, 2).contiguous()
        u = u.view(B, -1, self.out_dim)
        return self.squash(u)

    def squash(self, s):
        norm = (s ** 2).sum(dim=2, keepdim=True)
        scale = norm / (1 + norm) / torch.sqrt(norm + 1e-9)
        return scale * s



class DigitCaps1D(nn.Module):
    def __init__(self, num_caps_in, dim_caps_in,
                 num_caps_out=2, dim_caps_out=16, num_routing=3):
        super().__init__()
        self.num_caps_out = num_caps_out
        self.dim_caps_out = dim_caps_out
        self.num_routing = num_routing

        self.W = nn.Parameter(0.01 * torch.randn(
            1, num_caps_in, num_caps_out,
            dim_caps_out, dim_caps_in
        ))

    def forward(self, x):
        B, N, D = x.shape
        x = x.unsqueeze(2).unsqueeze(4)
        u_hat = torch.matmul(self.W, x).squeeze(4)

        b_ij = torch.zeros(1, N, self.num_caps_out, 1).to(x.device)

        for r in range(self.num_routing):
            c_ij = F.softmax(b_ij, dim=2)
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = self.squash(s_j)
            if r < self.num_routing - 1:
                b_ij = b_ij + (u_hat * v_j).sum(dim=-1, keepdim=True)

        return v_j.squeeze(1)

    def squash(self, s):
        norm = (s ** 2).sum(dim=2, keepdim=True)
        scale = norm / (1 + norm) / torch.sqrt(norm + 1e-9)
        return scale * s



class CapsNet1D(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=9, stride=1)
        self.primary_caps = PrimaryCaps1D()

        # Determine capsule shape dynamically
        dummy = torch.zeros(1, 1, input_dim)
        out = F.relu(self.conv1(dummy))
        out = self.primary_caps(out)
        num_caps_in = out.size(1)
        dim_caps_in = out.size(2)

        self.digit_caps = DigitCaps1D(
            num_caps_in=num_caps_in,
            dim_caps_in=dim_caps_in,
            num_caps_out=num_classes,
            dim_caps_out=16
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = self.primary_caps(x)
        x = self.digit_caps(x)
        out = torch.norm(x, dim=2)
        return out



# ===================================================
# Step 6: Capsule model training & evaluation
# ===================================================
def run_variant(X, y, variant):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X_feat = build_features(X, variant)

    X_train, X_test, y_train, y_test = train_test_split(
        X_feat, y, test_size=0.2, stratify=y, random_state=42
    )

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                             torch.tensor(y_train, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                            torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1)

    model = CapsNet1D(X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # training
    model.train()
    for epoch in range(5):   # small demo
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # evaluation
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            out = model(xb)
            pred = out.argmax(dim=1).item()
            y_true.append(yb.item())
            y_pred.append(pred)

    pre = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {"Precision": round(pre, 3), "Recall": round(rec, 3), "F1": round(f1, 3)}



# ===================================================
# Step 7: Run all ablation modes
# ===================================================
def run_ablation(file, window_size=24, stride=24):
    X_all, y_all = load_windowed_data(file, window_size, stride)
    X_bal, y_bal = balance_data(X_all, y_all)

    results = {}
    variants = ["full", "w/o Deviation Modulation", "w/o DWT", "w/o Attention"]

    for v in variants:
        print(f"\n[Running Variant] {v}")
        results[v] = run_variant(X_bal, y_bal, v)

    print(f"\n===== Ablation Results (window={window_size}h) =====")
    for k, v in results.items():
        print(k, v)
    return results



# ===================================================
# Main
# ===================================================
if __name__ == "__main__":

    raw_file = r"D:\your_path\scenario_2_AAF0535_hourly.csv"

    run_ablation(raw_file, window_size=24, stride=24)
    run_ablation(raw_file, window_size=72, stride=24)
    run_ablation(raw_file, window_size=168, stride=24)

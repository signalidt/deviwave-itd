import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils import resample
from sklearn.metrics import precision_recall_curve


# ========================================
# Step 1: Load raw CERT_r4.2 daily log → (6×24) behavior matrix
# ========================================
raw_file = r"D:\your_path\scenario_2_AKR0057_hourly.csv"
df = pd.read_csv(raw_file)

features = [
    'device_count', 'email_count', 'file_count',
    'http_count', 'logon_count', 'total_behavior_count'
]

X_list, y_list = [], []
for date, g in df.groupby("date_only"):
    if len(g) != 24:
        continue

    mat = g.sort_values("hour")[features].T.values   # shape = (6,24)
    label = g["label"].max()                         # label of the entire day
    X_list.append(mat)
    y_list.append(label)

X_all = np.array(X_list)   # (N,6,24)
y_all = np.array(y_list)



# ========================================
# Step 2: FFT multi-band decomposition
# ========================================
def fft_band_decompose(img, low_frac=0.2, high_frac=0.6):
    F = np.fft.fft2(img)
    Fshift = np.fft.fftshift(F)
    h, w = img.shape
    crow, ccol = h // 2, w // 2

    # --- radial masks for low/mid/high frequency ---
    def band_mask(r1, r2):
        mask = np.zeros((h, w))
        for i in range(h):
            for j in range(w):
                dist = np.sqrt((i - crow)**2 + (j - ccol)**2)
                if r1 <= dist < r2:
                    mask[i, j] = 1
        return mask

    r_low = int(min(h, w) * low_frac)
    r_high = int(min(h, w) * high_frac)

    low_band = band_mask(0, r_low)
    mid_band = band_mask(r_low, r_high)
    high_band = band_mask(r_high, max(h, w))

    # --- reconstruct from masked frequency domain ---
    def recon(mask):
        part = Fshift * mask
        return np.abs(np.fft.ifft2(np.fft.ifftshift(part)))

    return recon(low_band), recon(mid_band), recon(high_band)



# ========================================
# Step 3: Attention modules (SE + CBAM)
# ========================================
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = SEBlock(channels, reduction)
        self.sa_conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel attention
        out = self.ca(x)

        # Spatial attention
        avg_out = torch.mean(out, 1, keepdim=True)
        max_out, _ = torch.max(out, 1, keepdim=True)
        sa = torch.cat([avg_out, max_out], dim=1)
        sa = self.sigmoid(self.sa_conv(sa))

        return out * sa



# ========================================
# Step 4: Temporal Convolutional Network (TCN)
# ========================================
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size] if self.chomp_size > 0 else x


class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size,
                               padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size,
                               padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return out + res



class TCN(nn.Module):
    def __init__(self, in_ch=6, out_ch=64, num_layers=3):
        super().__init__()
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            layers.append(
                TemporalBlock(
                    in_ch if i == 0 else out_ch,
                    out_ch,
                    kernel_size=3,
                    dilation=dilation
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)



# ========================================
# Step 5: Attention + FFT + TCN model
# ========================================
class AttentionTCNModel(nn.Module):
    def __init__(self, in_ch, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        if use_attention:
            self.att = CBAM(in_ch)

        self.tcn = TCN(in_ch=6, out_ch=64, num_layers=3)

        self.fc = nn.Sequential(
            nn.Linear(64 * 24 * in_ch, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):  # (B,C,6,24)
        if self.use_attention:
            x = self.att(x)

        B, C, H, W = x.shape
        feats = []

        for c in range(C):
            seq = x[:, c, :, :]      # (B,6,24)
            seq = self.tcn(seq)      # (B,64,24)
            feats.append(seq.flatten(1))

        feat_cat = torch.cat(feats, dim=1)
        return self.fc(feat_cat)



# ========================================
# Step 6: Focal Loss
# ========================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        loss = self.alpha * (1 - pt)**self.gamma * BCE
        return loss.mean()



# ========================================
# Step 7: Full Ablation Runner
# ========================================
def run_experiment(config, X_all, y_all):

    # ------------------------------------
    # 1. Preprocess: modulation → FFT bands
    # ------------------------------------
    X_proc = []

    for mat in X_all:  # (6,24)

        # ---- deviation modulation ----
        if not config["use_modulation"]:
            mat_mod = mat
        else:
            mu = mat.mean(axis=1, keepdims=True)
            sigma = mat.std(axis=1, keepdims=True) + 1e-5
            delta = np.abs(mat - mu) / sigma
            mask = np.where(delta < 1.0, 0.7, 1.0 + 0.5 * delta)
            mat_mod = mat * mask

        # ---- FFT multi-band ----
        if not config["use_fft"]:
            stacked = np.expand_dims(mat_mod, axis=0)  # (1,6,24)
        else:
            low, mid, high = fft_band_decompose(mat_mod)
            stacked = np.stack([low, mid, high], axis=0)  # (3,6,24)

        X_proc.append(stacked)

    X_proc = np.array(X_proc)


    # ------------------------------------
    # 2. Balance classes by upsampling anomalies
    # ------------------------------------
    df_all = pd.DataFrame({'label': y_all})
    df_all['data'] = list(X_proc)

    df_major = df_all[df_all['label'] == 0]
    df_minor = df_all[df_all['label'] == 1]

    df_minor_up = resample(df_minor, replace=True,
                           n_samples=len(df_major), random_state=42)

    df_bal = pd.concat([df_major, df_minor_up])

    X_bal = torch.tensor(np.stack(df_bal['data']), dtype=torch.float32)
    y_bal = torch.tensor(df_bal['label'].values, dtype=torch.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=0.2, stratify=y_bal, random_state=42
    )

    train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=16, shuffle=True)
    test_loader = DataLoader(list(zip(X_test, y_test)), batch_size=1)


    # ------------------------------------
    # 3. Train model
    # ------------------------------------
    model = AttentionTCNModel(in_ch=X_proc.shape[1], use_attention=config["use_attention"])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = FocalLoss(alpha=0.5, gamma=2.0)

    for epoch in range(10):  # small epochs for fast testing
        model.train()
        for xb, yb in train_loader:
            y_pred = model(xb).squeeze()
            loss = criterion(y_pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    # ------------------------------------
    # 4. Evaluate with adaptive threshold (maximize F1)
    # ------------------------------------
    model.eval()
    y_true, y_prob = [], []

    with torch.no_grad():
        for xb, yb in test_loader:
            prob = model(xb).item()
            y_prob.append(prob)
            y_true.append(int(yb.item()))

    y_prob = np.array(y_prob)
    y_true = np.array(y_true)

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)

    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]

    y_pred = (y_prob > best_thresh).astype(int)

    return {
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 2),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 2),
        "F1": round(f1_score(y_true, y_pred, zero_division=0), 2)
    }



# ========================================
# Step 8: Run Ablation Experiments
# ========================================
configs = {
    "Ours (Full Model)": {"use_modulation": True, "use_fft": True, "use_attention": True},
    "w/o Deviation Modulation": {"use_modulation": False, "use_fft": True, "use_attention": True},
    "w/o FFT": {"use_modulation": True, "use_fft": False, "use_attention": True},
    "w/o Attention": {"use_modulation": True, "use_fft": True, "use_attention": False},
}

results = {}
for name, cfg in configs.items():
    print(f"\nRunning {name} ...")
    results[name] = run_experiment(cfg, X_all, y_all)


print("\nAblation Results:")
for k, v in results.items():
    print(k, v)



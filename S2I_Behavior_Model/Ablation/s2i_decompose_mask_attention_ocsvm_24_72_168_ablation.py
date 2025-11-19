import numpy as np
import pandas as pd
import pywt
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.svm import OneClassSVM


# =====================================================
# Step 1: Load window-based behavioral matrices
# =====================================================
def load_windowed_data(file, window_size=24, stride=24):
    """
    Convert CERT_r4.2 hourly logs to a (6 × window_size) behavior matrix.
    """
    df = pd.read_csv(file)
    feature_cols = [
        'device_count', 'email_count', 'file_count',
        'http_count', 'logon_count', 'total_behavior_count'
    ]

    df = df.sort_values(["date_only", "hour"]).reset_index(drop=True)
    df["time_idx"] = np.arange(len(df))

    X_list, y_list = [], []
    for start in range(0, len(df) - window_size + 1, stride):
        window = df.iloc[start:start+window_size]
        if len(window) != window_size:
            continue

        mat = window[feature_cols].T.values     # shape = (6, window_size)
        label = window["label"].max()          # union rule (1 if any anomaly within window)
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
    df = pd.DataFrame({"label": y})
    df["data"] = list(X)

    df_major = df[df.label == 0]
    df_minor = df[df.label == 1]

    df_minor_up = resample(
        df_minor, replace=True,
        n_samples=len(df_major),
        random_state=42
    )

    df_bal = pd.concat([df_major, df_minor_up])
    X_bal = np.stack(df_bal["data"])
    y_bal = df_bal["label"].values

    return X_bal, y_bal



# =====================================================
# Step 4: Feature construction (Ablation support)
# =====================================================
def build_features(X, variant="full"):
    feats = []

    for img in X:

        # ---- DWT (remove in ablation) ----
        if variant == "w/o DWT":
            feat = img.flatten()
        else:
            cA, cH, cV, cD = dwt_decompose(img)
            feat = np.concatenate([
                cA.flatten(), cH.flatten(),
                cV.flatten(), cD.flatten()
            ])

        # ---- Simple attention (mean/max) ----
        if variant != "w/o Attention":
            feat = np.concatenate([feat, [feat.mean(), feat.max()]])

        # ---- Deviation modulation ----
        if variant != "w/o Deviation Modulation":
            feat = feat / (np.std(feat) + 1e-6)

        feats.append(feat)

    return np.array(feats)



# =====================================================
# Step 5: Train and evaluate OCSVM
# =====================================================
def run_variant(X, y, variant):

    X_feat = build_features(X, variant)

    X_train, X_test, y_train, y_test = train_test_split(
        X_feat, y, test_size=0.2, stratify=y, random_state=42
    )

    # OCSVM trains only on normal samples
    X_train_norm = X_train[y_train == 0]

    clf = OneClassSVM(kernel='rbf', nu=0.1, gamma='scale')
    clf.fit(X_train_norm)

    y_pred = clf.predict(X_test)
    y_pred = np.where(y_pred == -1, 1, 0)   # convert: -1 → anomaly, 1 → normal

    pre = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    return {
        "Precision": round(pre, 2),
        "Recall": round(rec, 2),
        "F1": round(f1, 2)
    }



# =====================================================
# Step 6: Run ablation experiments
# =====================================================
def run_ablation(file, window_size=24, stride=24):
    X_all, y_all = load_windowed_data(file, window_size, stride)
    X_bal, y_bal = balance_data(X_all, y_all)

    variants = ["full", "w/o Deviation Modulation", "w/o DWT", "w/o Attention"]

    results = {}
    for v in variants:
        print(f"\n[RUNNING] Variant = {v}")
        results[v] = run_variant(X_bal, y_bal, v)

    print(f"\nAblation Results (window = {window_size}h):")
    for k, v in results.items():
        print(k, v)

    return results



# =====================================================
# Main Program
# =====================================================
if __name__ == "__main__":

    raw_file = r"D:\your_path\scenario_1_EHD0584_hourly.csv"

    run_ablation(raw_file, window_size=24, stride=24)
    run_ablation(raw_file, window_size=72, stride=24)
    run_ablation(raw_file, window_size=168, stride=24)

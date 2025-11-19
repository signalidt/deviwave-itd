import numpy as np
import pandas as pd
import pywt
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import precision_score, recall_score, f1_score
from xgboost import XGBClassifier


# ======================
# Step 1: Load windowed CERT logs
# ======================
def load_windowed_data(file, window_size=24, stride=24):
    """
    Convert CERT_r4.2 hourly logs into (6 × window_size) behavior matrices.

    :param file: Path to the input CSV
    :param window_size: Time window length (24=1 day, 72=3 days, 168=1 week)
    :param stride: Sliding step
    :return: (X_all, y_all)
    """

    df = pd.read_csv(file)
    features = [
        'device_count', 'email_count', 'file_count',
        'http_count', 'logon_count', 'total_behavior_count'
    ]

    # Ensure correct time order
    df = df.sort_values(["date_only", "hour"]).reset_index(drop=True)

    # Add continuous time index
    df["time_idx"] = np.arange(len(df))

    X_list, y_list = [], []

    for start in range(0, len(df) - window_size + 1, stride):
        window = df.iloc[start:start + window_size]

        if len(window) != window_size:
            continue

        mat = window[features].T.values  # (6, window_size)
        label = window["label"].max()   # union rule
        X_list.append(mat)
        y_list.append(label)

    X_all = np.array(X_list)
    y_all = np.array(y_list)

    print(f"[INFO] Window={window_size}, Samples={X_all.shape[0]}, InputShape={X_all.shape[1:]}, PosRatio={y_all.mean():.3f}")
    return X_all, y_all


# ======================
# Step 2: DWT decomposition
# ======================
def dwt_decompose(img, wavelet="db1", level=1):
    """
    Apply 2D DWT to matrix input.
    Returns (cA, cH, cV, cD).
    """
    coeffs2 = pywt.wavedec2(img, wavelet=wavelet, level=level)
    cA, (cH, cV, cD) = coeffs2
    return cA, cH, cV, cD


# ======================
# Step 3: Balance classes via upsampling
# ======================
def balance_data(X, y):
    df_all = pd.DataFrame({'label': y})
    df_all['data'] = list(X)

    df_major = df_all[df_all['label'] == 0]
    df_minor = df_all[df_all['label'] == 1]

    df_minor_upsampled = resample(
        df_minor, replace=True,
        n_samples=len(df_major), random_state=42
    )

    df_balanced = pd.concat([df_major, df_minor_upsampled])
    X_bal = np.stack(df_balanced['data'])
    y_bal = df_balanced['label'].values

    return X_bal, y_bal


# ======================
# Step 4: Build ablation features
# ======================
def build_features(X, variant="full"):
    feats = []

    for img in X:

        # Step 1: DWT
        if variant == "w/o DWT":
            feat = img.flatten()
        else:
            cA, cH, cV, cD = dwt_decompose(img)
            feat = np.concatenate([
                cA.flatten(), cH.flatten(),
                cV.flatten(), cD.flatten()
            ])

        # Step 2: Simplified Attention (mean + max)
        if variant != "w/o Attention":
            feat = np.concatenate([feat, [feat.mean(), feat.max()]])

        # Step 3: Deviation modulation
        if variant != "w/o Deviation Modulation":
            feat = feat / (np.std(feat) + 1e-6)

        feats.append(feat)

    return np.array(feats)


# ======================
# Step 5: Train & Evaluate with XGBoost
# ======================
def run_variant(X, y, variant):
    X_feat = build_features(X, variant)

    X_train, X_test, y_train, y_test = train_test_split(
        X_feat, y, test_size=0.2,
        stratify=y, random_state=42
    )

    clf = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss"
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    pre = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    return {
        "Precision": round(pre, 2),
        "Recall": round(rec, 2),
        "F1": round(f1, 2)
    }


# ======================
# Step 6: Run ablation study
# ======================
def run_ablation(file, window_size=24, stride=24):
    X_all, y_all = load_windowed_data(file, window_size, stride)
    X_bal, y_bal = balance_data(X_all, y_all)

    results = {}
    variants = ["full", "w/o Deviation Modulation", "w/o DWT", "w/o Attention"]

    for variant in variants:
        results[variant] = run_variant(X_bal, y_bal, variant)

    print(f"\nAblation Results (Window={window_size}h):")
    for k, v in results.items():
        print(k, v)

    return results


# ======================
# Main
# ======================
if __name__ == "__main__":
    raw_file = r"D:\your_path\scenario_3_JGT0221_hourly.csv"

    # Run 24h window
    run_ablation(raw_file, window_size=24, stride=24)

    # Run 72h window
    run_ablation(raw_file, window_size=72, stride=24)

    # Run 168h window
    run_ablation(raw_file, window_size=168, stride=24)



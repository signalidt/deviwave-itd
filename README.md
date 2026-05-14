# DeviWave-ITD

<p align="center">
  <a href="./README.md">English</a> |
  <a href="./README_CN.md">简体中文</a>
</p>

# DeviWave-ITD: Multi-Resolution Insider Threat Detection with Behavioral Matrices & Wavelet Decomposition

## 📘 Overview

**DeviWave-ITD** is an end-to-end insider threat detection framework that transforms raw user activity logs into multi-resolution behavioral signals.  
It integrates:

- **Behavioral matrix construction**
- **Deviation-aware modulation**
- **DWT/FFT frequency decomposition**
- **Attention-enhanced sequence modeling**
- **Unified ablation testing across multiple models**

The complete pipeline converts raw CERT logs into structured multi-day behavior sequences (24h/72h/168h) and feeds them into deep and traditional models for anomaly detection.

---

# 📂 Project Structure

```
Project/
│── imgs/
│
│── Preprocess/
│   ├── output/
│   │   ├── log_merged/
│   │   ├── log_merged_24_hours/
│   │   └── log_split/
│   │       ├── http_domains.csv
│   │       ├── http_domains_with_category.csv
│   │       ├── ldap_with_device.csv
│   │       ├── ldap_with_device_department.csv
│   │       └── ...
│   ├── config.yaml
│   ├── department_relationship_extract.py
│   ├── device_extract.py
│   ├── domain_categories.py
│   ├── domain_extract.py
│   ├── step1_log_split.py
│   ├── step2_log_merge.py
│   ├── step3_log_labeling.py
│   └── step4_hourly_stat.py
│
│── S2I_Behavior_Model/
│   ├── Base_Model/
│   │   └── s2i_decompose_mask_attention_tcn.py
│   ├── Ablation/
│   │   ├── ..._caps_24_72_168_ablation.py
│   │   ├── ..._cate_24_72_168_ablation.py
│   │   ├── ..._ocsvm_24_72_168_ablation.py
│   │   ├── ..._tcn_24_72_168_ablation.py
│   │   └── ..._xgb_24_72_168_ablation.py
│
└── README.md
```

---

# 🧩 1. Data Preprocessing Pipeline

The raw CERT logs are converted into structured hourly behavioral matrices through four stages:

## **Step 1 — Log Splitting**
Splits raw logs into category-specific CSVs:

- HTTP  
- LDAP  
- Email  
- File  
- Device  

## **Step 2 — Log Merging**
Merges all event categories by timestamp for each user.

## **Step 3 — Log Labeling**
Assigns labels according to scenario-based anomaly rules.

## **Step 4 — Hourly Behavioral Statistics**
Produces a **6×24 behavioral matrix** containing hourly counts of:

- device_count  
- email_count  
- file_count  
- http_count  
- logon_count  
- total_behavior_count  

This 6-channel behavioral signal is the fundamental input to all models.

---

# 🔬 2. S2I (Signal-to-Insight) Modeling Framework

The S2I framework converts raw behavioral matrices into multi-modal feature representations.

## **2.1 Frequency Decomposition**

### **Wavelet (DWT) Decomposition**
Generates multi-resolution components:

- Approximation: **cA**  
- Horizontal detail: **cH**  
- Vertical detail: **cV**  
- Diagonal detail: **cD**

### **FFT Band Decomposition**
Produces three frequency sub-bands:

- **Low-frequency**
- **Mid-frequency**
- **High-frequency**

---

## **2.2 Deviation Mask Modulation (DMM)**

Used to highlight abnormal behavioral spikes while suppressing normal noise.

```
delta = |x - μ| / σ

if delta < 1.0:
    mask = 0.7
else:
    mask = 1 + 0.5 * delta

enhanced = x * mask
```

Effects:
- **Suppresses normal behavior**
- **Amplifies deviations**
- **Enhances anomaly separability**

---

## **2.3 Attention Modules**

Two attention mechanisms refine the decomposed signals:

- **SEBlock** — channel-wise recalibration  
- **CBAM** — combined channel & spatial attention  

---

# 🔥 3. Base Model: Attention-TCN

The main deep learning model integrates:

- Multi-resolution decomposition  
- Deviation modulation  
- CBAM attention  
- Dilated TCN layers  
- MLP classifier  

Supports three temporal windows:

- **24 hours**  
- **72 hours**  
- **168 hours**

---

# 🧪 4. Ablation Models

Implemented models for component-level evaluation:

| Model Type | Script |
|------------|--------|
| Capsule Network | s2i_decompose_mask_attention_caps_24_72_168_ablation.py |
| CATE Sequence Model | s2i_decompose_mask_attention_cate_24_72_168_ablation.py |
| OC-SVM | s2i_decompose_mask_attention_ocsvm_24_72_168_ablation.py |
| TCN Baseline | s2i_decompose_mask_attention_tcn_24_72_168_ablation.py |
| XGBoost | s2i_decompose_mask_attention_xgb_24_72_168_ablation.py |

---

# 📊 5. Ablation Settings

Each model contains four variants:

1. **Full configuration**  
2. **Without deviation modulation**  
3. **Without DWT/FFT**  
4. **Without attention module**

Metrics reported:

- Precision  
- Recall  
- F1-score  

Across 24h / 72h / 168h behavior windows.

---

# ▶️ 6. Usage Instructions

## Install dependencies
```bash
pip install -r requirements.txt
```

## Run preprocessing
```bash
python Preprocess/step1_log_split.py  -c config.yaml
python Preprocess/step2_log_merge.py  -c config.yaml
python Preprocess/step3_log_labeling.py  -c config.yaml
python Preprocess/step4_hourly_stat.py  -c config.yaml
```

## Run Base experiments
```bash
python S2I_Behavior_Model/Base_Model/s2i_decompose_mask_attention_tcn.py
```


## Run ablation experiments
```bash
# TCN
python S2I_Behavior_Model/Ablation/s2i_decompose_mask_attention_tcn_24_72_168_ablation.py

# CapsNet
python S2I_Behavior_Model/Ablation/s2i_decompose_mask_attention_caps_24_72_168_ablation.py

# XGBoost
python S2I_Behavior_Model/Ablation/s2i_decompose_mask_attention_xgb_24_72_168_ablation.py

# OCSVM
python S2I_Behavior_Model/Ablation/s2i_decompose_mask_attention_ocsvm_24_72_168_ablation.py

# CATE
python S2I_Behavior_Model/Ablation/s2i_decompose_mask_attention_cate_24_72_168_ablation.py

```
---


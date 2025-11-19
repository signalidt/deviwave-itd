# deviwave-itd
Deviation-aware multi-resolution insider-threat detection from logs via behavioral matrices, DWT subbands, and resolution-aware attention.

## Overview
DeviWave-ITD builds behavioral matrices from user logs, applies deviation-aware reweighting, performs DWT-based multi-resolution decomposition with resolution-aware attention, and feeds the representation to a detector to produce anomaly scores.


### Multi-Modal Decomposition, Attention, and Sequential Modeling for Insider Threat Detection

This repository provides a complete end-to-end pipeline for **insider threat detection** based on user behavior analytics.  
The system includes:

- A full **data preprocessing pipeline** that converts raw CERT logs into structured 24h / 72h / 168h behavioral sequences.
- A unified **S2I (Signal-to-Insight)** modeling framework integrating:
  - Wavelet & FFT decomposition  
  - Deviation-based masking  
  - Attention modules (SE/CBAM)  
  - TCN, Capsule Network, CATE, XGBoost, OCSVM models  
- A consistent **ablation evaluation framework** for all models.

---
# 📂 Project Structure

Project/
│── imgs/
│
│── Preprocess/
│ ├── output/
│ │ ├── log_merged/
│ │ ├── log_merged_24_hours/
│ │ └── log_split/
│ │ ├── http_domains.csv
│ │ ├── http_domains_with_category.csv
│ │ ├── ldap_with_device.csv
│ │ ├── ldap_with_device_department.csv
│ │ └── ...
│ │
│ ├── config.yaml
│ ├── department_relationship_extract.py
│ ├── device_extract.py
│ ├── domain_categories.py
│ ├── domain_extract.py
│ ├── step1_log_split.py
│ ├── step2_log_merge.py
│ ├── step3_log_labeling.py
│ └── step4_hourly_stat.py
│
│── S2I_Behavior_Model/
│ ├── Base_Model/
│ │ └── s2i_decompose_mask_attention_tcn.py
│ │
│ ├── Ablation/
│ │ ├── s2i_decompose_mask_attention_caps_24_72_168_ablation.py
│ │ ├── s2i_decompose_mask_attention_cate_24_72_168_ablation.py
│ │ ├── s2i_decompose_mask_attention_ocsvm_24_72_168_ablation.py
│ │ ├── s2i_decompose_mask_attention_tcn_24_72_168_ablation.py
│ │ └── s2i_decompose_mask_attention_xgb_24_72_168_ablation.py
│
└── README.md


---

# 🧩 1. Preprocessing Pipeline

The raw CERT logs are transformed into hourly behavioral profiles through the following steps:

### **Step 1 — Log Splitting**
Separates HTTP, LDAP, EMAIL, FILE, DEVICE logs into structured CSVs.

### **Step 2 — Log Merging**
Merges all event categories by timestamp for each user.

### **Step 3 — Log Labeling**
Assigns anomaly labels based on scenario descriptions.

### **Step 4 — Hourly Behavioral Statistics**
Produces 24-hour daily behavior matrices containing:

- device_count  
- email_count  
- file_count  
- http_count  
- logon_count  
- total_behavior_count  

These form the 6×24 behavioral signals used by all models.

---

# 🔬 2. S2I Behavior Modeling Framework

The S2I framework converts behavioral matrices into enriched multi-modal signals:

---

## **2.1 Frequency Decomposition**

### **Wavelet Decomposition (DWT)**
Extracts:
- Approximation coefficients (cA)  
- Horizontal, Vertical, Diagonal details (cH, cV, cD)

### **FFT Band Decomposition**
Produces:
- Low-frequency band  
- Mid-frequency band  
- High-frequency band  

---

## **2.2 Deviation Mask Modulation (DMM)**

A robust noise-resilient enhancement:

delta = |x - μ| / σ
if delta < 1.0 → mask = 0.7
else → mask = 1 + 0.5*delta
enhanced = x * mask


Amplifies unusual spikes while reducing normal background noise.

---

## **2.3 Attention Module**

Two attention mechanisms are applied:

- **SEBlock**: channel-wise recalibration  
- **CBAM**: channel + spatial attention  

Integrated before sequence modeling.

---

# 🔥 3. Base Model: Attention-TCN

The primary deep model combines:

- Input decomposition (DWT or FFT)
- Mask-based modulation  
- CBAM attention  
- TCN layers with dilation  
- MLP classifier  

It supports three window configurations:  
**24h**, **72h**, **168h**

---

# 🧪 4. Ablation Models

To deeply analyze each component, multiple models are implemented.

### **✔ Capsule Network (CapsNet)**
`Ablation/s2i_decompose_mask_attention_caps_24_72_168_ablation.py`

### **✔ CATE Sequence Model**
`Ablation/s2i_decompose_mask_attention_cate_24_72_168_ablation.py`

### **✔ One-Class SVM (OC-SVM)**
`Ablation/s2i_decompose_mask_attention_ocsvm_24_72_168_ablation.py`

### **✔ TCN Baseline**
`Ablation/s2i_decompose_mask_attention_tcn_24_72_168_ablation.py`

### **✔ XGBoost (Tree-Based Baseline)**
`Ablation/s2i_decompose_mask_attention_xgb_24_72_168_ablation.py`

---

# 📊 5. Ablation Settings

Each model supports four ablation variants:

1. **full**  
2. **w/o Deviation Modulation**  
3. **w/o DWT / w/o FFT**  
4. **w/o Attention**

Experiments run on:

- **24-hour windows**  
- **72-hour windows**  
- **168-hour windows**  

Metrics reported:

- **Precision**  
- **Recall**  
- **F1 score**

---

# ▶️ 6. Usage Instructions

## **1. Install dependencies**

pip install -r requirements.txt

## **2. Run preprocessing**

python Preprocess/step1_log_split.py
python Preprocess/step2_log_merge.py
python Preprocess/step3_log_labeling.py
python Preprocess/step4_hourly_stat.py


## **3. Run ablation experiments**

TCN: python S2I_Behavior_Model/Ablation/s2i_decompose_mask_attention_tcn_24_72_168_ablation.py

CapsNet: python S2I_Behavior_Model/Ablation/s2i_decompose_mask_attention_caps_24_72_168_ablation.py

XGBoost: python S2I_Behavior_Model/Ablation/s2i_decompose_mask_attention_xgb_24_72_168_ablation.py



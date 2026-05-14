# DeviWave-ITD

<p align="center">
  <a href="./README.md">English</a> |
  <a href="./README_CN.md">简体中文</a>
</p>

# DeviWave-ITD：基于多分辨率行为矩阵与小波分解的内部威胁检测框架

## 📘 项目简介

**DeviWave-ITD** 是一种面向内部威胁检测的端到端行为分析框架，
旨在将原始用户行为日志转换为多分辨率行为信号，
并结合频域分解与深度时序建模实现异常行为检测。:contentReference[oaicite:0]{index=0}

该框架主要融合了：

- 行为矩阵构建（Behavioral Matrix Construction）
- 偏差增强调制（Deviation-aware Modulation）
- DWT / FFT 频域分解
- 注意力增强时序建模
- 多模型统一消融实验框架

整个流程能够将原始 CERT 日志转换为：

- 24 小时行为窗口
- 72 小时行为窗口
- 168 小时行为窗口

等多尺度行为序列，
并进一步输入深度学习模型与传统机器学习模型进行异常检测。:contentReference[oaicite:1]{index=1}

---

# 📂 项目结构

```bash
Project/
│── imgs/
│
│── Preprocess/
│   ├── output/
│   │   ├── log_merged/
│   │   ├── log_merged_24_hours/
│   │   └── log_split/
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
│   ├── Ablation/
│   └── ...
│
└── README.md
```

---

# 🧩 数据预处理流程

项目首先对 CERT 原始日志进行统一预处理，
并通过四个阶段构建结构化行为矩阵。:contentReference[oaicite:2]{index=2}

## Step 1 — 日志切分（Log Splitting）

将原始日志按照类别进行拆分：

- HTTP
- LDAP
- Email
- File
- Device

---

## Step 2 — 日志融合（Log Merging）

按照时间顺序，
对不同类别日志进行用户级别融合。

---

## Step 3 — 行为标注（Log Labeling）

依据场景规则对行为进行异常标签标注。

---

## Step 4 — 小时级行为统计（Hourly Behavioral Statistics）

最终生成：

## 6×24 行为矩阵

统计以下行为计数：

- device_count
- email_count
- file_count
- http_count
- logon_count
- total_behavior_count

该 6 通道行为矩阵作为所有模型的基础输入。:contentReference[oaicite:3]{index=3}

---

# 🔬 S2I（Signal-to-Insight）行为建模框架

S2I 框架用于将行为矩阵进一步转换为多模态行为特征表示。:contentReference[oaicite:4]{index=4}

---

# 🌊 频域分解模块

## 小波分解（DWT）

通过离散小波变换生成多分辨率行为特征：

- cA：低频近似分量
- cH：水平细节分量
- cV：垂直细节分量
- cD：对角细节分量

---

## FFT 频带分解

进一步生成：

- 低频特征
- 中频特征
- 高频特征

用于捕获不同尺度下的行为变化模式。:contentReference[oaicite:5]{index=5}

---

# 🔥 偏差增强调制（DMM）

项目提出 DeviWave 偏差增强机制，
用于强化异常行为峰值并抑制正常噪声。:contentReference[oaicite:6]{index=6}

核心思想如下：

```python
delta = |x - μ| / σ

if delta < 1.0:
    mask = 0.7
else:
    mask = 1 + 0.5 * delta

enhanced = x * mask
```

其作用包括：

- 抑制正常行为波动
- 放大异常偏差
- 提升异常可分离性

---

# 🎯 注意力增强模块

项目采用两种注意力机制：

- SEBlock
- CBAM

分别实现：

- 通道注意力增强
- 空间与通道联合建模

用于提升频域行为特征表达能力。:contentReference[oaicite:7]{index=7}

---

# 🧠 基础模型：Attention-TCN

核心深度学习模型融合了：:contentReference[oaicite:8]{index=8}

- 多分辨率分解
- 偏差调制
- CBAM 注意力
- 膨胀卷积 TCN
- MLP 分类器

支持：

- 24 小时行为窗口
- 72 小时行为窗口
- 168 小时行为窗口

---

# 🧪 消融实验模型

项目实现了多个消融实验模型：:contentReference[oaicite:9]{index=9}

| 模型 | 说明 |
|---|---|
| CapsNet | 胶囊网络 |
| CATE | 时序行为模型 |
| OC-SVM | 单类支持向量机 |
| TCN | 时序卷积网络 |
| XGBoost | 梯度提升树 |

---

# 📊 消融实验设置

每个模型均包含以下四种配置：:contentReference[oaicite:10]{index=10}

1. 完整模型
2. 去除偏差增强
3. 去除 DWT / FFT
4. 去除注意力模块

评估指标包括：

- Precision
- Recall
- F1-score

并分别在：

- 24h
- 72h
- 168h

行为窗口下进行实验。

---

# ▶️ 使用说明

## 安装依赖

```bash
pip install -r requirements.txt
```

---

## 数据预处理

```bash
python Preprocess/step1_log_split.py  -c config.yaml

python Preprocess/step2_log_merge.py  -c config.yaml

python Preprocess/step3_log_labeling.py  -c config.yaml

python Preprocess/step4_hourly_stat.py  -c config.yaml
```

---

## 运行基础模型

```bash
python S2I_Behavior_Model/Base_Model/s2i_decompose_mask_attention_tcn.py
```

---

## 运行消融实验

### TCN

```bash
python S2I_Behavior_Model/Ablation/s2i_decompose_mask_attention_tcn_24_72_168_ablation.py
```

### CapsNet

```bash
python S2I_Behavior_Model/Ablation/s2i_decompose_mask_attention_caps_24_72_168_ablation.py
```

### XGBoost

```bash
python S2I_Behavior_Model/Ablation/s2i_decompose_mask_attention_xgb_24_72_168_ablation.py
```

### OCSVM

```bash
python S2I_Behavior_Model/Ablation/s2i_decompose_mask_attention_ocsvm_24_72_168_ablation.py
```

### CATE

```bash
python S2I_Behavior_Model/Ablation/s2i_decompose_mask_attention_cate_24_72_168_ablation.py
```

---

# 👥 作者

- 孔凯传（Kaichuan Kong）
- 刘东杰（Dongjie Liu）
- 耿光刚（Guanggang Geng）

---

# 🏫 作者单位

暨南大学（Jinan University）  
网络空间安全学院（College of Cyberspace Security）  
中国 · 广州

---

# ⚠️ 使用声明

本项目仅用于学术研究、教学实验与科研复现。  

项目中涉及的数据集（如 CERT 数据集）均遵循其原始许可协议使用。  

作者及所属单位不对因使用本项目所造成的任何直接或间接后果承担责任。

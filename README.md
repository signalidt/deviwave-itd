# deviwave-itd
Deviation-aware multi-resolution insider-threat detection from logs via behavioral matrices, DWT subbands, and resolution-aware attention.

> ⚠️ **Project status:** Work in progress. We are actively adding code, docs, and experiments.  


---

## Overview
DeviWave-ITD builds behavioral matrices from user logs, applies deviation-aware reweighting, performs DWT-based multi-resolution decomposition with resolution-aware attention, and feeds the representation to a detector to produce anomaly scores.

## Getting Started (WIP)
```bash
# (Coming soon) environment setup
# conda create -n deviwave python=3.10 -y
# conda activate deviwave
# pip install -r requirements.txt

# (Coming soon) data preparation
# python scripts/prepare_data.py --dataset cert42 ...

# (Coming soon) training & evaluation
# python scripts/train.py --config configs/cert42_transformer.yaml
# python scripts/eval.py --checkpoint outputs/ckpt.pt

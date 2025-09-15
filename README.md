# Breast Cancer Cell Classification (Bright-field, Stain-free) using CNN

This repo provides a **reproducible PyTorch pipeline** to classify **live vs. dead** breast cancer cells from **bright-field** microscopy, without staining.  
It includes data loaders, training/evaluation scripts, metrics (AUC/F1) and **Grad-CAM** visualizations.

## TL;DR
```
# 1) Create env
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Train
python -m breast_cancer_cnn.train --config configs/default.yaml

# 3) Evaluate & Grad-CAM
python -m breast_cancer_cnn.eval --ckpt runs/best.ckpt
python -m breast_cancer_cnn.infer --image assets/sample.png --ckpt runs/best.ckpt
```

Our paper on dead and living breast cancer cell image classification was placed as one of the top 100 downloaded cancer papers for Nature Scientific Reports (among more than 1,440 cancer papers in 2021). More info here: https://www.nature.com/collections/gdfhjfggib.

![image](https://user-images.githubusercontent.com/91725761/163794416-6b2592f5-817d-4b33-8122-ef2db6b531ed.png)
![Patches segmentation](https://user-images.githubusercontent.com/91725761/163794193-8cb07bac-561d-46f3-b5e3-37f188d0c741.jpg)
![image](https://user-images.githubusercontent.com/91725761/163794383-653fb0d8-33d3-487f-b789-c07255b0838c.png)

The image dataset and further resources are available in the public github repository:: https://github.com/emmanueliarussi/live-dead-JIMT-1.git

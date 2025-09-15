# Breast Cancer Cell Classification (Live vs Dead, Bright-field)

This repository provides a **reproducible PyTorch pipeline** to classify **live vs dead breast cancer cells** from **bright-field microscopy images**, without staining.  
It includes modular code for data loading, model definition, training, and evaluation, following best practices in reproducible machine learning.

---

## Features
- CNN classifier based on **ResNet18**.
- **Grayscale, resize & augmentations** for stain-free images.
- Modular structure:
  - `src/data.py` → data loading utilities.
  - `src/models/cnn.py` → model definition.
  - `src/train.py` → training loop.
- Configurable hyperparameters via YAML.
- Saves the **best model checkpoint** automatically.
- Ready to extend with Grad-CAM and advanced metrics.

---

## Repository structure
```

Breast-Cancer-Cell-CNN-classification/
│
├── src/
│   ├── data.py          # data transforms & dataloaders
│   ├── models/
│   │   └── cnn.py       # ResNet18 definition
│   └── train.py         # training script
│
├── configs/
│   └── default.yaml     # training hyperparameters
│
├── requirements.txt     # dependencies
├── README.md            # project documentation
└── LICENSE              # open source license

```
## Dataset

This project uses the **JIMT-1 live/dead cell dataset** (bright-field microscopy).  
The dataset must be structured as:

```
data/
train/
live/*.png
dead/*.png
val/
live/*.png
dead/*.png
test/
live/*.png
dead/*.png
```

## Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/gpattarone/Breast-Cancer-Cell-CNN-classification.git
cd Breast-Cancer-Cell-CNN-classification
python -m venv .venv
source .venv/bin/activate    # On Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
````
---

## Training

Run the training script with default config:

```bash
python src/train.py --config configs/default.yaml
```

This will:

* Load training and validation sets from `data/`.
* Train for the specified number of epochs.
* Save the **best model** at `runs/best_model.pt`.

---

## Evaluation

An `eval.py` script provided to:

* Evaluate on the test set.
* Generate metrics: accuracy, AUC, F1-score.
* Plot confusion matrix and ROC curves.

---

## Model Card (summary)

* **Intended use**: research/educational purposes on live/dead classification in breast cancer cells.
* **Not intended for**: direct clinical decisions without validation.
* **Dataset**: bright-field microscopy, JIMT-1 cells.
* **Limitations**: performance may degrade on different imaging setups or cell lines.
* **Biases**: only trained on one dataset → limited generalization.

---
## Citation

If you use this code, please cite:

```
@article{Pattarone2025BCC,
  author  = {Gisela R. Pattarone},
  title   = {Breast Cancer Cell Classification (Bright-field, Stain-free)},
  year    = {2025},
  journal = {GitHub Repository},
  url     = {https://github.com/gpattarone/Breast-Cancer-Cell-CNN-classification}
}
```
Our paper on dead and living breast cancer cell image classification was placed as one of the top 100 downloaded cancer papers for Nature Scientific Reports (among more than 1,440 cancer papers in 2021). More info here: https://www.nature.com/collections/gdfhjfggib.

![image](https://user-images.githubusercontent.com/91725761/163794416-6b2592f5-817d-4b33-8122-ef2db6b531ed.png)
![Patches segmentation](https://user-images.githubusercontent.com/91725761/163794193-8cb07bac-561d-46f3-b5e3-37f188d0c741.jpg)
![image](https://user-images.githubusercontent.com/91725761/163794383-653fb0d8-33d3-487f-b789-c07255b0838c.png)

The image dataset and further resources are available in the following public repository: [(https://github.com/gpattarone/https-github.com-gpattarone-live-dead-JIMT-1.git)](https://github.com/gpattarone/https-github.com-gpattarone-live-dead-JIMT-1.git)

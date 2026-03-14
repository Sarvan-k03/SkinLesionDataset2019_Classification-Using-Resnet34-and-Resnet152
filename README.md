# Skin Lesion Classification using Deep Residual Networks (ResNet-34 & ResNet-152)

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Dataset](https://img.shields.io/badge/Dataset-ISIC%202019-green)
![Task](https://img.shields.io/badge/Task-Medical%20Image%20Classification-orange)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

## Project Overview

Early detection of skin cancer significantly improves patient survival rates. This project explores **deep convolutional neural networks for automated skin lesion classification** using the **ISIC 2019 dataset**.

Two architectures are implemented and evaluated:

| Model          | Approach          | Description                          |
| -------------- | ----------------- | ------------------------------------ |
| **ResNet-34**  | From Scratch      | Custom implementation using PyTorch  |
| **ResNet-152** | Transfer Learning | Fine-tuned ImageNet pretrained model |

The objective is to compare **training from scratch vs transfer learning** for medical image classification.

---

# Dataset

**Dataset:** ISIC 2019 Skin Lesion Challenge Dataset

Total Images:

| Split    | Images |
| -------- | ------ |
| Training | 25,331 |
| Test     | 8,238  |

Number of classes: **8**

### Lesion Categories

| Code | Disease                 |
| ---- | ----------------------- |
| MEL  | Melanoma                |
| NV   | Melanocytic Nevus       |
| BCC  | Basal Cell Carcinoma    |
| AK   | Actinic Keratosis       |
| BKL  | Benign Keratosis        |
| DF   | Dermatofibroma          |
| VASC | Vascular Lesion         |
| SCC  | Squamous Cell Carcinoma |

Dataset link:
https://challenge.isic-archive.com/data/

---

# Model Architectures

## ResNet-34 (From Scratch)

ResNet-34 is implemented manually using PyTorch to understand the internal architecture of residual networks.

Key components:

• Residual blocks
• Batch normalization
• ReLU activations
• Skip connections
• Global average pooling

This model learns **all parameters directly from the skin lesion dataset**.

---

## ResNet-152 (Transfer Learning)

A deeper architecture pretrained on **ImageNet** is fine-tuned for skin lesion classification.

Modifications:

• Final fully connected layer replaced
• Output neurons changed to **8 classes**
• Earlier layers optionally frozen during training

Transfer learning allows the model to leverage **general visual features learned from millions of images**.


---

# Installation

Clone the repository

```
git clone https://github.com/Sarvan-k03/SkinLesionDataset2019_Classification-Using-Resnet34-and-Resnet152.git
cd skin-lesion-classification
```

Install dependencies

```
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn tqdm pillow opencv-python
```

---

# Dataset Setup

Example download using `wget`

```
wget https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip
wget https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv

unzip ISIC_2019_Training_Input.zip
```

Organize the dataset into:

```
dataset/
   images/
   labels.csv
```

---

# Training Pipeline

The training workflow consists of:

1. Dataset loading with custom PyTorch `Dataset` class
2. Image preprocessing and augmentation
3. Model training with cross entropy loss
4. Optimization using Adam optimizer
5. Validation and performance evaluation

---

# Evaluation Metrics

Medical classification requires robust evaluation. The following metrics are used:

• Accuracy
• Balanced Accuracy
• Precision
• Recall (Sensitivity)
• Specificity
• F1 Score
• ROC-AUC
• Confusion Matrix
• Cohen's Kappa

---

# Example Results

*(Replace with your actual results)*

| Model      | Accuracy | Balanced Accuracy | F1 Score |
| ---------- | -------- | ----------------- | -------- |
| ResNet-34  | 0.59     | 0.51              | 0.61     |
| ResNet-152 | 0.69     | 0.54              | 0.68     |

Key observations:

• Transfer learning significantly improves performance
• Deeper networks capture more complex lesion patterns
• Class imbalance affects rare disease prediction

---

# Future Improvements

Possible extensions:

• Vision Transformers (ViT) for dermatology
• Class imbalance handling using **Focal Loss**
• Data augmentation using **GANs**
• Ensemble learning with multiple CNNs
• Clinical interpretability using **Grad-CAM**

---

# Applications

Potential applications include:

• AI-assisted dermatology diagnosis
• Clinical decision support systems
• Early melanoma detection
• Tele-dermatology screening tools

---

# Author

**Sarvan Kumar**
---

# License

This project is intended for **research and educational purposes only**.



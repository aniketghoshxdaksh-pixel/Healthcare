# Multi-Modal Hybrid Alzheimer's Stage Classification
### Deep Learning Analysis on OASIS MRI Neuroimaging & Clinical Metadata

![Alzheimer's Disease](https://img.shields.io/badge/Disease-Alzheimer's%20Disease-orange)
![Deep Learning](https://img.shields.io/badge/Approach-Multi--Modal%20Hybrid%20CNN-blue)
![Dataset](https://img.shields.io/badge/Dataset-OASIS--1%20MRI-green)

## 📌 Project Overview

This project implements a **Multi-Modal Hybrid CNN** that combines spatial features extracted from MRI brain scans with clinical tabular metadata (e.g., age, socioeconomic status, MMSE scores, nWBV, eTIV). By fusing these complementary data modalities, the model achieves a more comprehensive and accurate classification of Alzheimer's Disease (AD) progression across four stages:

- Non-Demented
- Very Mild Demented
- Mild Demented
- Moderate Demented

The hybrid approach leverages both visual neuroimaging patterns and quantitative clinical indicators for improved diagnostic performance.

## 📂 Project Structure

```plaintext
oasis-alzheimers-detection-multi-class-c/
├── train/
│   ├── NonDemented/              # MRI slices: healthy controls
│   ├── VeryMildDemented/         # Early-stage neurodegeneration
│   ├── MildDemented/             # Progressive cognitive decline
│   ├── ModerateDemented/         # Advanced stage atrophy
│   └── train.roboflow.txt        # Dataset manifest/versioning
├── test/
│   ├── NonDemented/
│   ├── VeryMildDemented/
│   ├── MildDemented/
│   ├── ModerateDemented/
│   └── test.roboflow.txt
├── oasis_train_patients_metadata.csv  # Clinical features for training
└── oasis_test_patients_metadata.csv   # Clinical features for testing

```

## 🔬 Technical Methodology

### 1. Hybrid Multi-Modal Architecture

Built using Keras Functional API with two parallel branches:

- **Vision Branch**: CNN processing 128×128×3 MRI slices through Conv2D, BatchNormalization, and MaxPooling layers.
- **Tabular Branch**: Dense Multi-Layer Perceptron (MLP) processing normalized clinical features (e.g., MMSE, SES, nWBV, eTIV, ASF).
- **Fusion Layer**: Concatenation of flattened 128-D image embeddings with 16-D clinical feature vector.
- **Classification Head**: Dense layers ending in Softmax for 4-class probability output.

### 2. Mathematical Foundation

The final layer uses **Softmax activation**:

$$
\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}, \quad K = 4
$$

Loss function: **Categorical Cross-Entropy**

### 3. Training Hyperparameters

| Parameter              | Value                          |
| ---------------------- | ------------------------------ |
| Optimizer              | Adam (Keras 3 optimized)       |
| Initial Learning Rate  | 0.001                          |
| LR Scheduler           | ReduceLROnPlateau              |
| Input Image Size       | 128 × 128 × 3                  |
| Total Parameters       | 103,572 (404.58 KB)            |

## 📊 Performance Analysis

### Classification Report (Hybrid Model on Test Set)

| Stage                  | Precision | Recall | F1-Score |
| ---------------------- | --------- | ------ | -------- |
| Non-Demented           | 0.98      | 0.86   | 0.93     |
| Very Mild Demented     | 0.31      | 0.57   | 0.40     |
| Mild Demented          | 0.23      | 0.80   | 0.36     |
| Moderate Demented      | 0.00      | 0.00   | 0.00     |

**Overall Accuracy**: **0.84**

> Note: Severe class imbalance affects performance on the ModerateDemented class.

## 🔑 Key Findings & Conclusion

- **Multi-Modal Synergy**: Combining MRI structural features (e.g., brain atrophy via nWBV) with cognitive/clinical scores (MMSE) significantly enhances stage discrimination.
- **High Screening Utility**: 80% recall on Mild Demented stage indicates strong potential as an automated first-line screening tool.
- **Challenge Identified**: Extreme underrepresentation of ModerateDemented samples limits performance on advanced stages.

**Future Improvements**:
- Apply class-balancing techniques (SMOTE, focal loss, weighted sampling).
- Generate synthetic MRI samples using GANs or diffusion models.
- Explore attention-based fusion mechanisms.



# Run the Project
The full implementation and interactive training notebook is available on Kaggle: 
- https://www.kaggle.com/code/daksh4/hybrid-cnn-clinical-metadata-84
- 🔗 Hybrid CNN + Clinical Metadata for Alzheimer's Classification
- (Replace with your actual Kaggle notebook link)
Inference Example
## 🛠️ Installation & Reproduction

### Environment Setup

```bash
pip install tensorflow keras pandas numpy scikit-learn matplotlib seaborn



)
```

# 🎓 Citation & Credits

- Dataset: OASIS-1 Cross-Sectional MRI Data
- https://www.oasis-brains.org
- Kaggle Dataset Host: Shreyan Mohanty
- Model & Implementation: Aniket Ghosh
- Developed as part of MTeach graduate Research
LinkedIn: Aniket Ghosh


# Model expects two inputs: image batch and tabular batch
```bash
prediction = model.predict([image_batch, tabular_batch])

# Predicted probabilities for 4 classes
print(prediction)

# Predicted class
predicted_class = np.argmax(prediction, axis=1)

```

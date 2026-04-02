# Heart Disease Prediction & Patient Risk Segmentation

## 📋 Project Overview

This is a machine learning project that predicts heart disease risk and segments patients into risk groups to assist in clinical decision-making. The project implements multiple supervised and unsupervised learning algorithms to analyze cardiac health indicators and provide comprehensive patient risk assessments.

**Dataset**: UCI Heart Disease Dataset (Real-world clinical data)

---

## 🎯 Project Objectives

1. **Disease Prediction**: Develop accurate models to predict presence/absence of heart disease
2. **Risk Stratification**: Segment patients into distinct risk groups for targeted interventions
3. **Feature Analysis**: Identify key cardiac indicators that predict disease risk
4. **Model Comparison**: Compare performance across multiple algorithm approaches

---

## 📊 Dataset Information

- **Source**: UCI Machine Learning Repository - Heart Disease Dataset
- **Total Instances**: 303 (Cleveland), 294 (Hungarian), 200 (Long Beach VA), 123 (Switzerland)
- **Features**: 13 clinical attributes
- **Target Variable**: Heart disease presence (0 = absent, 1-4 = present)
- **Missing Values**: Marked as -9.0

### Key Features

| Feature  | Description                                                 |
| -------- | ----------------------------------------------------------- |
| age      | Age in years                                                |
| sex      | Gender (1=male, 0=female)                                   |
| cp       | Chest pain type (1-4)                                       |
| trestbps | Resting blood pressure (mm Hg)                              |
| chol     | Serum cholesterol (mg/dl)                                   |
| fbs      | Fasting blood sugar > 120 mg/dl (1=true, 0=false)           |
| restecg  | Resting electrocardiographic results (0-2)                  |
| thalach  | Maximum heart rate achieved                                 |
| exang    | Exercise-induced angina (1=yes, 0=no)                       |
| oldpeak  | ST depression induced by exercise                           |
| slope    | Slope of peak exercise ST segment (1-3)                     |
| ca       | Number of major vessels colored by fluoroscopy (0-3)        |
| thal     | Thalassemia (3=normal, 6=fixed defect, 7=reversible defect) |

---

## 🤖 Algorithm Components

### 1. **Logistic Regression** - Member 1

- **Type**: Supervised Learning (Baseline)
- **Purpose**: Binary classification (disease / no disease)
- **Key Task**: Establish baseline model performance
- **Output**: Probability scores, classification, and evaluation plots
- **Files**: `src/data_preprocessing.py`, `src/logistic_regression_model.py`, `src/evaluation.py`
- **Notebook**: `notebooks/01_logistic_regression_analysis.ipynb`

### 2. **Random Forest** - Member 2

- **Type**: Supervised Learning (Ensemble)
- **Purpose**: Improved prediction with feature importance analysis
- **Key Task**: Identify most influential cardiac indicators
- **Output**: Feature rankings and prediction scores
- **File**: `src/02_random_forest.py`

### 3. **Support Vector Machine (SVM)** - Member 3

- **Type**: Supervised Learning (Kernel Methods)
- **Purpose**: Classification with multiple kernel options
- **Kernels**: Linear, RBF, Polynomial
- **Key Task**: Find optimal decision boundaries
- **File**: `src/03_svm_classification.ipynb`

### 4. **K-Means Clustering** - Member 4

- **Type**: Unsupervised Learning
- **Purpose**: Patient risk segmentation
- **Key Task**: Identify natural patient subgroups
- **Output**: Risk group assignments (Low, Medium, High, Critical)
- **File**: `src/04_kmeans_clustering.py`

---

## 📁 Project Structure

```
heart-disease-prediction/
│
├── Data_set/                          # Raw UCI dataset files
│   ├── processed.cleveland.data       # Cleveland (303 instances) ✅ used
│   ├── processed.hungarian.data       # Hungarian (294 instances)
│   ├── processed.switzerland.data     # Switzerland (123 instances)
│   ├── processed.va.data              # Long Beach VA (200 instances)
│   ├── cleveland.data                 # Full Cleveland data
│   ├── hungarian.data                 # Full Hungarian data
│   ├── long-beach-va.data             # Full VA data
│   ├── switzerland.data               # Full Switzerland data
│   ├── reprocessed.hungarian.data     # Reprocessed Hungarian data
│   ├── heart-disease.names            # Feature descriptions
│   ├── new.data
│   ├── cleve.mod
│   ├── bak
│   ├── ask-detrano
│   ├── WARNING
│   ├── Index
│   └── costs/                         # Cost metadata
│       ├── heart-disease.cost
│       ├── heart-disease.delay
│       ├── heart-disease.expense
│       ├── heart-disease.group
│       ├── heart-disease.README
│       └── Index
│
├── src/                               # Model scripts and utilities
│   ├── data_preprocessing.py          # Preprocessing utilities for Member 1
│   ├── logistic_regression_model.py    # Logistic Regression training for Member 1
│   ├── evaluation.py                  # Evaluation utilities for Member 1
│   ├── 02_random_forest.py            # Member 2 — Random Forest ✅
│   ├── 03_svm_classification.ipynb    # Member 3 — Support Vector Machine
│   └── 04_kmeans_clustering.py        # Member 4 — K-Means Clustering
│
├── models/                            # Saved trained models
│   ├── logistic_regression_model.pkl  # Member 1 model
│   ├── random_forest_model.pkl        # Member 2 model ✅
│   ├── svm_heart_disease_model.joblib # Member 3 model
│   └── kmeans_model.pkl               # Member 4 model
│
├── results/                           # Generated outputs for Logistic Regression
├── outputs/                           # Generated outputs per member
│   ├── logistic_regression/           # Member 1 outputs
│   │   ├── confusion_matrix.png
│   │   ├── roc_curve.png
│   │   └── summary_report.txt
│   ├── random_forest/                 # Member 2 outputs ✅
│   │   ├── feature_importance.png
│   │   ├── confusion_matrix.png
│   │   ├── roc_curve.png
│   │   ├── cv_scores.png
│   │   ├── top5_features_boxplot.png
│   │   └── summary_report.txt
│   ├── svm_classification/                           # Member 3 outputs
│   │   ├── confusion_matrix.csv
│   │   ├── confusion_matrix.png
│   │   ├── roc_curve.png
│   │   ├── summary_report.txt
│   │   └── svm_predictions.csv
│   │
│   │
│   │
│   └── kmeans/                        # Member 4 outputs
│       ├── 01_elbow_plot.png
│       ├── 02_silhouette_analysis.png
│       ├── 03_pca_scatter_3d.png
│       ├── 04_cluster_heatmap.png
│       ├── 05_cluster_summary.csv
│       ├── clustered_patients.csv
│       ├── pca_transformed_data.csv
│       └── ANALYSIS_REPORT.txt
│
└── README.md
```

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager
- Git

### Step 1: Clone Repository

```bash
git clone https://github.com/Yenura/heart-disease-prediction.git
cd heart-disease-prediction
```

## Member 1 Contribution

This contribution includes the Logistic Regression pipeline for heart disease prediction:

- `notebooks/01_logistic_regression_analysis.ipynb` for analysis, training, and evaluation.
- `src/data_preprocessing.py` for dataset loading, cleaning, type conversion, missing value handling, and binary target conversion.
- `src/logistic_regression_model.py` for scaling and model training.
- `src/evaluation.py` for metrics, confusion matrix, ROC curve, and result persistence.
- `results/` for saved logistic regression outputs.

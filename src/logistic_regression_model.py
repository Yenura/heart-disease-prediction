import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

COLUMN_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol',
    'fbs', 'restecg', 'thalach', 'exang',
    'oldpeak', 'slope', 'ca', 'thal', 'target'
]

DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'Data_set',
    'processed.cleveland.data'
)


def load_data(data_path: Optional[str] = None) -> pd.DataFrame:
    """Load the Cleveland heart disease dataset from disk."""
    if data_path is None:
        data_path = DEFAULT_DATA_PATH

    # Read the dataset into a pandas DataFrame.
    return pd.read_csv(data_path, names=COLUMN_NAMES, na_values='?', encoding='latin-1')


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw dataset, impute missing values, and binarize the target."""
    df = df.copy()
    df.replace('?', np.nan, inplace=True)

    for column in COLUMN_NAMES:
        df[column] = pd.to_numeric(df[column], errors='coerce')

    df.fillna(df.median(numeric_only=True), inplace=True)
    df['target'] = df['target'].apply(lambda value: 0 if value == 0 else 1)
    return df


def summarize_target(df: pd.DataFrame) -> pd.Series:
    """Return the class distribution for the binary target."""
    return df['target'].value_counts().sort_index()


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split the cleaned dataset into features and target label."""
    X = df.drop(columns=['target'])
    y = df['target']
    return X, y


def scale_train_test(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Fit StandardScaler on the training data and transform both splits."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    solver: str = 'lbfgs',
    max_iter: int = 1000,
    random_state: int = 42
) -> LogisticRegression:
    """Train a LogisticRegression classifier on the given data."""
    model = LogisticRegression(solver=solver, max_iter=max_iter, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def get_sorted_coefficients(model: LogisticRegression, feature_names: List[str]) -> pd.DataFrame:
    """Return a dataframe of coefficients sorted by absolute importance."""
    coefficients = model.coef_.flatten()
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients,
        'abs_coefficient': np.abs(coefficients)
    })
    coef_df = coef_df.sort_values('abs_coefficient', ascending=False).reset_index(drop=True)
    return coef_df


def build_logistic_regression_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
    solver: str = 'lbfgs',
    max_iter: int = 1000,
    stratify: Optional[pd.Series] = None
) -> Tuple[LogisticRegression, np.ndarray, np.ndarray, pd.Series, pd.Series, StandardScaler]:
    """Split data, scale features, and train a logistic regression model."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )
    X_train_scaled, X_test_scaled, scaler = scale_train_test(X_train, X_test)
    model = train_logistic_regression(
        X_train_scaled,
        y_train,
        solver=solver,
        max_iter=max_iter,
        random_state=random_state
    )
    return model, X_train_scaled, X_test_scaled, y_train, y_test, scaler


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, object]:
<<<<<<< HEAD
    """Compute standard binary classification evaluation metrics."""
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]

    return {
=======
    """Compute evaluation metrics for a binary classifier."""
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]

    metrics = {
>>>>>>> 22d0e86389bff860ffc4bca2eb3eb9f23cfe6ca0
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_score),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
<<<<<<< HEAD
        'classification_report': classification_report(y_test, y_pred, digits=4),
    }


def _ensure_folder(file_path: str):
    """Create parent directories for a path if they do not exist."""
=======
        'classification_report': classification_report(y_test, y_pred, digits=4)
    }
    return metrics


def _ensure_folder(file_path: str):
>>>>>>> 22d0e86389bff860ffc4bca2eb3eb9f23cfe6ca0
    os.makedirs(os.path.dirname(file_path), exist_ok=True)


def save_metrics(metrics: Dict[str, object], file_path: str):
<<<<<<< HEAD
    """Save evaluation metrics and classification report to a text file."""
=======
    """Write evaluation metrics and classification report to a text file."""
>>>>>>> 22d0e86389bff860ffc4bca2eb3eb9f23cfe6ca0
    _ensure_folder(file_path)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write('Logistic Regression Evaluation Metrics\n')
        file.write('===============================\n')
        file.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        file.write(f"Precision: {metrics['precision']:.4f}\n")
        file.write(f"Recall: {metrics['recall']:.4f}\n")
        file.write(f"F1 Score: {metrics['f1_score']:.4f}\n")
        file.write(f"ROC AUC Score: {metrics['roc_auc']:.4f}\n\n")
        file.write('Classification Report:\n')
        file.write(metrics['classification_report'])


def plot_confusion_matrix(confusion_matrix_data: np.ndarray, save_path: Optional[str] = None):
    """Plot and optionally save the confusion matrix figure."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(confusion_matrix_data, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['No Disease', 'Disease'])
    ax.set_yticklabels(['No Disease', 'Disease'])

    for i in range(confusion_matrix_data.shape[0]):
        for j in range(confusion_matrix_data.shape[1]):
            ax.text(j, i, int(confusion_matrix_data[i, j]), ha='center', va='center', color='black')

    plt.tight_layout()
    if save_path:
        _ensure_folder(save_path)
        plt.savefig(save_path, dpi=200)
    plt.close(fig)
    return fig


def plot_roc_curve(y_test: pd.Series, y_score: np.ndarray, save_path: Optional[str] = None):
    """Plot and optionally save the ROC curve figure."""
    fpr, tpr, _ = roc_curve(y_test, y_score)
    auc_value = roc_auc_score(y_test, y_score)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_value:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc='lower right')
    plt.tight_layout()

    if save_path:
        _ensure_folder(save_path)
        plt.savefig(save_path, dpi=200)
    plt.close(fig)
    return fig


if __name__ == '__main__':
<<<<<<< HEAD
    print('This module contains all logistic regression preprocessing, training, and evaluation helpers.')
=======
    print('This module now contains all logistic regression preprocessing, training, and evaluation helpers.')
>>>>>>> 22d0e86389bff860ffc4bca2eb3eb9f23cfe6ca0

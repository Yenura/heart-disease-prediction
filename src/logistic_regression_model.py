from typing import List, Optional, Tuple

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def scale_train_test(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fit a scaler on training data and transform both train and test sets."""
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
    """Train a logistic regression classifier on scaled training data."""
    model = LogisticRegression(solver=solver, max_iter=max_iter, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def get_sorted_coefficients(model: LogisticRegression, feature_names: List[str]) -> pd.DataFrame:
    """Return model coefficients sorted by absolute importance."""
    coefficients = model.coef_.flatten()
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients,
        'abs_coefficient': abs(coefficients)
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
) -> Tuple[LogisticRegression, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
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


if __name__ == '__main__':
    print('This module provides logistic regression helper functions for the notebooks.')

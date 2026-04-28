"""
Model evaluation utilities.

This module provides a unified interface for evaluating models across:
- Regression tasks
- Classification tasks (binary + multi-class)

Design goals:
- Keep evaluation consistent across models
- Use standard, interpretable metrics
- Return results in a simple dictionary format (easy to log / save)
"""

# -------------------- Imports --------------------
from sklearn.metrics import (
    root_mean_squared_error, accuracy_score, roc_auc_score
)

# -------------------- Evaluation Function --------------------
def evaluate(model, x_test, y_test, task="classification"):
    results = {}

    # -------------------- Regression --------------------
    if task == "regression":
        preds = model.predict(x_test)
        results["RMSE"] = root_mean_squared_error(y_test, preds)
    
    # -------------------- Classification --------------------
    else:
        preds = model.predict(x_test)

        # Basic classification accuracy
        results["Accuracy"] = accuracy_score(y_test, preds)
        proba = model.predict_proba(x_test)

        # Binary classification case
        if proba.shape[1] == 2:
            results["AUC"] = roc_auc_score(y_test, proba[:, 1])
        
        # Multiclass classification case
        else:
            results["AUC"] = roc_auc_score(y_test, proba, multi_class="ovr")
        
    return results
from sklearn.metrics import (
    root_mean_squared_error, accuracy_score, roc_auc_score
)

def evaluate(model, x_test, y_test, task="classification"):
    results = {}

    if task == "regression":
        preds = model.predict(x_test)
        results["RMSE"] = root_mean_squared_error(y_test, preds)
    
    else:
        preds = model.predict(x_test)
        results["Accuracy"] = accuracy_score(y_test, preds)
        proba = model.predict_proba(x_test)
        if proba.shape[1] == 2:
            results["AUC"] = roc_auc_score(y_test, proba[:, 1])
        else:
            results["AUC"] = roc_auc_score(y_test, proba, multi_class="ovr")
        
    return results
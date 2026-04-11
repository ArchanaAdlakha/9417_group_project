import time
import numpy as np
from xrfm import xRFM
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, accuracy_score

xRFM_max_samples = 5_000

seed = 42

XGBOOST_PARAM_GRID = {
    "n_estimators":     [100, 200, 300],
    "max_depth":        [3, 5, 7],
    "learning_rate":    [0.01, 0.05, 0.1, 0.2],
    "subsample":        [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0]
}

RF_PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth":    [None, 5, 10, 20],
    "max_features": ["sqrt", "log2"] 
}

XRFM_PARAM_GRID = {
    "n_iterations": [1, 3, 5, 10],
    "leaf_size":    [10, 20, 50, 100]  
}

def compute_val_score(model, x_val, y_val, task):
    if task == "regression":
        preds = model.predict(x_val)
        return root_mean_squared_error(y_val, preds)
    else:
        preds = model.predict(x_val)
        return 1 - accuracy_score(y_val, preds)

def tune_xgboost(x_train, y_train, x_val, y_val, task, n_trials=20):
    best_score, best_params = np.inf, None
    np.random.seed(seed)
    for _ in range(n_trials):
        params = {k: np.random.choice(v) for k, v in XGBOOST_PARAM_GRID.items()}
        model = get_xgboost(task, **params)
        model.fit(x_train, y_train)
        score = compute_val_score(model, x_val, y_val, task)
        if score < best_score:
            best_score, best_params = score, params
    print(f"Best XGBoost params: {best_params} (val score: {best_score:.4f})")
    return best_params

def tune_random_forest(x_train, y_train, x_val, y_val, task, n_trials=20):
    best_score, best_params = np.inf, None
    np.random.seed(seed)
    for _ in range(n_trials):
        params = {k: np.random.choice(v) for k, v in RF_PARAM_GRID.items()}
        model = get_random_forest(task, **params)
        model.fit(x_train, y_train)
        score = compute_val_score(model, x_val, y_val, task)
        if score < best_score:
            best_score, best_params = score, params
    print(f"Best RandomForest params: {best_params} (val score: {best_score:.4f})")
    return best_params

def tune_xrfm(x_train, y_train, x_val, y_val, task, n_trials=10):
    best_score, best_params = np.inf, None
    np.random.seed(seed)
    for _ in range(n_trials):
        params = {k: np.random.choice(v) for k, v in XRFM_PARAM_GRID.items()}
        model = get_xrfm(task, **params)
        model.fit(x_train, y_train, x_val, y_val)
        score = compute_val_score(model, x_val, y_val, task)
        if score < best_score:
            best_score, best_params = score, params
    print(f"    Best xRFM params: {best_params} (val score: {best_score:.4f})")
    return best_params

def _is_xrfm(model):
    return isinstance(model, xRFM)

def _subsample(x, y, n, rng):
    idx = rng.choice(len(x), n, replace=False)
    return x[idx], y[idx]

def train_and_time(model, x_train, y_train, x_val=None, y_val=None):
    rng = np.random.default_rng(42)

    if _is_xrfm(model) and len(x_train) > xRFM_max_samples:
        print(f"  [xRFM] subsampling {len(x_train):,} → {xRFM_max_samples:,} rows (O(n²) memory limit)")
        x_fit, y_fit = _subsample(x_train, y_train, xRFM_max_samples, rng)
        if x_val is not None and len(x_val) > xRFM_max_samples:
            x_vfit, y_vfit = _subsample(x_val, y_val, xRFM_max_samples, rng)
        else:
            x_vfit, y_vfit = x_val, y_val
    else:
        x_fit, y_fit = x_train, y_train
        x_vfit, y_vfit = x_val, y_val

    start = time.perf_counter()
    if x_vfit is not None:
        try:
            from xgboost import XGBClassifier, XGBRegressor
            if isinstance(model, (XGBClassifier, XGBRegressor)):
                model.fit(x_fit, y_fit,
                          eval_set=[(x_vfit, y_vfit)],
                          verbose=False)
            else:
                model.fit(x_fit, y_fit, x_vfit, y_vfit)
        except TypeError:
            model.fit(x_fit, y_fit)
    else:
        model.fit(x_fit, y_fit)

    return model, time.perf_counter() - start

def infer_and_time(model, x_test):
    start = time.perf_counter()
    preds = model.predict(x_test)
    infer_time = (time.perf_counter() - start) / len(x_test)
    return preds, infer_time

def get_xrfm(task="classification", **kwargs):
    return xRFM(task=task, **kwargs)

def get_xgboost(task="classification", **kwargs):
    if task == "classification":
        return XGBClassifier(random_state=42, eval_metric="logloss", **kwargs)
    return XGBRegressor(random_state=42, **kwargs)

def get_random_forest(task="classification", **kwargs):
    if task == "classification":
        return RandomForestClassifier(random_state=42, **kwargs)
    return RandomForestRegressor(random_state=42, **kwargs)
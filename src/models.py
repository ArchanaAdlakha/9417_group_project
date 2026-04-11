import time
import numpy as np
from xrfm import xRFM
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def train_and_time(model, x_train, y_train):
    start = time.perf_counter()
    model.fit(x_train, y_train)
    train_time = time.perf_counter() - start
    return model, train_time

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
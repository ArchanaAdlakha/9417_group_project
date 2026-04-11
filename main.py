from src.datasets import load_diabetes, load_housing, load_wine_quality, load_steel_plates_fault, load_shoppers
from src.models import get_xrfm, get_xgboost, get_random_forest, train_and_time, infer_and_time, tune_xgboost, tune_random_forest, tune_xrfm
from src.evaluate import evaluate
import pandas as pd

datasets = ["diabetes", "housing", "wine", "steel", "shoppers"]
models = {"xRFM": get_xrfm, "XGBoost": get_xgboost, "RandomForest": get_random_forest}

dataset_tasks = {
    "diabetes": "classification",
    "housing": "regression", 
    "wine": "regression",
    "steel": "classification",
    "shoppers": "classification"
}

dataset_loaders = {
    "diabetes": load_diabetes,
    "housing":  load_housing,
    "wine":     load_wine_quality,
    "steel":    load_steel_plates_fault,
    "shoppers": load_shoppers
}

def get_task(dataset_name):
    return dataset_tasks[dataset_name]

results = []
for dataset in datasets:
    task = get_task(dataset)
    data = dataset_loaders[dataset]()

    x_train, x_val, x_test, y_train, y_val, y_test, _ = data
    print(f"  Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")

    tuners = {
        "xRFM":         (tune_xrfm,         get_xrfm),
        "XGBoost":      (tune_xgboost,       get_xgboost),
        "RandomForest": (tune_random_forest, get_random_forest)
    }

    for model_name, (tune_fn, model_fn) in tuners.items():
        print(f"\n  ── {model_name} ──")
        
        try:
            print(f"    Tuning...")
            best_params = tune_fn(x_train, y_train, x_val, y_val, task)

            print(f"    Training final model with best params...")
            model = model_fn(task=task, **best_params)
            model, train_time = train_and_time(model, x_train, y_train, x_val, y_val)

            print(f"    Running inference...")
            preds, infer_time = infer_and_time(model, x_test)

            metrics = evaluate(model, x_test, y_test, task)

            results.append({
                "dataset":               dataset,
                "model":                 model_name,
                "best_params":           str(best_params),
                "train_time_s":          round(train_time, 4),
                "infer_time_per_sample": round(infer_time, 6),
                **metrics
            })
            print(f"    Results: {metrics}")

        except Exception as e:
            print(f"    FAILED: {e}")
            results.append({
                "dataset": dataset,
                "model":   model_name,
                "error":   str(e)
            })
            continue

pd.DataFrame(results).to_csv("results/main_table.csv", index=False)
print("\nDone. Results saved to results/main_table.csv")
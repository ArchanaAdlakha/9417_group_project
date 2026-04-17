from src.datasets import load_diabetes, load_housing, load_wine_quality, load_steel_plates_fault, load_shoppers
from src.models import get_xrfm, get_xgboost, get_random_forest, train_and_time, infer_and_time, tune_xgboost, tune_random_forest, tune_xrfm
from src.scaling import run_scaling_experiment, plot_scaling_results
from src.evaluate import evaluate
from src.interpretability import (
    global_agop_importance,
    extract_leaf_agops,
    compute_pca_importance,
    compute_mutual_info,
    compute_permutation_importance,
    plot_interpretability_comparison,
    compute_rank_correlations,
)
import numpy as np
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

# Will be populated with the first successfully trained xRFM for AGOP testing
_agop_test_state = {"model": None, "feature_names": None, "dataset": None}

results = []
for dataset in datasets:
    task = get_task(dataset)
    data = dataset_loaders[dataset]()

    x_train, x_val, x_test, y_train, y_val, y_test, feature_names = data
    print(f"  Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")

    num_classes = len(np.unique(y_train)) if task == "classification" else 2

    tuners = {
        "xRFM":         (tune_xrfm,         get_xrfm),
        "XGBoost":      (tune_xgboost,       get_xgboost),
        "RandomForest": (tune_random_forest, get_random_forest)
    }

    for model_name, (tune_fn, model_fn) in tuners.items():
        print(f"\n  ── {model_name} ──")

        try:
            print(f"    Tuning...")
            xgb_kwargs = {"num_classes": num_classes} if model_name == "XGBoost" else {}
            best_params = tune_fn(x_train, y_train, x_val, y_val, task, **xgb_kwargs)

            print(f"    Training final model with best params...")
            model = model_fn(task=task, **xgb_kwargs, **best_params)
            model, train_time = train_and_time(model, x_train, y_train, x_val, y_val)

            print(f"    Running inference...")
            preds, infer_time = infer_and_time(model, x_test)

            metrics = evaluate(model, x_test, y_test, task)

            # Capture first trained xRFM for AGOP verification later
            if model_name == "xRFM" and _agop_test_state["model"] is None:
                _agop_test_state["model"]         = model
                _agop_test_state["feature_names"] = feature_names
                _agop_test_state["dataset"]       = dataset

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


# ---------------------------------------------------------------------------
# AGOP extraction verification — uses the first successfully trained xRFM
# ---------------------------------------------------------------------------
if _agop_test_state["model"] is not None:
    _test_model    = _agop_test_state["model"]
    _test_features = _agop_test_state["feature_names"]
    _test_dataset  = _agop_test_state["dataset"]

    print(f"\n{'='*60}")
    print(f"AGOP extraction check — {_test_dataset}")
    print(f"{'='*60}")

    _leaf_agops = extract_leaf_agops(_test_model)
    _n_leaves   = len(_leaf_agops)

    # ── Debug: inspect raw per-leaf AGOP diagonals ──────────────────────────
    print(f"\n  [DEBUG] extract_leaf_agops returned {_n_leaves} array(s)")
    for _li, _la in enumerate(_leaf_agops):
        print(f"  [DEBUG] leaf {_li}: shape={_la.shape}  dtype={_la.dtype}"
              f"  min={_la.min():.6f}  max={_la.max():.6f}  mean={_la.mean():.6f}")
        print(f"          first 10 values: {_la[:10]}")
        if _la.max() == _la.min():
            print(f"  [DEBUG] WARNING — leaf {_li} diagonal is constant "
                  f"({_la[0]:.6f}); normalise() will zero it out")
    # ────────────────────────────────────────────────────────────────────────

    if _n_leaves == 0:
        print("  WARNING: no leaf AGOP matrices found (M may still be None after fit).")
    else:
        _sample_shape = _leaf_agops[0].shape   # (d,) — always 1-D after extraction
        _global_imp   = global_agop_importance(_test_model)

        # Top-10 features by AGOP importance
        _top_k   = min(10, len(_test_features))
        _top_idx = np.argsort(_global_imp)[::-1][:_top_k]

        print(f"\n  Found {_n_leaves} leaf/leaves")
        print(f"  Per-leaf AGOP diagonal shape : {_sample_shape}")
        print(f"  Global AGOP vector shape     : {_global_imp.shape}")
        print(f"  Global AGOP — min={_global_imp.min():.6f}  "
              f"max={_global_imp.max():.6f}  mean={_global_imp.mean():.6f}")
        print(f"\n  Top {_top_k} features by AGOP importance:")
        print(f"  {'Rank':<6} {'Feature':<30} {'Score':>10}")
        print(f"  {'-'*48}")
        for rank, idx in enumerate(_top_idx, 1):
            fname = _test_features[idx] if _test_features is not None else f"feature_{idx}"
            print(f"  {rank:<6} {str(fname):<30} {_global_imp[idx]:>10.6f}")
else:
    print("\nWARNING: no xRFM model was trained successfully — skipping AGOP check.")


def run_interpretability(
    trained_xrfm,
    trained_xgb,
    X_train,
    X_val,
    y_train,
    y_val,
    feature_names,
    task,
    dataset_name,
):
    """
    Compute and compare feature importances across methods for one dataset.

    Saves a grouped bar chart to results/{dataset_name}_interpretability.png
    and prints Spearman rank correlations between every pair of methods.

    Parameters
    ----------
    trained_xrfm : fitted xRFM model
    trained_xgb  : fitted XGBoost model (sklearn-compatible)
    X_train, X_val : np.ndarray
    y_train, y_val : np.ndarray
    feature_names  : list[str]
    task           : 'regression' or 'classification'
    dataset_name   : str  – used to name the output file
    """
    print(f"\n{'='*60}")
    print(f"Interpretability analysis — {dataset_name}")
    print(f"{'='*60}")

    print("  Computing AGOP importance (xRFM)...")
    agop = global_agop_importance(trained_xrfm)

    print("  Computing PCA importance...")
    pca = compute_pca_importance(X_train)

    print("  Computing mutual information...")
    mi = compute_mutual_info(X_train, y_train, task)

    print("  Computing permutation importance (XGBoost on val set)...")
    perm = compute_permutation_importance(trained_xgb, X_val, y_val, task)

    importances_dict = {
        "AGOP":        agop,
        "PCA":         pca,
        "MI":          mi,
        "Permutation": perm,
    }

    save_path = f"results/{dataset_name}_interpretability.png"
    print(f"  Saving plot → {save_path}")
    plot_interpretability_comparison(importances_dict, feature_names, save_path)

    print("  Spearman rank correlations:")
    compute_rank_correlations(importances_dict)


# ---------------------------------------------------------------------------
# Run interpretability once — using the diabetes dataset as the showcase
# ---------------------------------------------------------------------------
_interp_dataset = "diabetes"
_interp_task    = get_task(_interp_dataset)
_interp_data    = dataset_loaders[_interp_dataset]()
_ix_train, _ix_val, _ix_test, _iy_train, _iy_val, _iy_test, _ifeat = _interp_data

print(f"\nFitting models for interpretability analysis ({_interp_dataset})...")
_xrfm_interp = get_xrfm(task=_interp_task)
_xrfm_interp, _t = train_and_time(_xrfm_interp, _ix_train, _iy_train, _ix_val, _iy_val)

_xgb_interp = get_xgboost(task=_interp_task)
_xgb_interp, _t = train_and_time(_xgb_interp, _ix_train, _iy_train, _ix_val, _iy_val)

run_interpretability(
    trained_xrfm  = _xrfm_interp,
    trained_xgb   = _xgb_interp,
    X_train       = _ix_train,
    X_val         = _ix_val,
    y_train       = _iy_train,
    y_val         = _iy_val,
    feature_names = _ifeat,
    task          = _interp_task,
    dataset_name  = _interp_dataset,
)


# ---------------------------------------------------------------------------
# Scaling experiment — largest dataset (diabetes, ~9 000 training rows)
# ---------------------------------------------------------------------------

def run_scaling(dataset_name, X_train, y_train, X_val, y_val, X_test, y_test, task):
    """
    Train xRFM, XGBoost, and Random Forest on subsamples of increasing size,
    always evaluating on the same full test set, then save the two result plots.
    """
    print(f"\n{'='*60}")
    print(f"Scaling experiment — {dataset_name}  (task={task})")
    print(f"Full train size: {len(X_train):,}  |  test size: {len(X_test):,}")
    print(f"{'='*60}")

    metric_name = "RMSE" if task == "regression" else "Accuracy"
    tuning_metric = "mse" if task == "regression" else "accuracy"

    model_factories = {
        "xRFM": lambda: get_xrfm(task=task, tuning_metric=tuning_metric),
        "XGBoost": lambda: get_xgboost(task=task),
        "RF": lambda: get_random_forest(task=task, n_estimators=200),
    }

    all_results = {}
    for name, factory in model_factories.items():
        print(f"\n── {name} ──")
        records = run_scaling_experiment(
            model_name  = name,
            get_model_fn= factory,
            X_train     = X_train,
            y_train     = y_train,
            X_val       = X_val,
            y_val       = y_val,
            X_test      = X_test,
            y_test      = y_test,
            task        = task,
        )
        all_results[name] = records

    print(f"\nSaving scaling plots...")
    plot_scaling_results(all_results, metric_name, dataset_name)


# Use diabetes — the largest dataset (~9 000 training rows after split)
_scale_dataset = "diabetes"
_scale_task    = get_task(_scale_dataset)
_scale_data    = dataset_loaders[_scale_dataset]()
_sx_train, _sx_val, _sx_test, _sy_train, _sy_val, _sy_test, _ = _scale_data

run_scaling(
    dataset_name = _scale_dataset,
    X_train      = _sx_train,
    y_train      = _sy_train,
    X_val        = _sx_val,
    y_val        = _sy_val,
    X_test       = _sx_test,
    y_test       = _sy_test,
    task         = _scale_task,
)
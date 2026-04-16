import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, accuracy_score

from src.models import train_and_time


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_numpy(arr):
    """Convert a torch Tensor or any array-like to a numpy array."""
    if hasattr(arr, "detach"):          # torch.Tensor
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def _evaluate(model, X_test, y_test, task):
    """Return a single scalar metric: RMSE for regression, accuracy for classification."""
    preds = _to_numpy(model.predict(X_test))
    if task == "regression":
        return float(root_mean_squared_error(y_test, preds))
    else:
        # xRFM may return floats; round to nearest int for accuracy
        preds = np.round(preds).astype(int)
        return float(accuracy_score(y_test, preds))


# ---------------------------------------------------------------------------
# Core scaling experiment
# ---------------------------------------------------------------------------

def run_scaling_experiment(
    model_name,
    get_model_fn,
    X_train, y_train,
    X_val,   y_val,
    X_test,  y_test,
    task,
):
    """
    Train a model on subsamples of increasing size and record test performance
    and training time at each size.

    Parameters
    ----------
    model_name : str
        Label used in progress messages (e.g. 'xRFM').
    get_model_fn : callable
        Zero-argument callable that returns a fresh, unfitted model instance.
    X_train, y_train : np.ndarray  —  full training set
    X_val,   y_val   : np.ndarray  —  validation set (passed to xRFM / XGBoost)
    X_test,  y_test  : np.ndarray  —  held-out test set (NEVER subsampled)
    task : str        —  'regression' or 'classification'

    Returns
    -------
    list of dict  with keys 'n', 'metric', 'time'
    """
    n_full = len(X_train)
    sizes  = [s for s in [500, 1000, 2000, 5000, n_full] if s <= n_full]
    # de-duplicate while preserving order (n_full may equal 5000)
    seen, unique_sizes = set(), []
    for s in sizes:
        if s not in seen:
            seen.add(s)
            unique_sizes.append(s)

    rng     = np.random.RandomState(42)
    records = []

    for n in unique_sizes:
        print(f"  Training {model_name} on {n:,} samples...", flush=True)

        # Subsample training data reproducibly
        idx     = rng.choice(n_full, size=n, replace=False)
        X_sub   = X_train[idx]
        y_sub   = y_train[idx]

        # Fresh model each time
        model = get_model_fn()

        # train_and_time handles xRFM val-data passing and XGBoost eval_set
        model, elapsed = train_and_time(model, X_sub, y_sub, X_val, y_val)

        metric = _evaluate(model, X_test, y_test, task)
        metric_label = "RMSE" if task == "regression" else "Accuracy"
        print(f"    n={n:,}  {metric_label}={metric:.4f}  time={elapsed:.2f}s")

        records.append({"n": n, "metric": metric, "time": elapsed})

    return records


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_scaling_results(all_results, metric_name, dataset_name, save_dir="results/"):
    """
    Save two PNG plots from scaling experiment results.

    Parameters
    ----------
    all_results : dict[str, list[dict]]
        Keys are model names; values are lists of {'n', 'metric', 'time'} dicts.
    metric_name : str
        Y-axis label for the performance plot (e.g. 'RMSE' or 'Accuracy').
    dataset_name : str
        Used to form output file names.
    save_dir : str
        Directory where PNGs are saved (must already exist).
    """
    os.makedirs(save_dir, exist_ok=True)

    # ── Performance plot ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    for model_name, records in all_results.items():
        ns      = [r["n"]      for r in records]
        metrics = [r["metric"] for r in records]
        ax.plot(ns, metrics, "o-", label=model_name)

    ax.set_xlabel("Training set size (n)")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{dataset_name} — {metric_name} vs training set size")
    ax.legend()
    fig.tight_layout()
    perf_path = os.path.join(save_dir, f"{dataset_name}_performance_vs_n.png")
    fig.savefig(perf_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {perf_path}")

    # ── Training-time plot ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    for model_name, records in all_results.items():
        ns    = [r["n"]    for r in records]
        times = [r["time"] for r in records]
        ax.plot(ns, times, "o-", label=model_name)

    ax.set_xlabel("Training set size (n)")
    ax.set_ylabel("Training time (seconds)")
    ax.set_title(f"{dataset_name} — training time vs training set size")
    ax.legend()
    fig.tight_layout()
    time_path = os.path.join(save_dir, f"{dataset_name}_time_vs_n.png")
    fig.savefig(time_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {time_path}")

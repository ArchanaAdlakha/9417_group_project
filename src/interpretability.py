import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr


def _collect_leaf_rfms(tree):
    """Recursively walk an xRFM tree dict and yield leaf RFM model objects."""
    if tree["type"] == "leaf":
        yield tree["model"]
    else:
        yield from _collect_leaf_rfms(tree["left"])
        yield from _collect_leaf_rfms(tree["right"])


def extract_leaf_agops(model):
    """
    Extract the AGOP diagonal (per-feature importance) from every leaf of a
    fitted xRFM model.

    The xRFM tree structure is stored in model.trees (list of tree dicts).
    Each leaf dict has a 'model' key holding a trained RFM instance whose
    attribute M is the learned Mahalanobis / AGOP matrix:
      - shape (d,)   when the RFM was constructed with diag=True
      - shape (d, d) when diag=False (the default full matrix)

    Parameters
    ----------
    model : xrfm.xRFM
        A fitted xRFM model.

    Returns
    -------
    list of np.ndarray
        One 1-D array of length d per leaf, containing the diagonal of M
        (i.e. per-feature importance scores).
    """
    # model.trees is None before fit() is called
    if not model.trees:
        return []

    diagonals = []
    for tree in model.trees:
        # Note: when the dataset is smaller than max_leaf_size (default 60 000)
        # xRFM builds a single-leaf tree.  The tree dict still has
        # type='leaf' and a 'model' key — _collect_leaf_rfms handles it fine.
        for rfm in _collect_leaf_rfms(tree):
            # Prefer agop_best_model over M.
            #
            # rfm.M is the Mahalanobis matrix that was plugged into the kernel
            # at the *best* iteration — it equals the AGOP from the iteration
            # *before* the best one, and can be a scaled identity when iteration
            # 0 is selected, making every diagonal entry identical (→ normalise
            # collapses them all to 0).
            #
            # rfm.agop_best_model is computed by xRFM after restoring the best
            # weights via  fit_M(X_train, best_weights, inplace=False), so it
            # reflects genuine per-feature gradient magnitudes of the final model.
            M = getattr(rfm, "agop_best_model", None)
            if M is None:
                M = rfm.M          # fall back to Mahalanobis M
            if M is None:
                continue

            # Convert to numpy (M may be a torch.Tensor on CPU or GPU)
            if hasattr(M, "detach"):
                M = M.detach().cpu().numpy()
            else:
                M = np.asarray(M)

            if M.ndim == 1:
                # diag=True: M is already the diagonal vector
                diagonals.append(M)
            else:
                # diag=False (default): M is a full d×d matrix
                diagonals.append(np.diag(M))

    return diagonals


def global_agop_importance(model):
    """
    Compute a single global per-feature importance vector by averaging the
    AGOP diagonal across all leaves of a fitted xRFM model.

    Parameters
    ----------
    model : xrfm.xRFM
        A fitted xRFM model.

    Returns
    -------
    np.ndarray of shape (d,)
        Mean AGOP diagonal across all leaves.  Higher values indicate
        features that the model weighted more heavily overall.
    """
    leaf_agops = extract_leaf_agops(model)
    if not leaf_agops:
        raise ValueError("No leaf AGOP matrices found — is the model fitted?")
    return np.mean(np.stack(leaf_agops, axis=0), axis=0)


# ---------------------------------------------------------------------------
# Baseline importance methods
# ---------------------------------------------------------------------------

def compute_pca_importance(X_train):
    """
    Per-feature importance from PCA: sum of squared loadings weighted by
    each component's explained variance ratio.

    Parameters
    ----------
    X_train : array-like of shape (n_samples, n_features)

    Returns
    -------
    np.ndarray of shape (n_features,)
    """
    X = np.asarray(X_train)
    pca = PCA().fit(X)
    # components_ shape: (n_components, n_features)
    importance = np.sum(pca.explained_variance_ratio_[:, None] * pca.components_ ** 2, axis=0)
    return importance


def compute_mutual_info(X_train, y_train, task):
    """
    Per-feature mutual information with the target.

    Parameters
    ----------
    X_train : array-like of shape (n_samples, n_features)
    y_train : array-like of shape (n_samples,)
    task : str
        'regression' or 'classification'

    Returns
    -------
    np.ndarray of shape (n_features,)
    """
    X = np.asarray(X_train)
    y = np.asarray(y_train)
    if task == "regression":
        return mutual_info_regression(X, y, random_state=42)
    return mutual_info_classif(X, y, random_state=42)


def compute_permutation_importance(model, X_val, y_val, task):
    """
    Per-feature permutation importance evaluated on the validation set.

    Parameters
    ----------
    model : fitted sklearn-compatible model
    X_val : array-like of shape (n_samples, n_features)
    y_val : array-like of shape (n_samples,)
    task : str
        'regression' or 'classification'

    Returns
    -------
    np.ndarray of shape (n_features,)
        Mean importance across the n_repeats permutations.
    """
    scoring = "neg_root_mean_squared_error" if task == "regression" else "accuracy"
    result = permutation_importance(
        model, X_val, y_val,
        n_repeats=10,
        random_state=42,
        scoring=scoring,
    )
    return result.importances_mean


# ---------------------------------------------------------------------------
# Normalisation helper
# ---------------------------------------------------------------------------

def normalise(arr):
    """
    Min-max normalise *arr* to [0, 1].  Returns a zero array if all values
    are identical (avoids division by zero).

    Parameters
    ----------
    arr : np.ndarray

    Returns
    -------
    np.ndarray
    """
    arr = np.asarray(arr, dtype=float)
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_interpretability_comparison(importances_dict, feature_names, save_path):
    """
    Grouped bar chart comparing per-feature importance across methods.

    Each method's scores are normalised to [0, 1] before plotting.
    If there are more than 15 features, only the top 15 by AGOP importance
    are shown.

    Parameters
    ----------
    importances_dict : dict[str, np.ndarray]
        Keys are method names (e.g. 'AGOP', 'PCA', 'MI', 'Permutation'),
        values are 1-D importance arrays of length n_features.
    feature_names : list[str]
        Feature names of length n_features.
    save_path : str
        File path to save the figure (PNG recommended).
    """
    feature_names = list(feature_names)
    methods = list(importances_dict.keys())

    # Normalise each method's scores
    normed = {m: normalise(v) for m, v in importances_dict.items()}

    # Optionally restrict to top 15 features by AGOP
    if len(feature_names) > 15:
        agop_key = next((k for k in methods if "agop" in k.lower()), methods[0])
        top_idx = np.argsort(normed[agop_key])[-15:][::-1]
    else:
        top_idx = np.arange(len(feature_names))

    shown_features = [feature_names[i] for i in top_idx]
    normed_subset = {m: normed[m][top_idx] for m in methods}

    x = np.arange(len(shown_features))
    bar_width = 0.8 / len(methods)

    fig, ax = plt.subplots(figsize=(max(10, len(shown_features) * 0.7), 5))
    for i, method in enumerate(methods):
        offset = (i - len(methods) / 2 + 0.5) * bar_width
        ax.bar(x + offset, normed_subset[method], bar_width, label=method)

    ax.set_xticks(x)
    ax.set_xticklabels(shown_features, rotation=45, ha="right")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Normalised importance [0, 1]")
    ax.set_title("Interpretability comparison across methods")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Rank correlations
# ---------------------------------------------------------------------------

def compute_rank_correlations(importances_dict):
    """
    Spearman rank correlation between every pair of importance methods.

    Parameters
    ----------
    importances_dict : dict[str, np.ndarray]

    Returns
    -------
    dict[str, dict]
        Keys are 'A_vs_B' strings; values are {'corr': float, 'pval': float}.
    """
    methods = list(importances_dict.keys())
    results = {}
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            a, b = methods[i], methods[j]
            corr, pval = spearmanr(importances_dict[a], importances_dict[b])
            key = f"{a}_vs_{b}"
            results[key] = {"corr": float(corr), "pval": float(pval)}
            print(f"  {key:40s}  corr={corr:+.3f}  p={pval:.3e}")
    return results

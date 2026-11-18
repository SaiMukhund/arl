import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

def empirical_cvar(costs, alpha=0.05):
    costs = np.asarray(costs, dtype=float)
    if len(costs) == 0:
        return 0.0
    # For discrete costs with many zeros, threshold-based selection can include
    # far more than alpha of the samples. Use exactly the largest ceil(alpha*n).
    k = max(1, int(np.ceil(alpha * len(costs))))
    tail = np.partition(costs, len(costs) - k)[-k:]
    return float(tail.mean())

def safe_auc(y, p):
    try:
        return float(roc_auc_score(y, p))
    except ValueError:
        return float("nan")

def calibration_metrics(y, p):
    p = np.clip(np.asarray(p), 1e-6, 1-1e-6)
    y = np.asarray(y)
    return {
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p)),
        "auc": safe_auc(y, p),
        "mean_predicted_success": float(p.mean()),
        "observed_success": float(y.mean())
    }

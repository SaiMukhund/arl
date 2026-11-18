import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

class RiskModels:
    def __init__(self, quantiles=(0.50, 0.90, 0.95, 0.99)):
        self.success = HistGradientBoostingClassifier(max_iter=180, max_leaf_nodes=20, l2_regularization=1.0)
        self.cost = HistGradientBoostingRegressor(max_iter=180, max_leaf_nodes=20, l2_regularization=1.0)
        self.cat = HistGradientBoostingClassifier(max_iter=180, max_leaf_nodes=20, l2_regularization=1.0)
        self.quantiles = tuple(quantiles)
        self.qmodels = {
            q: GradientBoostingRegressor(loss="quantile", alpha=q, n_estimators=160,
                                         max_depth=2, learning_rate=0.04, random_state=17)
            for q in self.quantiles
        }
        self.success_calibrator = None

    @staticmethod
    def features(X):
        # Include action ID and mild nonlinear/context interactions via tree models.
        return np.asarray(X, dtype=float)

    def fit(self, X, success, cost, catastrophic):
        X = self.features(X)
        self.success.fit(X, success)
        self.cost.fit(X, np.log1p(cost))
        self.cat.fit(X, catastrophic)
        for q, model in self.qmodels.items():
            model.fit(X, np.log1p(cost))
        return self

    def calibrate_success(self, X_val, y_val):
        p = self.success.predict_proba(self.features(X_val))[:, 1]
        # Logistic calibration on logit scores.
        eps = 1e-6
        z = np.log(np.clip(p, eps, 1-eps) / np.clip(1-p, eps, 1-eps)).reshape(-1, 1)
        self.success_calibrator = LogisticRegression().fit(z, y_val)
        return self

    def predict_success(self, X):
        p = self.success.predict_proba(self.features(X))[:, 1]
        if self.success_calibrator is not None:
            eps = 1e-6
            z = np.log(np.clip(p, eps, 1-eps) / np.clip(1-p, eps, 1-eps)).reshape(-1, 1)
            p = self.success_calibrator.predict_proba(z)[:, 1]
        return np.clip(p, 1e-5, 1-1e-5)

    def predict_cost(self, X):
        return np.expm1(self.cost.predict(self.features(X))).clip(0)

    def predict_catastrophic(self, X):
        return self.cat.predict_proba(self.features(X))[:, 1]

    def predict_cvar(self, X, alpha=0.05):
        # CVaR approximation from predicted quantiles.
        qs = sorted(self.qmodels)
        preds = np.column_stack([np.expm1(self.qmodels[q].predict(self.features(X))).clip(0) for q in qs])
        # Average predicted quantiles in the upper alpha tail; interpolate if needed.
        tail_start = 1.0 - alpha
        selected = [j for j, q in enumerate(qs) if q >= tail_start]
        if not selected:
            selected = [len(qs)-1]
        return preds[:, selected].mean(axis=1)

def pinball_loss(y, pred, q):
    err = y - pred
    return np.mean(np.maximum(q*err, (q-1)*err))

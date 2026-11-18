import numpy as np

def upper_cvar(costs, alpha=0.05):
    x = np.asarray(costs, dtype=np.float64)
    if x.size == 0: return 0.0
    k = max(1, int(np.ceil(alpha*x.size)))
    return float(np.mean(np.partition(x, x.size-k)[-k:]))

def entropic_risk(costs, theta=2.0, cost_scale=100.0):
    """Stable entropic risk: cost_scale/theta * log E[exp(theta*C/cost_scale)]."""
    x = np.asarray(costs, dtype=np.float64)
    if x.size == 0: return 0.0
    z = theta*x/cost_scale
    m = np.max(z)
    return float(cost_scale*(m + np.log(np.mean(np.exp(z-m))))/theta)

def entropic_premium(costs, theta=2.0, cost_scale=100.0):
    x = np.asarray(costs, dtype=np.float64)
    if x.size == 0: return 0.0
    return max(0.0, entropic_risk(x, theta, cost_scale) - float(x.mean()))

def score_actions(success, mean_cost, cvar, premium, cfg, method):
    base = np.asarray(success) - cfg.lambda_cost*np.asarray(mean_cost)/cfg.cost_scale
    if method == "expected_cost": return base
    if method == "cvar": return base - cfg.beta_cvar*np.asarray(cvar)/cfg.cost_scale
    if method == "entropic": return base - cfg.gamma_entropic*np.asarray(premium)/cfg.cost_scale
    if method == "cvar_entropic":
        return (base - cfg.beta_cvar*np.asarray(cvar)/cfg.cost_scale
                - cfg.gamma_entropic*np.asarray(premium)/cfg.cost_scale)
    if method == "safe_constraint":
        return base - cfg.beta_cvar*np.asarray(cvar)/cfg.cost_scale
    raise ValueError(method)

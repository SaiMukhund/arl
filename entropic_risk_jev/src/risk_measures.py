"""Risk measures for evaluation and for risk-greedy distributional control.

All measures follow the package convention (rewards: higher = better), so a
*higher* value of every measure is better, and a risk-averse agent maximises it.

Two flavours:
* sample-based  (``*_samples``)   -- for Monte-Carlo evaluation of returns.
* categorical   (``*_categorical``)-- for distributional DP over a fixed support.
"""

from __future__ import annotations

import numpy as np

from .risk import stable_logsumexp


# --------------------------------------------------------------- sample based
def mean_samples(returns: np.ndarray) -> float:
    return float(np.mean(returns))


def entropic_samples(returns: np.ndarray, beta: float) -> float:
    returns = np.asarray(returns, dtype=np.float64)
    if abs(beta) < 1e-12:
        return float(np.mean(returns))
    n = returns.shape[0]
    lme = stable_logsumexp(-beta * returns, axis=0) - np.log(n)
    return float(-(1.0 / beta) * lme)


def cvar_samples(returns: np.ndarray, alpha: float = 0.1) -> float:
    """CVaR_alpha of returns = mean of the worst (lowest) alpha-fraction.

    Higher is better (a policy with a less severe lower tail has higher CVaR).
    """
    returns = np.sort(np.asarray(returns, dtype=np.float64))
    k = max(1, int(np.ceil(alpha * returns.shape[0])))
    return float(np.mean(returns[:k]))


# ---------------------------------------------------------------- categorical
def mean_categorical(z: np.ndarray, p: np.ndarray) -> float:
    return float(np.sum(z * p))


def entropic_categorical(z: np.ndarray, p: np.ndarray, beta: float) -> float:
    if abs(beta) < 1e-12:
        return float(np.sum(z * p))
    # -(1/beta) log sum_i p_i exp(-beta z_i), computed stably.
    with np.errstate(divide="ignore"):
        logp = np.log(p)
    lse = stable_logsumexp(logp - beta * z, axis=0)
    return float(-(1.0 / beta) * lse)


def cvar_categorical(z: np.ndarray, p: np.ndarray, alpha: float = 0.1) -> float:
    """CVaR_alpha of a categorical distribution (z ascending). Higher = better."""
    order = np.argsort(z)
    z_s, p_s = z[order], p[order]
    acc, mass = 0.0, 0.0
    for zi, pi in zip(z_s, p_s):
        if pi <= 0:
            continue                 # skip empty atoms (do NOT break)
        take = min(pi, alpha - mass)
        acc += take * zi
        mass += take
        if mass >= alpha:
            break
    return float(acc / alpha) if mass > 0 else float(z_s[0])


def make_risk_fn(kind: str, beta: float = 0.15, alpha: float = 0.1):
    """Return a categorical risk functional rho(z, p) for distributional control."""
    if kind == "neutral":
        return lambda z, p: mean_categorical(z, p)
    if kind == "entropic":
        return lambda z, p: entropic_categorical(z, p, beta)
    if kind == "cvar":
        return lambda z, p: cvar_categorical(z, p, alpha)
    raise ValueError(f"unknown risk kind: {kind}")

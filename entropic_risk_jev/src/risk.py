"""Entropic (exponential-utility) risk measures and the risk-sensitive Bellman backup.

Convention used throughout this package
---------------------------------------
Rewards are "higher = better" (a catastrophe is a large *negative* reward).

The entropic risk measure (certainty equivalent under exponential utility) of a
random reward ``R`` with risk parameter ``beta`` is

    U_beta(R) = -(1/beta) * log E[ exp(-beta * R) ]

Properties (see e.g. Howard & Matheson 1972; Follmer & Schied 2011):

* ``beta -> 0``  =>  U_beta(R) -> E[R]                 (risk neutral)
* ``beta  > 0``  =>  risk averse. Taylor: U_beta ~= E[R] - (beta/2) Var(R).
                     Low-reward (catastrophic) tails are amplified by exp(-beta*R),
                     so rare but costly failures dominate the value.
* ``beta  < 0``  =>  risk seeking.

Because ``x -> -(1/beta) log E[exp(-beta x)]`` is *monotone increasing* in the
underlying reward, maximising U_beta is a well-defined control objective and the
greedy (argmax) operator is unchanged -- only the backed-up value changes.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "stable_logsumexp",
    "entropic_value",
    "entropic_backup",
]


def stable_logsumexp(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable ``log sum exp`` that tolerates ``-inf`` entries.

    Entries equal to ``-inf`` (e.g. ``log`` of a zero transition probability)
    contribute zero to the sum and are handled without producing ``nan``.
    """
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x, axis=axis, keepdims=True)
    # If an entire slice is -inf, m is -inf; replace with 0 so the shift is a
    # no-op and the result stays -inf.
    m_safe = np.where(np.isfinite(m), m, 0.0)
    shifted = x - m_safe
    summed = np.sum(np.exp(shifted), axis=axis, keepdims=True)
    out = np.log(summed) + m_safe
    return np.squeeze(out, axis=axis)


def entropic_value(returns: np.ndarray, beta: float) -> float:
    """Entropic risk certainty-equivalent of a sample of returns.

    ``U_beta(R) = -(1/beta) log mean_i exp(-beta * R_i)``.
    For ``beta == 0`` this is the sample mean.
    """
    returns = np.asarray(returns, dtype=np.float64)
    if abs(beta) < 1e-12:
        return float(np.mean(returns))
    n = returns.shape[0]
    lme = stable_logsumexp(-beta * returns, axis=0) - np.log(n)
    return float(-(1.0 / beta) * lme)


def entropic_backup(
    P: np.ndarray, R: np.ndarray, V: np.ndarray, beta: float
) -> np.ndarray:
    """One risk-sensitive Bellman backup, returning Q[s, a].

    Implements the entropic risk-sensitive Bellman operator

        Q_beta(s, a) = -(1/beta) log sum_{s'} P(s'|s,a)
                           exp( -beta * ( R(s,a,s') + V(s') ) )

    which, for ``beta -> 0``, reduces to the ordinary expected-value backup
    ``Q(s, a) = sum_{s'} P(s'|s,a) ( R(s,a,s') + V(s') )``.

    Parameters
    ----------
    P : array (S, A, S)   transition probabilities P(s'|s,a)
    R : array (S, A, S)   rewards R(s, a, s')  (higher is better)
    V : array (S,)        current state values
    beta : float          risk aversion (>0 averse, 0 neutral, <0 seeking)
    """
    P = np.asarray(P, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    if abs(beta) < 1e-12:
        # Risk-neutral expected-value backup.
        return np.einsum("sap,sap->sa", P, R + V[None, None, :])

    # exponent[s, a, s'] = log P(s'|s,a) - beta * (R + V(s'))
    with np.errstate(divide="ignore"):
        logP = np.log(P)
    exponent = logP + (-beta) * (R + V[None, None, :])
    lse = stable_logsumexp(exponent, axis=2)  # (S, A)
    return -(1.0 / beta) * lse

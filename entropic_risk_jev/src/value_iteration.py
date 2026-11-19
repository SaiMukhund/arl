"""Exact risk-sensitive value iteration on the tabular RiskyCheckout MDP.

This is the ground truth for the project: because the MDP is small and fully
known, we can solve the entropic risk-sensitive control problem exactly and read
off how the optimal policy changes with the risk parameter ``beta``.  No sampling
noise, no function approximation -- it isolates the *effect* of entropic risk.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .env import RiskyCheckout
from .risk import entropic_backup


@dataclass
class VISolution:
    beta: float
    V: np.ndarray            # (S,)
    Q: np.ndarray            # (S, A)
    policy: np.ndarray       # (S,) greedy action per state
    iterations: int


def risk_sensitive_value_iteration(
    env: RiskyCheckout,
    beta: float,
    tol: float = 1e-10,
    max_iter: int = 10_000,
) -> VISolution:
    """Solve ``V*(s) = max_a Q_beta(s, a)`` under the entropic Bellman operator."""
    S = env.n_states
    V = np.zeros(S, dtype=np.float64)
    terminals = np.array(env.terminal_states)

    iters = 0
    for iters in range(1, max_iter + 1):
        Q = entropic_backup(env.P, env.R, V, beta)
        V_new = Q.max(axis=1)
        V_new[terminals] = 0.0  # absorbing terminals carry no future value
        delta = np.max(np.abs(V_new - V))
        V = V_new
        if delta < tol:
            break

    Q = entropic_backup(env.P, env.R, V, beta)
    policy = Q.argmax(axis=1)
    return VISolution(beta=beta, V=V, Q=Q, policy=policy, iterations=iters)


def entropic_policy_value(
    env: RiskyCheckout,
    policy: np.ndarray,
    beta: float,
    tol: float = 1e-10,
    max_iter: int = 10_000,
) -> float:
    """Exact entropic risk value U_beta of a fixed deterministic policy at start.

    Risk-sensitive policy evaluation:
        V(s) = -(1/beta) log sum_{s'} P(s'|s,pi(s)) exp(-beta (R + V(s')))
    Returns V(start_state). Exact (no Monte-Carlo noise), using the known model.
    """
    S = env.n_states
    V = np.zeros(S, dtype=np.float64)
    terminals = np.array(env.terminal_states)
    for _ in range(max_iter):
        Q = entropic_backup(env.P, env.R, V, beta)       # (S, A)
        V_new = Q[np.arange(S), policy]
        V_new[terminals] = 0.0
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    return float(V[env.start_state])

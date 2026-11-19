"""Baseline policies to compare against risk-sensitive learned value functions.

All policies are arrays of shape (n_states,) giving an action per state, so they
plug straight into ``rollout.policy_statistics``.
"""

from __future__ import annotations

import numpy as np

from .env import ASK, FAST, SAFE


def fixed_action_policy(env, action: int) -> np.ndarray:
    """Always take `action` (e.g. always FAST / always SAFE / always ASK)."""
    return np.full(env.n_states, action, dtype=int)


def random_policy(env, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, env.n_actions, size=env.n_states)


def risk_neutral_policy(env) -> np.ndarray:
    """Ordinary (risk-neutral) Jev value learning = exact VI at beta=0."""
    from .value_iteration import risk_sensitive_value_iteration

    return risk_sensitive_value_iteration(env, 0.0).policy


def oracle_entropic_policy(env, beta: float) -> np.ndarray:
    """Ground-truth risk-sensitive optimal policy (exact entropic VI)."""
    from .value_iteration import risk_sensitive_value_iteration

    return risk_sensitive_value_iteration(env, beta).policy


def stock_yev_policy(yev_backbone) -> np.ndarray:
    """Zero-shot policy from stock yev0-4b's calibrated option distribution.

    yev0-4b is used untrained: for each state we read its calibrated distribution
    over the three options (fast/safe/ask) and take the argmax. This is the
    'vanilla Jev decision model' baseline -- risk-neutral by construction.
    """
    env = yev_backbone.env
    policy = np.zeros(env.n_states, dtype=int)
    for s in range(env.n_states):
        if env.is_terminal(s):
            continue
        probs = yev_backbone.jev_success_probs(s)
        policy[s] = int(np.argmax(probs.cpu().numpy()))
    return policy


ACTION_LOOKUP = {"FAST": FAST, "SAFE": SAFE, "ASK": ASK}

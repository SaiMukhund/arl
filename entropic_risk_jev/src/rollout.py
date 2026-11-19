"""Monte-Carlo rollout utilities and policy outcome statistics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .env import RiskyCheckout
from .risk import entropic_value


@dataclass
class PolicyStats:
    mean_return: float
    std_return: float
    success_rate: float
    catastrophe_rate: float
    cvar_5: float            # mean of worst 5% of returns (tail risk)
    worst_return: float
    entropic_values: dict    # beta -> certainty-equivalent return

    def as_row(self) -> dict:
        return {
            "mean_return": self.mean_return,
            "std_return": self.std_return,
            "success_rate": self.success_rate,
            "catastrophe_rate": self.catastrophe_rate,
            "cvar_5": self.cvar_5,
            "worst_return": self.worst_return,
        }


def rollout_policy(
    env: RiskyCheckout,
    policy: np.ndarray,
    n_episodes: int = 20_000,
    max_steps: int = 100,
    seed: int = 0,
) -> np.ndarray:
    """Return an array of episode returns under a (deterministic) policy."""
    rng = np.random.default_rng(seed)
    returns = np.empty(n_episodes, dtype=np.float64)
    for ep in range(n_episodes):
        s = env.start_state
        total = 0.0
        for _ in range(max_steps):
            a = int(policy[s])
            s, r, done = env.step(s, a, rng)
            total += r
            if done:
                break
        returns[ep] = total
    return returns


def policy_statistics(
    env: RiskyCheckout,
    policy: np.ndarray,
    n_episodes: int = 20_000,
    seed: int = 0,
    betas: tuple = (0.0, 0.1, 0.4),
) -> PolicyStats:
    rng = np.random.default_rng(seed)
    returns = np.empty(n_episodes, dtype=np.float64)
    successes = 0
    catastrophes = 0
    for ep in range(n_episodes):
        s = env.start_state
        total = 0.0
        for _ in range(100):
            a = int(policy[s])
            s, r, done = env.step(s, a, rng)
            total += r
            if done:
                break
        returns[ep] = total
        if s == env.GOAL:
            successes += 1
        elif s == env.CATASTROPHE:
            catastrophes += 1

    sorted_r = np.sort(returns)
    k = max(1, int(0.05 * len(sorted_r)))
    cvar_5 = float(np.mean(sorted_r[:k]))
    return PolicyStats(
        mean_return=float(returns.mean()),
        std_return=float(returns.std()),
        success_rate=successes / n_episodes,
        catastrophe_rate=catastrophes / n_episodes,
        cvar_5=cvar_5,
        worst_return=float(sorted_r[0]),
        entropic_values={b: entropic_value(returns, b) for b in betas},
    )

"""Tests for risk measures (sample and categorical) and distributional DP."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from entropic_risk_jev.src.env import RiskyCheckout
from entropic_risk_jev.src.distributional import (
    distributional_policy_evaluation,
    distributional_risk_control,
)
from entropic_risk_jev.src.risk_measures import (
    cvar_categorical,
    cvar_samples,
    entropic_categorical,
    entropic_samples,
    make_risk_fn,
    mean_categorical,
)
from entropic_risk_jev.src.value_iteration import risk_sensitive_value_iteration


def test_cvar_samples_worst_tail():
    r = np.arange(100.0)  # 0..99
    # worst 10% = 0..9, mean 4.5
    assert np.isclose(cvar_samples(r, 0.1), 4.5)


def test_cvar_categorical_skips_empty_bottom_atom():
    # Bottom atom has zero mass; CVaR must not return it.
    z = np.array([-100.0, 0.0, 1.0, 2.0])
    p = np.array([0.0, 0.1, 0.4, 0.5])
    # worst 10% is entirely the atom at 0.0
    assert np.isclose(cvar_categorical(z, p, 0.1), 0.0)


def test_cvar_categorical_matches_samples():
    rng = np.random.default_rng(0)
    vals = rng.normal(size=20000)
    z, counts = np.unique(np.round(vals, 1), return_counts=True)
    p = counts / counts.sum()
    assert abs(cvar_categorical(z, p, 0.1) - cvar_samples(vals, 0.1)) < 0.05


def test_entropic_categorical_matches_samples():
    rng = np.random.default_rng(1)
    vals = rng.normal(size=50000)
    z, counts = np.unique(np.round(vals, 2), return_counts=True)
    p = counts / counts.sum()
    assert abs(entropic_categorical(z, p, 0.5) - entropic_samples(vals, 0.5)) < 0.03


def test_mean_categorical():
    z = np.array([-1.0, 0.0, 2.0])
    p = np.array([0.5, 0.25, 0.25])
    assert np.isclose(mean_categorical(z, p), -0.5 + 0.0 + 0.5)


def test_distributional_entropic_matches_exact_vi():
    env = RiskyCheckout()
    beta = 0.2
    exact = risk_sensitive_value_iteration(env, beta).policy
    dist = distributional_risk_control(env, make_risk_fn("entropic", beta=beta),
                                       natoms=401).policy
    nonterm = [s for s in range(env.n_states) if not env.is_terminal(s)]
    assert all(exact[s] == dist[s] for s in nonterm)


def test_policy_eval_mean_matches_vi_value_at_beta0():
    env = RiskyCheckout()
    sol = risk_sensitive_value_iteration(env, 0.0)
    p, z = distributional_policy_evaluation(env, sol.policy, natoms=601)
    assert np.isclose(mean_categorical(z, p), sol.V[env.start_state], atol=0.05)

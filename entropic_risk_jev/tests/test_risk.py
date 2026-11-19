"""Tests for the entropic risk measure and Bellman backup."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from entropic_risk_jev.src.risk import entropic_backup, entropic_value, stable_logsumexp


def test_logsumexp_matches_naive():
    x = np.array([0.1, -2.0, 3.0, -0.5])
    assert np.isclose(stable_logsumexp(x), np.log(np.sum(np.exp(x))))


def test_logsumexp_handles_neg_inf():
    x = np.array([-np.inf, 1.0, -np.inf, 2.0])
    assert np.isclose(stable_logsumexp(x), np.log(np.exp(1.0) + np.exp(2.0)))


def test_entropic_value_reduces_to_mean_at_beta0():
    r = np.array([1.0, -3.0, 2.0, 5.0, 0.0])
    assert np.isclose(entropic_value(r, 0.0), r.mean())


def test_entropic_value_is_risk_averse():
    # Same mean, different variance: risk-averse value prefers the safe one.
    safe = np.array([1.0, 1.0, 1.0, 1.0])
    risky = np.array([-3.0, 5.0, -3.0, 5.0])  # mean 1.0, high variance
    assert np.isclose(safe.mean(), risky.mean())
    beta = 0.5
    assert entropic_value(safe, beta) > entropic_value(risky, beta)


def test_entropic_value_penalizes_rare_catastrophe():
    # One rare but catastrophic outcome should tank the entropic value.
    r = np.array([1.0] * 999 + [-100.0])  # mean = 0.899 (positive)
    assert entropic_value(r, 0.0) > 0  # mean still positive
    assert entropic_value(r, 0.3) < -1.0  # risk-averse sees the tail


def test_backup_reduces_to_expected_value_at_beta0():
    rng = np.random.default_rng(0)
    S, A = 4, 3
    P = rng.random((S, A, S))
    P /= P.sum(axis=2, keepdims=True)
    R = rng.standard_normal((S, A, S))
    V = rng.standard_normal(S)
    q0 = entropic_backup(P, R, V, 0.0)
    q_tiny = entropic_backup(P, R, V, 1e-6)
    expected = np.einsum("sap,sap->sa", P, R + V[None, None, :])
    assert np.allclose(q0, expected)
    assert np.allclose(q_tiny, expected, atol=1e-3)

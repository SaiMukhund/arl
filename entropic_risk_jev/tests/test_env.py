"""Tests for the RiskyCheckout environment and value iteration."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from entropic_risk_jev.src.env import ASK, FAST, RiskyCheckout, RiskyCheckoutConfig
from entropic_risk_jev.src.value_iteration import risk_sensitive_value_iteration


def test_transition_rows_sum_to_one():
    env = RiskyCheckout()
    assert np.allclose(env.P.sum(axis=2), 1.0)


def test_terminals_absorbing():
    env = RiskyCheckout()
    for t in env.terminal_states:
        for a in range(env.n_actions):
            assert env.P[t, a, t] == 1.0
            assert env.R[t, a, t] == 0.0


def test_render_matches_jev_format():
    env = RiskyCheckout()
    r = env.render_state(0)
    assert set(r.keys()) == {"state", "question", "options"}
    assert len(r["options"]) == env.n_actions
    for opt in r["options"]:
        assert set(opt.keys()) == {"label", "key", "description"}
    assert env.render_state(env.GOAL)["options"] == []


def test_risk_neutral_uses_fast_more_than_risk_averse():
    env = RiskyCheckout(RiskyCheckoutConfig())
    neutral = risk_sensitive_value_iteration(env, 0.0).policy[: env.n_stages]
    averse = risk_sensitive_value_iteration(env, 0.6).policy[: env.n_stages]
    n_fast_neutral = int(np.sum(neutral == FAST))
    n_fast_averse = int(np.sum(averse == FAST))
    assert n_fast_neutral > n_fast_averse
    # Fully risk-averse policy escalates (ASK) and never takes FAST.
    assert n_fast_averse == 0
    assert np.all(averse == ASK)


def test_fast_peels_off_expensive_stages_first():
    # With a cost gradient, moderate risk aversion should keep FAST on the
    # cheapest (early) stages but not the most expensive (late) ones.
    env = RiskyCheckout(RiskyCheckoutConfig())
    pol = risk_sensitive_value_iteration(env, 0.08).policy[: env.n_stages]
    # Last (most expensive) stage must not be FAST; first must still be FAST.
    assert pol[-1] != FAST
    assert pol[0] == FAST


def test_value_monotone_decreasing_in_beta():
    # The certainty-equivalent start value cannot increase as we become more
    # risk averse (risk aversion is pessimistic).
    env = RiskyCheckout()
    betas = [0.0, 0.05, 0.1, 0.3, 0.6]
    v0 = [risk_sensitive_value_iteration(env, b).V[env.start_state] for b in betas]
    for a, b in zip(v0, v0[1:]):
        assert b <= a + 1e-6

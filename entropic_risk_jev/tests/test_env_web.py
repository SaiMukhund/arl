"""Tests for the larger RiskyWeb environment and exact policy evaluation."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from entropic_risk_jev.src.env import FAST
from entropic_risk_jev.src.env_web import RiskyWeb, RiskyWebConfig
from entropic_risk_jev.src.value_iteration import (
    entropic_policy_value,
    risk_sensitive_value_iteration,
)


def test_web_transition_rows_sum_to_one():
    env = RiskyWeb(RiskyWebConfig(seed=1))
    assert np.allclose(env.P.sum(axis=2), 1.0)


def test_web_render_jev_format():
    env = RiskyWeb()
    r = env.render_state(0)
    assert len(r["options"]) == env.n_actions
    assert env.render_state(env.GOAL)["options"] == []


def test_web_risk_aversion_reduces_fast():
    env = RiskyWeb(RiskyWebConfig(seed=0))
    nonterm = [s for s in range(env.n_states) if not env.is_terminal(s)]
    neutral = risk_sensitive_value_iteration(env, 0.0).policy
    averse = risk_sensitive_value_iteration(env, 0.4).policy
    n_fast_neutral = sum(neutral[s] == FAST for s in nonterm)
    n_fast_averse = sum(averse[s] == FAST for s in nonterm)
    assert n_fast_neutral > n_fast_averse


def test_oracle_beats_risk_neutral_on_entropic_objective():
    env = RiskyWeb(RiskyWebConfig(seed=0))
    beta = 0.15
    oracle = risk_sensitive_value_iteration(env, beta).policy
    rn = risk_sensitive_value_iteration(env, 0.0).policy
    u_oracle = entropic_policy_value(env, oracle, beta)
    u_rn = entropic_policy_value(env, rn, beta)
    assert u_oracle > u_rn  # risk-aware policy wins the risk objective


def test_policy_eval_matches_vi_value_for_optimal_policy():
    env = RiskyWeb(RiskyWebConfig(seed=2))
    beta = 0.1
    sol = risk_sensitive_value_iteration(env, beta)
    u = entropic_policy_value(env, sol.policy, beta)
    assert np.isclose(u, sol.V[env.start_state], atol=1e-6)

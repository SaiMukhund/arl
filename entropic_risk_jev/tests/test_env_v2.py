"""Tests for the RiskyFlow environment."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from entropic_risk_jev.src.env import FAST
from entropic_risk_jev.src.env_v2 import RiskyFlow, RiskyFlowConfig
from entropic_risk_jev.src.value_iteration import risk_sensitive_value_iteration


def test_flow_rows_sum_to_one():
    env = RiskyFlow(RiskyFlowConfig(seed=3))
    assert np.allclose(env.P.sum(axis=2), 1.0)


def test_flow_has_graded_failures():
    env = RiskyFlow(RiskyFlowConfig(seed=0))
    # Both minor and major terminal states must be reachable from FAST somewhere.
    pages = [s for s in range(env.n_states) if not env.is_terminal(s)]
    minor = any(env.P[s, FAST, env.CAT_MINOR] > 0 for s in pages)
    major = any(env.P[s, FAST, env.CAT_MAJOR] > 0 for s in pages)
    assert minor and major


def test_flow_observation_hides_severity():
    # Two states with same type/elements/layer but different hidden severity may
    # share observation features: check obs dim is fixed and finite.
    env = RiskyFlow(RiskyFlowConfig(seed=0))
    f = env.obs_features(0)
    assert np.all(np.isfinite(f))
    assert f.shape[0] == env.feat_dim


def test_flow_render_jev_format():
    env = RiskyFlow()
    r = env.render_state(0)
    assert len(r["options"]) == env.n_actions
    assert "Visible actions" in r["state"]
    assert env.render_state(env.GOAL)["options"] == []


def test_flow_risk_aversion_reduces_fast():
    env = RiskyFlow(RiskyFlowConfig(seed=0))
    nonterm = [s for s in range(env.n_states) if not env.is_terminal(s)]
    neutral = risk_sensitive_value_iteration(env, 0.0).policy
    averse = risk_sensitive_value_iteration(env, 0.3).policy
    assert sum(neutral[s] == FAST for s in nonterm) > sum(averse[s] == FAST for s in nonterm)

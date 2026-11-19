import numpy as np
from .environment import TransactionEnv, OBS_DIM, ACTION_NAMES

def test_state_and_sequential_actions():
    env = TransactionEnv(horizon=4, seed=11)
    obs = env.reset()
    assert obs.shape == (OBS_DIM,)
    assert len(ACTION_NAMES) == 3
    _, _, done, info = env.step(1)
    assert not done and info["verified"]
    _, _, done, info = env.step(2)
    assert not done and info["confirmed"]
    _, reward, done, info = env.step(0)
    assert done and info["success"] in (0, 1)
    assert info["cost"] >= info["operational_cost"]

def test_checks_reduce_catastrophe_probability():
    env = TransactionEnv(seed=4)
    env.reset()
    p0 = env.outcome_distribution(False, False)[2]
    p1 = env.outcome_distribution(True, False)[2]
    p2 = env.outcome_distribution(False, True)[2]
    p3 = env.outcome_distribution(True, True)[2]
    assert p1 <= p0 and p2 <= p0 and p3 <= p1 and p3 <= p2

def test_counterfactual_does_not_mutate_environment():
    env = TransactionEnv(seed=8)
    obs = env.reset()
    hazard = env.shifted_hazard
    a = env.sample_action_rollouts(obs, 1, 500, 99)
    b = env.sample_action_rollouts(obs, 1, 500, 99)
    assert env.shifted_hazard == hazard
    assert all(np.array_equal(x, y) for x, y in zip(a, b))

def test_deadline_forces_execution():
    env = TransactionEnv(horizon=2, seed=2)
    env.reset()
    _, _, done, _ = env.step(1)
    assert not done
    _, _, done, info = env.step(2)
    assert done and info["forced_execution"]

def test_forced_execution_keeps_check_fee():
    env = TransactionEnv(horizon=1, seed=3)
    env.reset()
    fee = 1.8 + 1.4*env.amount
    _, reward, done, info = env.step(1)
    assert done and info["forced_execution"]
    terminal = (28.0 if info["success"] else 0.0) - info["loss"] - 2.0
    assert np.isclose(reward, terminal - fee)

def test_counterfactual_charges_redundant_check():
    env = TransactionEnv(horizon=4, seed=5)
    obs = env.reset()
    obs, _, _, _ = env.step(1)
    obs, _, _, _ = env.step(2)  # both checks done; one slot left
    paid = env.total_operational_cost
    _, costs, _ = env.sample_action_rollouts(obs, 1, 200, 0)
    assert np.all(costs >= paid + 0.6 - 1e-5)

def test_regime_validation():
    try:
        TransactionEnv(regime="unknown")
    except ValueError:
        return
    raise AssertionError("invalid regime should fail")

if __name__ == "__main__":
    for test in [test_state_and_sequential_actions,
                 test_checks_reduce_catastrophe_probability,
                 test_counterfactual_does_not_mutate_environment,
                 test_deadline_forces_execution,
                 test_forced_execution_keeps_check_fee,
                 test_counterfactual_charges_redundant_check,
                 test_regime_validation]:
        test()
    print("7 environment tests passed")

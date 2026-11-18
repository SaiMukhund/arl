import numpy as np
from .config import ACTIONS

def sample_scenarios(n, rng, shifted=False):
    # Observable context variables. All are synthetic.
    amount = rng.uniform(100, 10000, n)
    recipient_risk = rng.beta(1.5, 8.0, n)
    urgency = rng.uniform(0, 1, n)
    network_risk = rng.beta(1.2, 12.0, n)
    novelty = rng.uniform(0, 1, n)
    if shifted:
        # Distribution shift: more risky recipients/network and heavier rare-event tail.
        recipient_risk = np.clip(recipient_risk * 1.8 + 0.08, 0, 1)
        network_risk = np.clip(network_risk * 1.7 + 0.05, 0, 1)
    return np.column_stack([amount, recipient_risk, urgency, network_risk, novelty])

FEATURE_NAMES = ["amount", "recipient_risk", "urgency", "network_risk", "novelty"]

def outcome_probabilities(x, action, shifted=False):
    amount, recipient, urgency, network, novelty = x
    # Failure probability depends on scenario and action.
    if action == "direct_transfer":
        p_fail = 0.006 + 0.035*recipient + 0.020*network + 0.004*novelty
        p_cat = 0.0015 + 0.012*recipient + 0.008*network
    elif action == "verify_recipient":
        p_fail = 0.018 + 0.018*recipient + 0.012*network + 0.003*novelty
        p_cat = 0.00025 + 0.0025*recipient + 0.0015*network
    else:  # request_confirmation
        p_fail = 0.035 + 0.010*recipient + 0.008*network
        p_cat = 0.00008 + 0.0010*recipient + 0.0008*network

    p_fail = float(np.clip(p_fail, 0.001, 0.25))
    p_cat = float(np.clip(p_cat, 0.00002, min(p_fail * 0.7, 0.05)))
    if shifted:
        p_cat = min(p_cat * 2.5, p_fail * 0.8)
    p_success = 1.0 - p_fail
    p_ordinary = p_fail - p_cat
    return p_success, p_ordinary, p_cat

def sample_outcome(x, action, rng, shifted=False):
    p_success, p_ordinary, p_cat = outcome_probabilities(x, action, shifted)
    u = rng.random()
    if u < p_success:
        return 1, 0.0, 0
    if u < p_success + p_ordinary:
        # Ordinary failure cost: action-specific and context-sensitive.
        base = {"direct_transfer": 18, "verify_recipient": 5, "request_confirmation": 2}[action]
        cost = base + 0.002 * float(x[0]) + 8 * float(x[1])
        return 0, float(cost), 0
    # Rare catastrophic failure. Heavy-tailed severity, with direct transfer worst.
    scale = {"direct_transfer": 1100, "verify_recipient": 180, "request_confirmation": 65}[action]
    if shifted:
        scale *= 1.8
    severity = float(rng.lognormal(mean=np.log(scale), sigma=0.65))
    return 0, severity, 1

def collect_dataset(n, rng, shifted=False):
    rows = []
    scenarios = sample_scenarios(n, rng, shifted=shifted)
    for sid, x in enumerate(scenarios):
        for action in ACTIONS:
            success, cost, catastrophic = sample_outcome(x, action, rng, shifted=shifted)
            rows.append([*x, ACTIONS.index(action), success, cost, catastrophic, sid])
    return rows

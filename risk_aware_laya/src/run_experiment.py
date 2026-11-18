import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from .config import Config, ACTIONS
from .environment import collect_dataset, sample_scenarios, sample_outcome, outcome_probabilities, FEATURE_NAMES
from .models import RiskModels
from .metrics import empirical_cvar, calibration_metrics

def make_action_features(scenarios):
    rows = []
    for x in scenarios:
        for a_idx, action in enumerate(ACTIONS):
            rows.append([*x, a_idx])
    return np.asarray(rows, dtype=float)

def describe_state(x, action):
    amount, recipient, urgency, network, novelty = x
    return (
        f"Transaction amount: ${amount:.2f}. Recipient risk score: {recipient:.3f}. "
        f"Urgency: {urgency:.3f}. Network risk: {network:.3f}. "
        f"Novelty: {novelty:.3f}. Candidate action: {action}. "
        "The task is to complete the transaction correctly while avoiding losses."
    )

def get_laya_predictions(scenarios, enabled):
    if not enabled:
        return None
    try:
        from .laya_adapter import LayaAdapter
        adapter = LayaAdapter()
        vals = []
        for x in scenarios:
            for action in ACTIONS:
                vals.append(adapter.predict_success(describe_state(x, action)))
        return np.asarray(vals)
    except Exception as e:
        print(f"[WARN] Laya unavailable or inference failed: {e}")
        print("[WARN] Continuing with learned local success predictor.")
        return None

def evaluate_policies(models, test_scenarios, cfg, rng, laya_probs=None):
    X = make_action_features(test_scenarios)
    ps = models.predict_success(X).reshape(-1, len(ACTIONS))
    costs = models.predict_cost(X).reshape(-1, len(ACTIONS))
    cvars = models.predict_cvar(X, cfg.cvar_alpha).reshape(-1, len(ACTIONS))
    cats = models.predict_catastrophic(X).reshape(-1, len(ACTIONS))

    if laya_probs is not None:
        p_laya = np.clip(laya_probs.reshape(-1, len(ACTIONS)), 1e-5, 1-1e-5)
    else:
        p_laya = ps

    # Normalize cost terms to keep the objective numerically interpretable.
    score_expected = p_laya - cfg.lambda_cost * costs / cfg.cost_scale
    score_cvar = p_laya - cfg.lambda_cost * costs / cfg.cost_scale - cfg.beta_cvar * cvars / cfg.cost_scale

    # Evaluate in a shifted regime, with fresh stochastic outcomes.
    policies = {
        "success_only": np.argmax(p_laya, axis=1),
        "expected_cost": np.argmax(score_expected, axis=1),
        "cvar": np.argmax(score_cvar, axis=1),
    }
    # Safety-constrained policy: choose highest CVaR score among actions under risk limit.
    safe_choices = []
    for i in range(len(test_scenarios)):
        allowed = np.where(cats[i] <= cfg.max_catastrophic_probability)[0]
        if len(allowed) == 0:
            safe_choices.append(int(np.argmin(cats[i])))
        else:
            safe_choices.append(int(allowed[np.argmax(score_cvar[i, allowed])]))
    policies["cvar_safety"] = np.asarray(safe_choices)

    rows = []
    # Repeated rollouts per held-out scenario reduce variance in rare-event metrics.
    replications = 20
    for name, choices in policies.items():
        successes, realized_costs, catastrophic = [], [], []
        for i, x in enumerate(test_scenarios):
            a = ACTIONS[int(choices[i])]
            for _ in range(replications):
                s, c, cat = sample_outcome(x, a, rng, shifted=True)
                successes.append(s); realized_costs.append(c); catastrophic.append(cat)
        successes = np.asarray(successes)
        realized_costs = np.asarray(realized_costs)
        catastrophic = np.asarray(catastrophic)
        rows.append({
            "policy": name,
            "success_rate": successes.mean(),
            "mean_cost": realized_costs.mean(),
            "catastrophic_rate": catastrophic.mean(),
            "cvar_5pct": empirical_cvar(realized_costs, 0.05),
            "total_cost": realized_costs.sum()
        })
    prediction_rows = []
    for i, x in enumerate(test_scenarios):
        for j, action in enumerate(ACTIONS):
            prediction_rows.append({
                **dict(zip(FEATURE_NAMES, x)),
                "action": action,
                "predicted_success": p_laya[i,j],
                "predicted_expected_cost": costs[i,j],
                "predicted_cvar": cvars[i,j],
                "predicted_catastrophic_probability": cats[i,j]
            })
    return pd.DataFrame(rows), pd.DataFrame(prediction_rows), p_laya, policies

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-laya", action="store_true")
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    cfg = Config(episodes=args.episodes, seed=args.seed, use_laya=not args.no_laya)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)

    print("Generating synthetic action-conditional trajectories...")
    raw = collect_dataset(cfg.episodes, rng, shifted=False)
    cols = FEATURE_NAMES + ["action_id", "success", "cost", "catastrophic", "scenario_id"]
    df = pd.DataFrame(raw, columns=cols)
    X = df[FEATURE_NAMES + ["action_id"]].to_numpy()
    y = df["success"].astype(int).to_numpy()
    c = df["cost"].astype(float).to_numpy()
    cat = df["catastrophic"].astype(int).to_numpy()

    # Split by scenario ID to avoid leaking the same scenario across splits.
    ids = df["scenario_id"].unique()
    train_ids, test_ids = train_test_split(ids, test_size=cfg.test_size, random_state=cfg.seed)
    train_mask = df["scenario_id"].isin(train_ids).to_numpy()
    test_mask = df["scenario_id"].isin(test_ids).to_numpy()
    train_ids2, val_ids = train_test_split(train_ids, test_size=cfg.validation_size, random_state=cfg.seed+1)
    train_mask = df["scenario_id"].isin(train_ids2).to_numpy()
    val_mask = df["scenario_id"].isin(val_ids).to_numpy()

    print("Training success, cost, catastrophic-risk and quantile models...")
    models = RiskModels()
    models.fit(X[train_mask], y[train_mask], c[train_mask], cat[train_mask])
    models.calibrate_success(X[val_mask], y[val_mask])

    val_metrics = calibration_metrics(y[val_mask], models.predict_success(X[val_mask]))
    print("Validation success calibration:", val_metrics)

    # Generate held-out scenarios under shifted distribution.
    test_scenarios = sample_scenarios(max(1000, len(test_ids)), rng, shifted=True)
    print("Running optional Laya inference (can be slow on CPU)...")
    laya_probs = None
    if cfg.use_laya:
        # Fit a probability calibrator for Laya on validation scenarios only.
        val_scenarios = sample_scenarios(max(500, len(val_ids)), rng, shifted=False)
        raw_val_laya = get_laya_predictions(val_scenarios, enabled=True)
        if raw_val_laya is not None:
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import brier_score_loss
            val_X = make_action_features(val_scenarios)
            val_y = models.success  # placeholder only to keep model labels separate
            # Obtain simulator labels for the same validation scenario/action pairs.
            val_labels = []
            for x in val_scenarios:
                for action in ACTIONS:
                    p_s, _, _ = outcome_probabilities(x, action, shifted=False)
                    val_labels.append(int(rng.random() < p_s))
            eps = 1e-6
            z = np.log(np.clip(raw_val_laya, eps, 1-eps) /
                       np.clip(1-raw_val_laya, eps, 1-eps)).reshape(-1, 1)
            laya_calibrator = LogisticRegression().fit(z, np.asarray(val_labels))
            raw_test_laya = get_laya_predictions(test_scenarios, enabled=True)
            if raw_test_laya is not None:
                zt = np.log(np.clip(raw_test_laya, eps, 1-eps) /
                            np.clip(1-raw_test_laya, eps, 1-eps)).reshape(-1, 1)
                laya_probs = laya_calibrator.predict_proba(zt)[:, 1]
                print("Laya calibrated on an independent validation set.")
    metrics, preds, p_used, policies = evaluate_policies(
        models, test_scenarios, cfg, rng, laya_probs=laya_probs
    )
    metrics.to_csv(out / "metrics.csv", index=False)
    preds.to_csv(out / "action_level_predictions.csv", index=False)
    with open(out / "summary.txt", "w") as f:
        f.write("Risk-aware Laya synthetic experiment\n")
        f.write(f"episodes={cfg.episodes}, seed={cfg.seed}\n")
        f.write(f"Laya used={laya_probs is not None}\n\n")
        f.write("Validation calibration:\n" + str(val_metrics) + "\n\n")
        f.write(metrics.to_string(index=False) + "\n")

    # Success vs catastrophic risk.
    plt.figure()
    plt.scatter(metrics["catastrophic_rate"], metrics["success_rate"])
    for _, r in metrics.iterrows():
        plt.annotate(r["policy"], (r["catastrophic_rate"], r["success_rate"]))
    plt.xlabel("Catastrophic failure rate (lower is better)")
    plt.ylabel("Task success rate (higher is better)")
    plt.title("Success-risk trade-off (shifted test environment)")
    plt.tight_layout()
    plt.savefig(out / "risk_tradeoff.png", dpi=160)
    plt.close()

    plt.figure()
    plt.bar(metrics["policy"], metrics["cvar_5pct"])
    plt.ylabel("Empirical upper-tail CVaR (5%)")
    plt.title("Tail cost by policy")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out / "cvar_by_policy.png", dpi=160)
    plt.close()

    print("\nResults:")
    print(metrics.to_string(index=False))
    print(f"\nSaved outputs to: {out.resolve()}")

if __name__ == "__main__":
    main()

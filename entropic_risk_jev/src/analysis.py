"""Extra analysis plots that surface conclusions beyond the main comparison table.

Produces:
  1. risk-return frontier        -- mean return vs tail risk, per method (the
     central trade-off: risk-neutral methods win mean but lose the tail).
  2. return-distribution overlay -- the actual return distributions of the
     risk-neutral / entropic / CVaR / oracle policies, showing the heavy tail
     being cut off.
  3. action-by-page-type heatmap -- which action each attitude takes on each page
     type, showing risk-averse policies escalate (ASK) on dangerous page types.
  4. RiskyFlow beta sweep        -- entropic objective and catastrophe rate of the
     oracle vs the fixed risk-neutral policy as risk aversion grows.

Model-free (DQN) methods appear in (1) via the saved metrics JSON; the policy-
level plots (2)-(4) use the exact model-based policies (deterministic, seed-fixed).

Run (after experiment_v2 has produced the JSON):
    python -m entropic_risk_jev.src.analysis --beta 0.1
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .distributional import (
    distributional_policy_evaluation,
    distributional_risk_control,
    policy_terminal_probs,
)
from .env import ACTION_NAMES
from .env_v2 import TYPE_NAMES, RiskyFlow, RiskyFlowConfig
from .risk_measures import (
    cvar_categorical,
    entropic_categorical,
    make_risk_fn,
    mean_categorical,
)
from .value_iteration import risk_sensitive_value_iteration

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def frontier_plot(beta, alpha, seed):
    plt = _mpl()
    path = os.path.join(RESULTS_DIR, f"riskyflow_beta{beta}.json")
    rows = json.load(open(path))["rows"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    for name, r in rows.items():
        ax1.scatter(max(r["p_major"], 1e-4), r["mean"], s=60)
        ax1.annotate(name, (max(r["p_major"], 1e-4), r["mean"]), fontsize=7,
                     xytext=(4, 2), textcoords="offset points")
        ax2.scatter(r["cvar"], r["mean"], s=60)
        ax2.annotate(name, (r["cvar"], r["mean"]), fontsize=7,
                     xytext=(4, 2), textcoords="offset points")
    ax1.set_xscale("log")
    ax1.set_xlabel("P(major catastrophe)  (log, lower=better)")
    ax1.set_ylabel("mean return (higher=better)")
    ax1.set_title("Risk-return frontier: mean vs catastrophe probability")
    ax2.set_xlabel(f"CVaR_{alpha}  (higher=better)")
    ax2.set_ylabel("mean return")
    ax2.set_title("Risk-return frontier: mean vs tail value")
    fig.suptitle(f"RiskyFlow risk-return trade-off (beta={beta})")
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, f"frontier_beta{beta}.png")
    fig.savefig(out, dpi=130); print(f"[saved] {out}")


def _model_based_policies(env, beta, alpha):
    return {
        "risk-neutral": distributional_risk_control(env, make_risk_fn("neutral"), natoms=251).policy,
        "entropic (oracle)": risk_sensitive_value_iteration(env, beta).policy,
        "nested CVaR": distributional_risk_control(env, make_risk_fn("cvar", alpha=alpha), natoms=251).policy,
    }


def return_distribution_plot(beta, alpha, seed):
    plt = _mpl()
    env = RiskyFlow(RiskyFlowConfig(seed=seed))
    pols = _model_based_policies(env, beta, alpha)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    colors = {"risk-neutral": "#dc2626", "entropic (oracle)": "#16a34a", "nested CVaR": "#7c3aed"}
    for name, pol in pols.items():
        p, z = distributional_policy_evaluation(env, pol, natoms=601)
        # coarse-bin for a readable step plot
        mask = p > 1e-5
        ax.step(z[mask], p[mask], where="mid", label=name, color=colors[name], lw=1.6)
    ax.set_yscale("log")
    ax.set_xlabel("total return")
    ax.set_ylabel("probability (log)")
    ax.set_title(f"Return distributions: risk aversion removes the left tail (beta={beta})")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, f"return_dist_beta{beta}.png")
    fig.savefig(out, dpi=130); print(f"[saved] {out}")


def action_heatmap(beta, alpha, seed):
    plt = _mpl()
    env = RiskyFlow(RiskyFlowConfig(seed=seed))
    pols = _model_based_policies(env, beta, alpha)
    pages = [s for s in range(env.n_states) if not env.is_terminal(s)]
    # For each policy and page type, fraction choosing ASK (most cautious).
    data = np.zeros((len(pols), len(TYPE_NAMES)))
    for i, (name, pol) in enumerate(pols.items()):
        for j, t in enumerate(TYPE_NAMES):
            states = [s for s in pages if env.page_type[s] == t]
            if states:
                # cautiousness score: FAST=0, SAFE=0.5, ASK=1
                score = np.mean([[0, 0.5, 1.0][pol[s]] for s in states])
                data[i, j] = score
    fig, ax = plt.subplots(figsize=(10, 3.6))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)
    ax.set_xticks(range(len(TYPE_NAMES))); ax.set_xticklabels(TYPE_NAMES, rotation=30, ha="right")
    ax.set_yticks(range(len(pols))); ax.set_yticklabels(list(pols.keys()))
    for i in range(len(pols)):
        for j in range(len(TYPE_NAMES)):
            ax.text(j, i, f"{data[i,j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Cautiousness by page type (0=FAST, 0.5=SAFE, 1=ASK/escalate)")
    fig.colorbar(im, ax=ax, label="cautiousness")
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, f"action_heatmap_beta{beta}.png")
    fig.savefig(out, dpi=130); print(f"[saved] {out}")


def beta_sweep_plot(betas, alpha, seed):
    plt = _mpl()
    env = RiskyFlow(RiskyFlowConfig(seed=seed))
    rn = risk_sensitive_value_iteration(env, 0.0).policy
    oU, rnU, oPm, rnPm = [], [], [], []
    for b in betas:
        orc = risk_sensitive_value_iteration(env, b).policy
        po, zo = distributional_policy_evaluation(env, orc, natoms=601)
        pr, zr = distributional_policy_evaluation(env, rn, natoms=601)
        oU.append(entropic_categorical(zo, po, b)); rnU.append(entropic_categorical(zr, pr, b))
        oPm.append(policy_terminal_probs(env, orc)[env.CAT_MAJOR])
        rnPm.append(policy_terminal_probs(env, rn)[env.CAT_MAJOR])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    a1.plot(betas, oU, "o-", color="#16a34a", label="entropic oracle")
    a1.plot(betas, rnU, "^-", color="#dc2626", label="risk-neutral policy")
    a1.set_xlabel("beta"); a1.set_ylabel("entropic U_beta"); a1.legend()
    a1.set_title("Risk objective: adapting vs fixed risk-neutral")
    a2.plot(betas, oPm, "o-", color="#16a34a", label="entropic oracle")
    a2.axhline(rnPm[0], ls="--", color="#dc2626", label="risk-neutral policy")
    a2.set_xlabel("beta"); a2.set_ylabel("P(major catastrophe)"); a2.legend()
    a2.set_title("Catastrophe rate of the entropic oracle vs beta")
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, f"beta_sweep_flow.png")
    fig.savefig(out, dpi=130); print(f"[saved] {out}")


def exp3_beta_sweep(betas, alpha, seed, overlay_native_yev=True):
    """Experiment-3 beta sweep: oracle-entropic (adapts) vs fixed risk-neutral and
    fixed nested-CVaR policies, scored by the entropic objective U_beta and the
    major-catastrophe rate at each beta. Optionally overlays natively fine-tuned
    yev0-4b points read from results/yev_finetune_flow_beta*.json."""
    plt = _mpl()
    env = RiskyFlow(RiskyFlowConfig(seed=seed))
    rn = risk_sensitive_value_iteration(env, 0.0).policy
    cvar_pol = distributional_risk_control(env, make_risk_fn("cvar", alpha=alpha),
                                           natoms=251).policy

    def metrics(pol, b):
        p, z = distributional_policy_evaluation(env, pol, natoms=601)
        return (entropic_categorical(z, p, b),
                policy_terminal_probs(env, pol)[env.CAT_MAJOR])

    curves = {k: {"U": [], "Pm": []} for k in ["oracle", "risk_neutral", "cvar"]}
    table = []
    for b in betas:
        orc = risk_sensitive_value_iteration(env, b).policy
        uo, po = metrics(orc, b)
        ur, pr = metrics(rn, b)
        uc, pc = metrics(cvar_pol, b)
        for k, (u, pm) in zip(curves, [(uo, po), (ur, pr), (uc, pc)]):
            curves[k]["U"].append(u); curves[k]["Pm"].append(pm)
        table.append((b, uo, ur, uc, po, pr, pc))

    # native yev overlay (verified points)
    yev_pts = {}
    if overlay_native_yev:
        for b in betas:
            f = os.path.join(RESULTS_DIR, f"yev_finetune_flow_beta{b}.json")
            if os.path.exists(f):
                r = json.load(open(f))["rows"].get("yev0-4b finetuned (beta={})".format(b))
                if r:
                    yev_pts[b] = (r["U_beta"], r["p_major"])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    a1.plot(betas, curves["oracle"]["U"], "o-", color="#16a34a", label="entropic oracle")
    a1.plot(betas, curves["risk_neutral"]["U"], "^-", color="#dc2626", label="risk-neutral policy (fixed)")
    a1.plot(betas, curves["cvar"]["U"], "s-", color="#7c3aed", label=f"nested CVaR policy (fixed)")
    if yev_pts:
        xs = sorted(yev_pts); a1.scatter(xs, [yev_pts[b][0] for b in xs], marker="*",
                    s=200, color="#f59e0b", zorder=5, label="yev0-4b finetuned (native)")
    a1.set_xlabel("risk aversion  beta"); a1.set_ylabel("entropic objective  U_beta")
    a1.set_title("Experiment 3 (RiskyFlow): risk objective vs beta"); a1.legend(fontsize=8)

    a2.plot(betas, curves["oracle"]["Pm"], "o-", color="#16a34a", label="entropic oracle")
    a2.plot(betas, curves["risk_neutral"]["Pm"], "^-", color="#dc2626", label="risk-neutral policy (fixed)")
    a2.plot(betas, curves["cvar"]["Pm"], "s-", color="#7c3aed", label="nested CVaR policy (fixed)")
    if yev_pts:
        xs = sorted(yev_pts); a2.scatter(xs, [yev_pts[b][1] for b in xs], marker="*",
                    s=200, color="#f59e0b", zorder=5, label="yev0-4b finetuned (native)")
    a2.set_xlabel("risk aversion  beta"); a2.set_ylabel("P(major catastrophe)")
    a2.set_title("Experiment 3: catastrophe rate vs beta"); a2.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "exp3_beta_sweep.png")
    fig.savefig(out, dpi=130); print(f"[saved] {out}")

    print("\nExp 3 beta sweep (U_beta | P(major)):")
    print(f"{'beta':>5} | {'oracle':>8} {'r-neut':>8} {'cvar':>8} | "
          f"{'orc Pmaj':>8} {'rn Pmaj':>8} {'cv Pmaj':>8}" +
          ("  yev*" if yev_pts else ""))
    for (b, uo, ur, uc, po, pr, pc) in table:
        extra = f"  U={yev_pts[b][0]:.2f}" if b in yev_pts else ""
        print(f"{b:>5.2f} | {uo:>8.2f} {ur:>8.2f} {uc:>8.2f} | "
              f"{po:>8.4f} {pr:>8.4f} {pc:>8.4f}{extra}")
    json.dump({"betas": betas, "curves": curves, "yev_pts": yev_pts},
              open(os.path.join(RESULTS_DIR, "exp3_beta_sweep.json"), "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sweep-only", action="store_true")
    args = ap.parse_args()
    if not args.sweep_only:
        frontier_plot(args.beta, args.alpha, args.seed)
        return_distribution_plot(args.beta, args.alpha, args.seed)
        action_heatmap(args.beta, args.alpha, args.seed)
        beta_sweep_plot([0.0, 0.02, 0.05, 0.1, 0.15, 0.25, 0.4], args.alpha, args.seed)
    exp3_beta_sweep([0.0, 0.05, 0.1, 0.15, 0.2, 0.3], args.alpha, args.seed)


if __name__ == "__main__":
    main()

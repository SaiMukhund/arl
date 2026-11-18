# Risk-Aware Jev / Laya Research Lab — Advanced Environment

This project now uses a **sequential, partially observable risk-sensitive
decision benchmark**. It remains a synthetic research simulator and must never
be connected to real financial transactions.

## What changed

The previous toy simulator exposed only a small risk score and treated the
episode outcome as a single terminal lottery. The new environment models a
multi-step decision process with hidden hazards and imperfect evidence.

### Decision problem

The agent chooses among three actions:

| Action | Meaning | Effect |
|---|---|---|
| `execute` | Commit the transaction now | Ends the episode; may succeed, fail ordinarily, or suffer a catastrophic loss |
| `verify` | Independently check recipient/source details | Consumes time and operational cost; reduces recipient mismatch, duplicate and source-integrity hazards |
| `request_confirmation` | Obtain out-of-band confirmation | Consumes time and operational cost; reduces compromised-channel risk |

Checks are not free. The agent has a finite horizon; if it spends all available
steps without executing, the environment forces execution and applies a
deadline penalty. Repeating a check is possible but wasteful.

### Partial observability and latent hazards

Each episode samples hidden factors including recipient mismatch, compromised
communication channel, duplicate request, unreliable source, transaction
amount, reversibility, context volatility, and baseline hazard. The agent sees
noisy evidence signals rather than the latent truth. Hazards can interact, so
multiple weak warning signs can create a disproportionate tail risk.

The observation has 16 features: risk context, amount, urgency, recipient
knowledge, check history, remaining time, recipient/channel/duplicate evidence,
reversibility, anomaly evidence, source reliability, elapsed time, and context
volatility. Hidden ground-truth flags are not exposed.

### Stress regimes

- `mixed`: ordinary mixture used for development/evaluation.
- `benign`: lower prevalence of hidden hazards.
- `adversarial`: higher hazard prevalence and baseline risk.
- `drift`: shifted hazard distribution intended to test robustness.

Use `--eval-regime adversarial` or `--eval-regime drift` to test distribution
shift. The `--eval-shift` scalar remains available as an additional hazard
offset.

## Research architecture

- **Laya**: frozen typed-judgement model, asked for success/catastrophe
  judgements for each action. These are model judgements, not calibrated
  probabilities unless validated.
- **RiskNet**: supervised predictor for eventual success, expected cost,
  catastrophic-event frequency, and cost quantiles.
- **DQN**: independent sequential scalar-reward baseline.
- **Risk-aware selectors**: expected-cost, CVaR, entropic-risk, combined, and
  catastrophe-constrained action selection.

The counterfactual dataset is generated from both fresh and reachable
intermediate states. Each state-action label is estimated with Monte Carlo
rollouts that take the candidate action and then follow a documented,
deadline-aware verification/confirmation fallback. These labels are simulator
oracle data, not external ground truth.

## Install and run

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python -m src.test_environment
python -m src.test_risk_measures
python -m src.laya_smoke_test
```

Quick experiment without Laya:
```bash
python -m src.run_experiments --episodes 1000 --eval-episodes 500 \\
  --train-states 3000 --mc-samples 100 --seeds 1 --without-laya
```

Robustness evaluation on a held-out stress regime:
```bash
python -m src.run_experiments --episodes 3000 --eval-episodes 2000 \\
  --train-states 8000 --mc-samples 160 --seeds 3 --without-laya \\
  --eval-regime adversarial
```

Laya-enabled run (first model initialization may download weights):
```bash
python -m src.run_experiments --episodes 3000 --eval-episodes 2000 \\
  --train-states 8000 --mc-samples 160 --seeds 3
```

Results are written to `results/`: per-seed metrics, policy summary, and plots.

## Important interpretation cautions

1. The simulator is a controlled benchmark, not a validated model of real
   banking, browser, or enterprise workflows.
2. Monte Carlo labels are only as credible as the simulator assumptions.
3. Laya outputs are judgements; assess Brier score, reliability diagrams and
   calibration before interpreting them as probabilities.
4. CVaR estimates are noisy in rare-event settings. Report uncertainty,
   bootstrap confidence intervals and event counts.
5. Tune policy weights and safety thresholds on validation regimes, not the
   final test regime.
6. Compare policies on paired seeds and report success, mean operational cost,
   total cost, catastrophic frequency, tail loss and deadline-forced execution.
7. The current Laya integration is frozen-model inference. It does not fine-tune
   Laya's encoder or attach trainable heads to its internal representation.

## References

- Laya model card: https://huggingface.co/convaiinnovations/laya
- Laya implementation: https://github.com/NandhaKishorM/laya

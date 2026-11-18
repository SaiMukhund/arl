# Risk-aware Laya experiments

A reproducible synthetic benchmark for studying risk-aware action selection with Laya.

## What this project tests

We compare four policies on a synthetic transaction environment:

1. **Success-only**: choose the action with the largest estimated success probability.
2. **Expected-cost**: success probability minus a penalty for expected cost.
3. **CVaR**: success probability minus expected-cost and upper-tail CVaR penalties.
4. **CVaR + safety constraint**: the CVaR policy, but reject actions whose estimated catastrophic-event probability exceeds a threshold.

The environment has three actions: direct transfer, verify recipient, and request confirmation. The probabilities and costs are synthetic and are configurable in `src/config.py`.

The project trains supervised predictors from simulated trajectories:
- success probability (binary classifier),
- expected cost (regressor),
- cost quantiles (quantile regressors),
- catastrophic-event probability (binary classifier).

Laya is an optional success-probability baseline. Its output is queried using the public `Router.predict` API and a `noul` question. Laya's confidence must not be assumed to equal task-success probability: we explicitly calibrate it on an independent validation set using logistic calibration. If Laya is not installed, run all experiments using the local learned success predictor.

## Setup

Python 3.10+ recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install laya                 # optional, for actual Laya inference
```

For GPU use, install a CUDA-compatible PyTorch build first, following the official PyTorch instructions.

## Run

From the project root:

```bash
python -m src.run_experiment --episodes 30000 --seed 7
```

Without Laya:
```bash
python -m src.run_experiment --episodes 30000 --seed 7 --no-laya
```

Quick smoke test:
```bash
python -m src.run_experiment --episodes 3000 --seed 7 --no-laya
```

Outputs are written to `results/`:
- `metrics.csv`
- `action_level_predictions.csv`
- `risk_tradeoff.png`
- `cvar_by_policy.png`
- `summary.txt`

## Laya integration

The adapter in `src/laya_adapter.py` uses:

```python
from laya import Router
router = Router()
result = router.predict(
    state_text,
    {
        "success": {
            "type": "noul",
            "instructions": "Would the proposed action complete the task successfully?"
        }
    }
)
p = result["answers"]["success"]["noul"]
```

The adapter is defensive about response structure and raises a useful error if the installed SDK differs. We evaluate Laya on a per-action state description. Laya is not trained by this project; it is used as a frozen decision model. Fine-tuning Laya or adding neural heads to its internal encoder is a later phase and requires inspecting the installed package/model architecture.

## Methodology and caveats

- The simulator generates counterfactual outcomes by sampling each action repeatedly from the same scenario. This gives action-conditional training labels.
- The learned models are trained on training scenarios and evaluated on held-out scenarios.
- The test scenarios come from a shifted "rare-event" regime with more severe tail outcomes; each selected action is evaluated with 20 stochastic replications per scenario.
- The synthetic costs are deliberately illustrative. They are not estimates of real banking risk.
- CVaR is estimated from predicted conditional cost quantiles. Quantile regression is trained with the pinball loss.
- A finite dataset cannot establish safety against unknown failure modes. The safety gate is evaluated empirically and should be interpreted as a benchmark mechanism, not a real-world guarantee.
- This is research code, not a system for executing financial transactions.

## Suggested extensions

1. Increase the number of rare-event rollouts or use importance sampling.
2. Compare CVaR levels alpha = 0.01, 0.05, 0.10.
3. Add bootstrap ensembles and upper confidence bounds for catastrophic risk.
4. Replace the tabular scenario features with a text encoder and train action-conditioned heads.
5. Add a sequential multi-step environment and compare trajectory-level risk estimates.

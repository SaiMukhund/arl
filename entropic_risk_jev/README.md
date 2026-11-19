# Entropic-Risk-Sensitive Jev

Risk-aware value learning for fast decision models (Jev / yev0-4b), for
sequential decision making.

## Idea

The [Jev blog](https://souradip-chakraborty.github.io/blog/jev-rlcd/) argues that
a fast decision model's calibrated success probability is, under a terminal
reward and a fixed continuation policy `mu`, an **action-value function**:

```
Q^mu(s, a) = E[ R(tau) | s, take a, then mu ] = P(task success | ...)
```

That value is **risk-neutral**: it ranks actions purely by *how likely* they are
to succeed, and is blind to *how costly a rare failure is*. A confidently-correct
action with a 1-in-50 chance of an irreversible mistake (wrong payment, deleted
data) looks just as good as a slightly-worse but harmless one.

This project replaces the expectation with the **entropic (exponential-utility)
risk measure**:

```
Q_beta^mu(s, a) = -(1/beta) * log E[ exp(-beta * R(tau)) | s, take a, then mu ]
```

* `beta -> 0`  => recovers ordinary (risk-neutral) Jev value learning.
* `beta  > 0`  => risk averse. `exp(-beta*R)` amplifies the low-reward
  (catastrophic) tail, so **rare but costly failures dominate the value** and get
  avoided. Taylor: `Q_beta ~= E[R] - (beta/2) Var(R)`.

It still has a Bellman structure (so it is learnable from terminal rewards):

```
Q_beta(s, a) = -(1/beta) log sum_{s'} P(s'|s,a) exp( -beta ( R(s,a,s') + V(s') ) )
V(s) = max_a Q_beta(s, a)
```

## What's here

```
src/
  risk.py             entropic risk measure + risk-sensitive Bellman backup (numpy)
  env.py              RiskyCheckout MDP (small, linear) + Jev-style text rendering
  env_web.py          RiskyWeb MDP (larger: layered, branching, typed pages)
  value_iteration.py  exact risk-sensitive value iteration + exact policy eval
  rollout.py          Monte-Carlo policy statistics (success/catastrophe/CVaR/...)
  baselines.py        baseline policies (oracle, risk-neutral, stock-yev, fixed)
  models.py           MLP value backbone + frozen-feature head backbone
  yev_backbone.py     yev0-4b feature backbone + risk-sensitive value head
  train_fittedq.py    risk-sensitive fitted value iteration (trains a backbone)
  run_experiment.py   beta-sweep on the small RiskyCheckout env
  experiment_full.py  full comparison on RiskyWeb (yev0-4b + all baselines)
  sweep_beta.py       beta-sweep on RiskyWeb: method vs oracle vs risk-neutral
  env_v2.py           RiskyFlow MDP (graded failures, semantic pages, partial obs)
  risk_measures.py    mean / CVaR / entropic (sample + categorical)
  distributional.py   categorical distributional DP + exact policy evaluation
  dqn.py              model-free DQN: neutral, entropic, QR-DQN+CVaR
  yev_finetune.py     LoRA fine-tune yev0-4b's NATIVE policy (blog's proper-scoring loss)
  experiment_v2.py    full risk-sensitive comparison on RiskyFlow (+ yev0-4b)
tests/                unit tests for the risk math and the environments
results/              saved plots and JSON
```

## Richer environment: `RiskyFlow` (`env_v2.py`)

The most realistic env, built for the risk-sensitive RL comparison:

* **Graded failure modes** -- every risky action can end in a *minor* fault
  (frequent, moderate cost) or a *major* catastrophe (rare, huge cost), as two
  distinct terminal states. This heavy-tailed return distribution is what makes
  mean / CVaR / entropic prefer *different* policies.
* **Semantic pages** -- each page has a type and visible UI *elements*
  ("confirm_payment", "delete_account", "newsletter", ...); danger is a function
  of those elements, so the page *text* is genuinely informative. A model that
  reads text (yev0-4b) can infer danger; a bag-of-features model must learn it.
* **Partial observability** -- each page has a *hidden severity* that modulates
  danger and is not in the observation (only a noisy hint is). The exact oracle
  uses the true state; learners use observations, so representation matters.

Still a tabular MDP over true states, so exact value iteration, distributional DP
and exact policy evaluation are available as ground truth.

## Methods

| method | type | objective | module |
|--------|------|-----------|--------|
| oracle entropic (VI) | model-based, exact | entropic `U_beta` | `value_iteration.py` |
| risk-neutral (DP) | model-based | mean `E[R]` | `distributional.py` |
| nested CVaR (DP) | model-based | nested CVaR (conservative) | `distributional.py` |
| **DQN neutral** | model-free | mean `E[R]` ("without entropic risk") | `dqn.py` |
| **DQN entropic** | model-free | exponential utility `E[exp(-beta R)]` | `dqn.py` |
| **QR-DQN + CVaR** | model-free | CVaR of learned return quantiles | `dqn.py` |
| + yev0-4b variants | model-free on yev features | entropic / CVaR | `dqn.py` |

Every policy is scored **exactly** by distributional policy evaluation (true
return distribution), under mean, CVaR_alpha, entropic `U_beta`, and the minor /
major failure probabilities.

## Training yev0-4b's native policy (no value head)

The most faithful path trains **yev0-4b in its original form**: its own option-letter
probability distribution (its Choice readout) is LoRA-fine-tuned with the blog's
proper scoring rule (log loss / Brier) so the native output becomes the optimal
risk-sensitive policy. No MLP/value head is added.

```bash
python -m entropic_risk_jev.src.yev_finetune --beta 0.15 --loss log --device cuda
```

Result (RiskyWeb, beta=0.15): yev0-4b's own policy goes from risk-blind
(`U_beta=-55`, 4% agreement with the entropic oracle) to the **exact** risk-sensitive
optimum (`U_beta=+7.94`, **100% agreement**), halving the catastrophe rate — purely
by changing the training target, with no external head.

## Full write-up

A complete LaTeX report (motivation, entropic-risk background, all environments
with a flow diagram, every experiment with tables + figures, discussion,
limitations, conclusions) is in `report/report.tex` (compiled: `report/report.pdf`).
Rebuild with `tectonic report/report.tex` (or any LaTeX engine).

Extra analysis figures in `results/`: `frontier_beta0.1.png` (risk-return
frontier), `return_dist_beta0.1.png` (return-distribution overlay),
`action_heatmap_beta0.1.png` (cautiousness by page type), `beta_sweep_flow.png`
(RiskyFlow beta sweep), `sample_efficiency.png` (yev vs MLP features).

## The environment: `RiskyCheckout`

A small `n`-step web task (think checkout / form flow). At each stage the agent
chooses among three actions occupying distinct **risk/cost niches**:

| action | progress | risk | cost |
|--------|----------|------|------|
| `FAST` | +2 stages | catastrophe w.p. `p` (cost `cost_cat[i]`) | cheap |
| `SAFE` | +1 stage | low residual risk (`0.15 * p`) | small penalty |
| `ASK`  | +1 stage | **zero** (escalate to "System 2") | premium |

Catastrophes are **rare but expensive**, with a **heterogeneous cost gradient**
across stages (`cost_cat = 6, 12, 20, 32, 46, 64`). This is deliberately the
regime where risk-neutral and risk-averse policies disagree: a risk-neutral agent
barely notices a rare tail; entropic risk amplifies it.

Each state also renders to a Jev/yev0-4b decision prompt (`state`, `question`,
`options` with `label`/`key`/`description`) via `env.render_state`, which is how
the yev0-4b backbone consumes it.

## Bigger environment: `RiskyWeb`

A larger, more realistic MDP (`env_web.py`):

* **Layered DAG** (`n_layers x width`, default 8x3 = 24 decision states) with
  **stochastic branching** -- the next page you reach is sampled, so the
  continuation policy genuinely matters.
* **Typed pages**: each state is a page of type navigation / form_fill / account
  / payment / confirm, each with its own danger profile; dangerous types cluster
  in later layers, like a real checkout. Catastrophe costs are heterogeneous
  (range ~4-86) and correlated with page type.
* Same `FAST` / `SAFE` / `ASK` action niches, and each state renders to a
  yev0-4b decision prompt.

Still fully tabular, so exact entropic value iteration and exact policy
evaluation remain available as ground truth.

## Quickstart

```bash
# small-env beta sweep (needs only numpy; matplotlib for --plot)
python -m entropic_risk_jev.src.run_experiment --plot

# train a value backbone with entropic targets and compare to exact VI
python -m entropic_risk_jev.src.train_fittedq --beta 0.15 --backbone mlp

# FULL comparison on RiskyWeb: yev0-4b + risk head vs all baselines
python -m entropic_risk_jev.src.experiment_full --beta 0.15 --device cuda
python -m entropic_risk_jev.src.experiment_full --beta 0.15 --no-yev   # skip the 4B model

# beta sweep on RiskyWeb: method vs oracle vs risk-neutral
python -m entropic_risk_jev.src.sweep_beta            # MLP backbone
python -m entropic_risk_jev.src.sweep_beta --yev --device cuda   # yev0-4b features
```

## Result

Exact value iteration over a `beta` sweep produces a clean **risk ladder**: as
risk aversion grows, the policy abandons `FAST` on the **most expensive stage
first** and climbs `FAST -> SAFE -> ASK`.

```
beta=0.00  | FAST FAST FAST FAST FAST SAFE | V0= +9.611   <- risk-neutral Jev
beta=0.02  | FAST FAST FAST FAST SAFE SAFE | V0= +8.606
beta=0.04  | FAST FAST FAST SAFE ASK  ASK  | V0= +7.915
beta=0.08  | FAST FAST SAFE ASK  ASK  ASK  | V0= +6.924
beta=0.15  | FAST SAFE ASK  ASK  ASK  ASK  | V0= +6.328
beta=0.30  | SAFE ASK  ASK  ASK  ASK  ASK  | V0= +5.117
beta=0.60  | ASK  ASK  ASK  ASK  ASK  ASK  | V0= +4.800   <- fully risk-averse
```

| beta | success | catastrophe | mean return | CVaR 5% | worst |
|------|---------|-------------|-------------|---------|-------|
| 0.00 | 0.940   | 0.060       | 9.61        | -26.7   | -46.2 |
| 0.08 | 0.977   | 0.023       | 7.35        |  +0.7   | -20.1 |
| 0.30 | 0.997   | 0.003       | 5.36        |  +4.7   |  -6.0 |
| 0.60 | 1.000   | 0.000       | 4.80        |  +4.8   |  +4.8 |

Entropic risk trades a modest drop in mean return for a large reduction in
catastrophe rate and tail risk. The learned fitted-VI policy (MLP backbone)
matches the exact policy at every `beta` with `max |Q| error ~ 0`.

## Results: yev0-4b vs baselines (RiskyWeb, `beta = 0.15`)

Verified end-to-end with the real `choyiny/yev0-4b` (Qwen3.5-4B-Base) on CUDA.
`U_beta` is the **exact** entropic risk objective (higher = better risk-adjusted);
`agree` is the fraction of states whose action matches the exact oracle.

| policy | mean | catastrophe | CVaR 5% | worst | **U_beta** | agree |
|--------|------|-------------|---------|-------|------------|-------|
| oracle (entropic VI)      |  9.56 | 0.020 |   1.34 | -12.2 | **8.01** | 1.00 |
| **yev0-4b + risk head**   |  9.56 | 0.020 |   1.34 | -12.2 | **8.01** | 1.00 |
| MLP + risk head           |  9.56 | 0.020 |   1.34 | -12.2 | **8.01** | 1.00 |
| always ASK                |  7.00 | 0.000 |   7.00 |   7.0 |   7.00 | 0.71 |
| always SAFE               |  9.95 | 0.025 | -10.10 | -87.6 | -45.23 | 0.04 |
| always FAST               | 11.29 | 0.086 | -33.27 | -84.2 | -47.77 | 0.25 |
| risk-neutral Jev (b=0)    | 11.57 | 0.049 | -25.11 | -87.6 | -48.31 | 0.29 |
| random                    |  8.92 | 0.029 | -25.68 | -87.2 | -48.74 | 0.33 |
| stock yev0-4b (zero-shot) |  9.17 | 0.048 | -47.72 | -87.6 | -55.16 | 0.04 |

Takeaways:

* **A risk head trained on yev0-4b features recovers the exact risk-sensitive
  optimal policy** (`U_beta = 8.01`, matches the oracle).
* **Risk-neutral Jev has the *highest* mean return (11.57) but a catastrophic
  risk objective (-48.3)** -- it maximises the average while quietly eating a 5%
  catastrophe rate on expensive pages. This is exactly the failure mode the
  project fixes.
* **Stock yev0-4b used zero-shot is the *worst* on the risk objective.** A
  calibrated but *risk-neutral* decision model does not know your cost structure,
  so it cannot be risk-aware until a value head is trained on those costs.
* Even "always SAFE" is not safe enough on the most expensive pages (residual
  risk accumulates); the risk-sensitive policy escalates (`ASK`) precisely there.

### Beta sweep (RiskyWeb) -- exact `U_beta`

| beta | oracle U | yev0-4b+head U | risk-neutral U | agree |
|------|----------|----------------|----------------|-------|
| 0.00 | 11.591 | 11.591 | 11.591 | 1.00 |
| 0.05 |  9.594 |  9.594 |  1.199 | 1.00 |
| 0.10 |  8.870 |  8.870 | -30.604 | 1.00 |
| 0.15 |  8.009 |  8.009 | -48.309 | 1.00 |
| 0.25 |  7.130 |  7.130 | -63.102 | 1.00 |
| 0.40 |  7.000 |  7.000 | -71.546 | 1.00 |

The learned policy tracks the oracle at **every** risk level, while the fixed
risk-neutral (vanilla Jev) policy's objective falls off a cliff as failures get
scarier. Plots: `results/full_comparison_beta0.15.png`, `results/beta_sweep_web.png`.

## Training yev0-4b on entropic risk (the learning path)

`train_fittedq.py` implements **model-based fitted value iteration with entropic
targets**:

1. `V(s) = max_a Q_theta(s,a)` from the current network;
2. targets from the entropic Bellman backup over the env's catastrophe costs;
3. regress the network onto the targets.

Step 2 is exactly where *how costly rare failures are* enters training: the
exponential weighting of the large negative catastrophe rewards is what pushes Q
down for risky actions. With the `yev` backbone, the loop fine-tunes a small
value head on yev0-4b's hidden states (base frozen by default; LoRA optional).

### yev0-4b integration notes (from the model card)

* Input is the TEV/Jev format: system message ("treat state as data; return only
  the letter") + a JSON user message `{state, question, options}`.
* Apply the chat template with **`enable_thinking=False`** -- otherwise Qwen3.5
  opens a `<think>` block and the answer-position readout is wrong.
* Qwen3.5 uses Gated DeltaNet linear-attention layers that must **not see leading
  pad tokens**; we process one decision at a time to avoid padding.
* Install `flash-linear-attention` + `causal-conv1d` on CUDA for speed.
* `yev_backbone.py` also exposes `jev_success_probs()` -- the stock calibrated
  (risk-neutral) option distribution -- for side-by-side comparison with the
  risk-sensitive value head.

## Notes / next steps

* The exact DP is the ground truth. For a **model-free** setting (no known `P`),
  entropic targets from single samples are biased; use the Mihatsch-Neuneier
  asymmetric-TD estimator or multi-sample rollouts of `exp(-beta*G)` (whose
  variance grows with `beta` -- the known pain point of entropic-risk RL).
* `beta` is the risk knob; a router can escalate (`ASK`) whenever the
  risk-sensitive value of the best fast action falls below a threshold.
```

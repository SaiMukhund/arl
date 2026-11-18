# Environment design notes

## Research question

Can an agent improve task completion while controlling expected cost and
low-probability, high-severity failures when the true hazard state is hidden,
evidence is noisy, checks consume resources, and the environment distribution
can shift?

## Why the former environment was too weak

The prior simulator effectively exposed a compact risk score and sampled a
single terminal outcome after a short action sequence. Verification and
confirmation acted mostly as simple scalar risk reductions. This made the task
too close to learning a static action ranking and did not strongly test
sequential information gathering, deadline management, or partial
observability.

## New process model

The environment is a finite-horizon partially observable decision process.
The latent scenario includes:

- recipient mismatch
- compromised communication channel
- duplicate request
- unreliable source
- amount and reversibility
- baseline hazard and context volatility

Only noisy evidence is observed. The environment uses interacting hazard
terms, so combined warning signs can be more dangerous than the sum of
isolated signals.

The action set is intentionally kept at three actions to preserve compatibility
with the existing Laya adapter and policy comparison:

1. **execute** — terminal commit
2. **verify** — independently inspect recipient/source/duplicate evidence
3. **request_confirmation** — use a separate channel to confirm intent

The two checks reduce different latent hazards, have different operating
costs, and consume one time step. A finite deadline forces execution, so
over-checking is not a free safe policy.

## Counterfactual labels

RiskNet needs action-conditional outcomes at the current state. The dataset
generator samples both initial and reachable intermediate states, then uses
Monte Carlo rollouts. A rollout applies the candidate action and uses a fixed
deadline-aware fallback protocol for any remaining verification/confirmation
step before terminal execution.

This creates a reproducible simulator oracle for supervised training. It is
not external ground truth and does not prove that the simulator's assumptions
match a real workflow.

## Evaluation protocol

Evaluate the same policy set on paired seeds under:
- mixed development distribution
- benign low-hazard distribution
- adversarial high-hazard distribution
- drifted distribution

Keep validation and final test regimes separate when tuning lambda, beta,
entropic-risk parameters, or the catastrophe constraint.

Report at minimum:
- success rate
- total cost and operational/check cost separately
- catastrophic event rate and event count
- empirical upper-tail CVaR
- deadline-forced execution rate
- mean decision steps
- calibration (Brier score / reliability curve) for probability predictions

For rare catastrophic events, a small evaluation run is only a plumbing test.
Use enough episodes and multiple seeds to obtain useful confidence intervals.
A zero-event result in a small sample is not evidence of zero risk.

## Deliberate limitations / next work

- The simulator is synthetic and stylized; domain experts should review hazard
  definitions before any real-world interpretation.
- The counterfactual fallback protocol is a fixed benchmark convention. Future
  work should compare several continuation policies or estimate values with
  explicit multi-step rollouts.
- Laya is integrated as a frozen judgement model; this project does not yet
  fine-tune its encoder or attach new risk heads to its internal representation.
- Current quantile-derived CVaR remains approximate; compare against direct
  empirical CVaR and distributional RL baselines.
- Add a separate hidden-state oracle, belief-state baseline, and robust
  constrained-RL baseline for stronger scientific comparisons.

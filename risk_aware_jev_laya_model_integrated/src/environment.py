"""Sequential, partially observable risk-sensitive transaction benchmark.

Synthetic only. Three actions are deliberately kept stable for compatibility:
0=execute, 1=verify, 2=request_confirmation.
"""
from __future__ import annotations
import numpy as np

ACTION_NAMES = ["execute", "verify", "request_confirmation"]
N_ACTIONS = len(ACTION_NAMES)
# Terminal reward for a successful transaction (reward units == cost units).
SUCCESS_REWARD = 28.0
# 16 observable features; latent hazards are never exposed directly.
OBS_DIM = 16

class TransactionEnv:
    """A finite-horizon POMDP with hidden hazards, action costs and rare losses.

    A non-execution action changes the state and consumes time. Execution ends
    the episode. If the horizon expires, the environment executes automatically.
    `regime` controls held-out stress scenarios; `shift` remains as a legacy
    scalar interface for older experiment scripts.
    """
    def __init__(self, horizon=4, seed=0, shift=0.0, regime="mixed"):
        self.horizon = int(horizon)
        self.rng = np.random.default_rng(seed)
        self.shift = float(shift)
        self.regime = regime
        self._valid_regimes = {"mixed", "benign", "adversarial", "drift"}
        if regime not in self._valid_regimes:
            raise ValueError(f"regime must be one of {sorted(self._valid_regimes)}")

    def reset(self):
        self.t = 0
        self.done = False
        self.total_cost = 0.0
        self.total_operational_cost = 0.0
        self.total_loss = 0.0
        self.verified = False
        self.confirmed = False
        self.executed = False
        self._sample_scenario()
        return self._obs()

    def _sample_scenario(self):
        r = self.rng
        if self.regime == "benign":
            p_bad = 0.08
        elif self.regime == "adversarial":
            p_bad = 0.42
        else:
            p_bad = 0.20
        self.amount = float(r.beta(2.0, 2.4))
        self.urgency = float(r.uniform())
        self.recipient_mismatch = bool(r.random() < (p_bad * 0.65 + 0.025))
        self.channel_compromised = bool(r.random() < (p_bad * 0.42 + 0.012))
        self.duplicate_request = bool(r.random() < (p_bad * 0.28 + 0.018))
        self.source_unreliable = bool(r.random() < (p_bad * 0.32 + 0.025))
        self.reversibility = float(r.beta(2.2, 2.0))  # 0=hard to reverse, 1=easier
        self.context_volatility = float(r.uniform())
        self.base_hazard = float(r.beta(1.8, 7.0))
        if self.regime == "adversarial":
            self.base_hazard = min(0.95, self.base_hazard + 0.16)
        if self.regime == "drift":
            self.base_hazard = min(0.95, self.base_hazard + 0.06)
        self.shifted_hazard = np.clip(self.base_hazard + self.shift, 0.001, 0.98)
        # Noisy evidence: informative, but not perfectly diagnostic.
        self.recipient_signal = float(np.clip(
            0.88 - 0.62*self.recipient_mismatch + self.rng.normal(0, .13), 0, 1))
        self.channel_signal = float(np.clip(
            0.91 - 0.68*self.channel_compromised + self.rng.normal(0, .15), 0, 1))
        self.duplicate_signal = float(np.clip(
            0.08 + 0.78*self.duplicate_request + self.rng.normal(0, .16), 0, 1))
        self.anomaly_signal = float(np.clip(
            .18 + .38*self.recipient_mismatch + .30*self.channel_compromised
            + .25*self.duplicate_request + self.rng.normal(0, .18), 0, 1))
        self.source_signal = float(np.clip(
            .92 - .72*self.source_unreliable + self.rng.normal(0, .14), 0, 1))
        self._initial_risk = self.shifted_hazard

    def _obs(self):
        remaining = max(0.0, (self.horizon-self.t)/max(1, self.horizon))
        known = float(self.recipient_signal > .78)
        hist = float(self.verified) + 2.0*float(self.confirmed)
        return np.asarray([
            self.shifted_hazard, self.amount, self.urgency, known,
            float(self.verified), float(self.confirmed), remaining,
            hist/3.0, self.recipient_signal, self.channel_signal,
            self.duplicate_signal, self.reversibility, self.anomaly_signal,
            self.source_signal, self.t/max(1,self.horizon),
            self.context_volatility
        ], dtype=np.float32)

    def _hazards(self, verified, confirmed):
        # A check is useful only for the uncertainty it actually resolves.
        mismatch = self.recipient_mismatch * (0.16 if verified else 1.0)
        compromised = self.channel_compromised * (0.34 if confirmed else 1.0)
        duplicate = self.duplicate_request * (0.24 if verified else 1.0)
        unreliable = self.source_unreliable * (0.42 if verified else 1.0)
        return mismatch, compromised, duplicate, unreliable

    def outcome_distribution(self, verified=None, confirmed=None):
        v = self.verified if verified is None else bool(verified)
        c = self.confirmed if confirmed is None else bool(confirmed)
        mismatch, compromised, duplicate, unreliable = self._hazards(v, c)
        risk = self.shifted_hazard
        exposure = (0.22 + 0.78*self.amount) * (1.18 - .72*self.reversibility)
        # Correlated hazards interact: multiple warning signs compound risk.
        interaction = mismatch*compromised + duplicate*unreliable
        p_cat = (0.0015 + .095*mismatch + .072*compromised
                 + .050*duplicate + .038*unreliable + .11*interaction)
        p_cat *= (0.42 + 1.35*risk) * exposure
        p_cat *= (1.0 + .30*self.context_volatility)
        p_cat = float(np.clip(p_cat, 0.0002, 0.42))
        p_ordinary = float(np.clip(
            .012 + .10*risk + .045*mismatch + .035*duplicate
            + .025*unreliable, .005, .38))
        p_ordinary = min(p_ordinary, .92-p_cat)
        p_success = float(max(.01, 1.0-p_cat-p_ordinary))
        severity = float(100 + 1900*self.amount*(1.15-.75*self.reversibility)
                         * (0.55 + risk))
        return p_success, p_ordinary, p_cat, severity

    def _finish(self, forced=False):
        ps, po, pc, severity = self.outcome_distribution()
        u = self.rng.random()
        if u < pc:
            catastrophic, success, loss = 1, 0, severity
        elif u < pc+po:
            catastrophic, success, loss = 0, 0, 18.0 + 8*self.amount
        else:
            catastrophic, success, loss = 0, 1, 0.0
        self.total_loss += loss
        self.total_cost += loss
        self.done = True
        self.executed = True
        reward = (SUCCESS_REWARD if success else 0.0) - loss
        if forced:
            reward -= 2.0  # time-overrun penalty is distinct from financial loss
        info = self._info(success, catastrophic, loss, forced, ps, po, pc, severity)
        return self._obs(), float(reward), True, info

    def _info(self, success=0, catastrophic=0, loss=0.0, forced=False,
              ps=None, po=None, pc=None, severity=None):
        return {
            "success": int(success), "catastrophic": int(catastrophic),
            "cost": float(self.total_cost),
            "operational_cost": float(self.total_operational_cost),
            "loss": float(loss), "forced_execution": bool(forced),
            "p_success_oracle": ps, "p_ordinary_oracle": po,
            "p_catastrophe_oracle": pc, "catastrophe_severity": severity,
            "steps": int(self.t), "verified": bool(self.verified),
            "confirmed": bool(self.confirmed), "regime": self.regime
        }

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode finished; call reset().")
        action = int(action)
        if action not in range(N_ACTIONS):
            raise ValueError(f"action must be in [0, {N_ACTIONS-1}]")
        if action == 0:
            return self._finish()
        if action == 1:
            if not self.verified:
                fee = 1.8 + 1.4*self.amount
                self.verified = True
                self.total_operational_cost += fee
                self.total_cost += fee
                # Verification improves recipient/source evidence and hazard.
                self.recipient_signal = float(np.clip(
                    self.recipient_signal + (0.20 if not self.recipient_mismatch else -0.02), 0, 1))
                self.source_signal = float(np.clip(
                    self.source_signal + (0.16 if not self.source_unreliable else -0.03), 0, 1))
                self.shifted_hazard *= .82
                reward = -fee
            else:
                fee = 0.6
                self.total_operational_cost += fee
                self.total_cost += fee
                reward = -fee - 1.0  # redundant checking consumes resources
        else:
            if not self.confirmed:
                fee = 2.8 + 1.0*self.amount
                self.confirmed = True
                self.total_operational_cost += fee
                self.total_cost += fee
                self.channel_signal = float(np.clip(
                    self.channel_signal + (0.22 if not self.channel_compromised else -0.03), 0, 1))
                self.shifted_hazard *= .76
                reward = -fee
            else:
                fee = 0.8
                self.total_operational_cost += fee
                self.total_cost += fee
                reward = -fee - 1.0
        self.t += 1
        if self.t >= self.horizon:
            # The deadline forces a commit; the agent must manage its time budget.
            # Keep this step's check fee in the reward alongside the terminal outcome.
            obs, terminal_reward, done, info = self._finish(forced=True)
            return obs, reward + terminal_reward, done, info
        return self._obs(), float(reward), False, self._info()

    def _simulate_after_action(self, action, rng):
        """Counterfactual: take action, then use a fixed verify/confirm/execute fallback."""
        verified, confirmed = self.verified, self.confirmed
        operational = self.total_operational_cost
        original_hazard = self.shifted_hazard
        steps_left = self.horizon-self.t
        if action == 0:
            ps, po, pc, severity = self.outcome_distribution(verified, confirmed)
        else:
            if action == 1:
                if not verified:
                    verified = True
                    operational += 1.8 + 1.4*self.amount
                    self.shifted_hazard *= .82
                else:
                    operational += 0.6  # redundant verification fee, as in step()
            else:
                if not confirmed:
                    confirmed = True
                    operational += 2.8 + self.amount
                    self.shifted_hazard *= .76
                else:
                    operational += 0.8  # redundant confirmation fee, as in step()
            # The fallback is conservative but deadline-aware: apply the other
            # complementary check only when at least two slots remain.
            if steps_left >= 2 and not (verified and confirmed):
                if not verified:
                    verified = True
                    operational += 1.8 + 1.4*self.amount
                    self.shifted_hazard *= .82
                elif not confirmed:
                    confirmed = True
                    operational += 2.8 + self.amount
                    self.shifted_hazard *= .76
            ps, po, pc, severity = self.outcome_distribution(verified, confirmed)
        u = rng.random()
        cat = (u < pc)
        ordinary = (not cat) and (u < pc+po)
        success = not cat and not ordinary
        loss = severity if cat else (18.0+8*self.amount if ordinary else 0.0)
        total = operational+loss
        self.shifted_hazard = original_hazard  # counterfactuals must not mutate live state
        return success, total, cat

    def sample_action_rollouts(self, obs, action, n, seed):
        """Monte Carlo eventual outcome labels after taking `action` now."""
        rng = np.random.default_rng(seed)
        success = np.zeros(n, np.float32)
        costs = np.zeros(n, np.float32)
        cats = np.zeros(n, np.float32)
        for i in range(n):
            success[i], costs[i], cats[i] = self._simulate_after_action(int(action), rng)
        return success, costs, cats

    def oracle_action_stats(self):
        """Exact immediate-commit oracle diagnostics for current state."""
        out = []
        for a in range(N_ACTIONS):
            s, c, cat = self.sample_action_rollouts(self._obs(), a, 20000, 1234+a)
            out.append({"action": ACTION_NAMES[a], "success": float(s.mean()),
                        "mean_cost": float(c.mean()), "catastrophe": float(cat.mean()),
                        "cvar95": float(np.quantile(c, .95))})
        return out

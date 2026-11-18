from .environment import ACTION_NAMES

class LayaAdapter:
    """Laya typed judgements for success and catastrophe for each action."""
    def __init__(self, preload=True):
        try:
            from laya import Router
        except Exception as exc:
            raise ImportError("Install Laya with `pip install laya`.") from exc
        self.router = Router(preload=preload)

    def predict_actions(self, obs):
        state = {
            "task": "Complete a partially observed transaction. Choose between immediate execution, independent verification, and out-of-band confirmation. Balance task success, operational cost, and rare irreversible loss.",
            "state": {
                "risk_context": float(obs[0]),
                "amount_scale": float(obs[1]),
                "urgency": float(obs[2]),
                "recipient_known": bool(obs[3] > 0.5),
                "verification_done": bool(obs[4] > 0.5),
                "confirmation_done": bool(obs[5] > 0.5),
                "time_remaining_fraction": float(obs[6]),
                "action_history": float(obs[7]),
                "recipient_match_evidence": float(obs[8]),
                "channel_trust_evidence": float(obs[9]),
                "duplicate_request_evidence": float(obs[10]),
                "reversibility": float(obs[11]),
                "anomaly_evidence": float(obs[12]),
                "source_reliability_evidence": float(obs[13]),
                "elapsed_fraction": float(obs[14]),
                "context_volatility": float(obs[15]),
            },
            "assumption": "Verification and confirmation consume one decision step and change what the agent knows and how much risk remains. Execution commits immediately. If the deadline expires, execution is forced.",
        }
        questions = {}
        for i, action in enumerate(ACTION_NAMES):
            questions[f"success_{i}"] = {
                "type": "noul",
                "instructions": (
                    f"If the agent takes the action '{action}' now and then continues "
                    "the task, will the requested transaction eventually succeed?"
                ),
            }
            questions[f"catastrophe_{i}"] = {
                "type": "noul",
                "instructions": (
                    f"If the agent takes the action '{action}' now and then follows the task protocol, will the episode lead to "
                    "a catastrophic outcome such as an irreversible major loss?"
                ),
            }
        result = self.router.predict(state, questions)
        rows = []
        for i, action in enumerate(ACTION_NAMES):
            try:
                ps = float(result["answers"][f"success_{i}"]["noul"])
                pc = float(result["answers"][f"catastrophe_{i}"]["noul"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "Unexpected Laya response. Expected "
                    "result['answers'][question]['noul']; inspect raw result."
                ) from exc
            rows.append({"action": action, "success": min(1., max(0., ps)),
                         "cat": min(1., max(0., pc))})
        return rows, result

    def close(self):
        unload = getattr(self.router, "unload", None)
        if callable(unload):
            unload()

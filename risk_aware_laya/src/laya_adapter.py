class LayaAdapter:
    def __init__(self, model=None):
        try:
            from laya import Router
        except ImportError as e:
            raise ImportError("Laya is optional. Install it with `pip install laya`.") from e
        self.router = Router()
        self.model = model

    def predict_success(self, state_text):
        questions = {
            "success": {
                "type": "noul",
                "instructions": (
                    "If the proposed action is taken in the described situation, "
                    "will the requested task complete successfully? Answer yes only "
                    "if the task completes correctly."
                )
            }
        }
        if self.model is None:
            result = self.router.predict(state_text, questions)
        else:
            result = self.router.predict(state_text, questions, model=self.model)
        try:
            answer = result["answers"]["success"]["noul"]
            return float(answer)
        except (KeyError, TypeError, ValueError) as e:
            raise RuntimeError(
                "Unexpected Laya response schema. Inspect `result` from Router.predict "
                "and update LayaAdapter. Response was: " + repr(result)
            ) from e

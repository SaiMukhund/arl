Laya API used:
- from laya import Router
- router = Router(preload=True)
- result = router.predict(state, questions)
- result['answers'][question_name]['noul']

Source: https://huggingface.co/convaiinnovations/laya

This project asks Laya for task-specific success and catastrophic-outcome
judgements. Validate them against held-out simulator outcomes; do not assume
they are already calibrated to this synthetic environment.

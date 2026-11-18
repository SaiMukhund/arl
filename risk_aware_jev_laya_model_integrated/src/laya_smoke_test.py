from .laya_adapter import LayaAdapter
from .environment import TransactionEnv

if __name__ == "__main__":
    env = TransactionEnv(seed=11)
    obs = env.reset()
    model = LayaAdapter(preload=True)
    rows, raw = model.predict_actions(obs)
    print("Laya action judgements (not yet validated against outcomes):")
    for row in rows:
        print(row)
    print("Routing metadata:", raw.get("routing", {}))
    model.close()

import argparse
import numpy as np
import pandas as pd
from .environment import collect_dataset, FEATURE_NAMES
from .config import ACTIONS

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=30000)
    p.add_argument("--output", default="data/trajectories.csv")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    rows = collect_dataset(args.episodes, rng)
    cols = FEATURE_NAMES + ["action_id", "success", "cost", "catastrophic", "scenario_id"]
    df = pd.DataFrame(rows, columns=cols)
    from pathlib import Path
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Wrote {len(df)} action outcomes to {path}")

if __name__ == "__main__":
    main()

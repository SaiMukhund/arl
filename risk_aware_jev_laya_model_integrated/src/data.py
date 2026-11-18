import numpy as np
from .environment import TransactionEnv, N_ACTIONS

def collect_counterfactual_dataset(n_states=10000, mc_samples=96, seed=0, horizon=4):
    """Collect action-conditional Monte Carlo labels from reachable states.

    States come from a mixture of random exploratory trajectories and fresh
    resets, so the predictor sees partially verified/confirmed states too.
    """
    rng = np.random.default_rng(seed)
    env = TransactionEnv(horizon=horizon, seed=seed)
    X, ys, yc, ycat, yq = [], [], [], [], []
    for i in range(n_states):
        if i % 3 == 0 or i == 0:
            obs = env.reset()
        else:
            if env.done:
                obs = env.reset()
            else:
                # Biased exploration ensures meaningful coverage of both checks.
                action = int(rng.choice(N_ACTIONS, p=[.42, .32, .26]))
                obs, _, done, _ = env.step(action)
                if done:
                    obs = env.reset()
        for action in range(N_ACTIONS):
            s, c, cat = env.sample_action_rollouts(
                obs, action, mc_samples, seed+i*101+action)
            X.append(np.concatenate([obs, np.eye(N_ACTIONS, dtype=np.float32)[action]]))
            ys.append(s.mean())
            yc.append(c.mean())
            ycat.append(cat.mean())
            yq.append(np.quantile(c, [0.50, 0.90, 0.95, 0.99]))
    return tuple(np.asarray(v, dtype=np.float32) for v in (X, ys, yc, ycat, yq))

def collect_dqn_episode(env, agent):
    obs = env.reset()
    done = False
    while not done:
        action = agent.act(obs, explore=True)
        nxt, reward, done, info = env.step(action)
        agent.remember(obs, action, reward, nxt, done)
        agent.learn()
        obs = nxt

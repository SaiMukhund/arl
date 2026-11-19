import argparse, os, random
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .config import Config
from .data import collect_counterfactual_dataset, collect_dqn_episode
from .models import train_risk_net
from .dqn import DQNAgent
from .environment import TransactionEnv, OBS_DIM, N_ACTIONS
from .evaluation import evaluate

def train_dqn(cfg, episodes, seed):
    env=TransactionEnv(cfg.horizon,seed)
    agent=DQNAgent(obs_dim=OBS_DIM,n_actions=N_ACTIONS,
        lr=cfg.dqn_lr,gamma=cfg.dqn_gamma,
        replay_size=cfg.dqn_replay_size,batch_size=cfg.dqn_batch_size,
        warmup=cfg.dqn_warmup,target_update=cfg.dqn_target_update,
        eps_decay_steps=cfg.dqn_epsilon_decay_steps,seed=seed)
    for _ in range(episodes): collect_dqn_episode(env,agent)
    return agent

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--episodes",type=int,default=3000,help="DQN training episodes")
    p.add_argument("--eval-episodes",type=int,default=1500)
    p.add_argument("--train-states",type=int,default=5000)
    p.add_argument("--mc-samples",type=int,default=80)
    p.add_argument("--risk-epochs",type=int,default=15)
    p.add_argument("--seeds",type=int,default=3)
    p.add_argument("--seed",type=int,default=7)
    p.add_argument("--eval-shift",type=float,default=0.0)
    p.add_argument("--eval-regime",choices=["mixed","benign","adversarial","drift"],default="mixed",
                   help="Held-out scenario distribution for robustness evaluation")
    p.add_argument("--without-laya",action="store_true", help="Skip Laya policies")
    args=p.parse_args()
    cfg=Config(); os.makedirs("results",exist_ok=True); rows=[]
    for si in range(args.seeds):
        seed=args.seed+si*101
        np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
        print(f"[{seed}] collecting counterfactual labels")
        X,ys,yc,ycat,yq,ycv,yent=collect_counterfactual_dataset(
            args.train_states,args.mc_samples,seed,cfg.horizon,
            alpha=cfg.alpha,theta=cfg.theta,cost_scale=cfg.cost_scale)
        print(f"[{seed}] training RiskNet")
        model,device=train_risk_net(X,ys,yc,ycat,yq,ycv,yent,hidden=cfg.risk_hidden,
            epochs=args.risk_epochs,batch_size=cfg.risk_batch_size,seed=seed,
            cost_scale=cfg.cost_scale)
        print(f"[{seed}] training DQN")
        dqn=train_dqn(cfg,args.episodes,seed)
        names=["expected_cost","cvar","entropic","cvar_entropic","safe_constraint","dqn","dqn_risk"]
        laya=None
        if not args.without_laya:
            from .laya_adapter import LayaAdapter
            print(f"[{seed}] loading Laya Router; first load may download checkpoint weights")
            laya=LayaAdapter(preload=True)
            names += ["laya","laya_expected_cost","laya_cvar","laya_entropic",
                      "laya_cvar_entropic","laya_safe"]
        for name in names:
            print(f"[{seed}] evaluating {name}")
            row=evaluate(name,args.eval_episodes,seed+9000,cfg,model,device,
                         dqn=dqn,laya=laya,shift=args.eval_shift,
                         regime=args.eval_regime)
            row["seed"]=seed; rows.append(row)
        if laya is not None:
            laya.close()
    df=pd.DataFrame(rows); df.to_csv("results/per_seed_metrics.csv",index=False)
    summary=df.groupby("policy").agg(
        success_rate_mean=("success_rate","mean"),success_rate_std=("success_rate","std"),
        mean_cost_mean=("mean_cost","mean"),mean_cost_std=("mean_cost","std"),
        catastrophic_rate_mean=("catastrophic_rate","mean"),
        catastrophic_rate_std=("catastrophic_rate","std"),
        cvar_5_mean=("cvar_5","mean"),entropic_risk_mean=("entropic_risk","mean"),
        operational_cost_mean=("mean_operational_cost","mean"),
        forced_execution_rate_mean=("forced_execution_rate","mean"),
        mean_steps=("mean_steps","mean"),
        mean_reward_mean=("mean_reward","mean")).reset_index()
    summary.to_csv("results/summary.csv",index=False)
    print(summary.to_string(index=False))
    fig,ax=plt.subplots(figsize=(9,6))
    for _,r in summary.iterrows():
        ax.scatter(r.success_rate_mean,r.catastrophic_rate_mean)
        ax.annotate(r.policy,(r.success_rate_mean,r.catastrophic_rate_mean))
    ax.set(xlabel="Task success rate",ylabel="Catastrophic failure rate",
           title="Success vs catastrophic failure trade-off")
    ax.grid(alpha=.3); fig.tight_layout(); fig.savefig("results/success_vs_catastrophe.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,6))
    ax.bar(summary.policy,summary.cvar_5_mean); ax.set_ylabel("Empirical upper-tail CVaR (worst 5%)")
    ax.tick_params(axis="x",rotation=30); fig.tight_layout()
    fig.savefig("results/cvar_comparison.png",dpi=160); plt.close(fig)
    print("Saved results/summary.csv and plots.")

if __name__=="__main__": main()

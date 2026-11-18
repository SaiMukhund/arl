import numpy as np
from .environment import TransactionEnv, ACTION_NAMES
from .risk_measures import upper_cvar, entropic_risk, entropic_premium, score_actions
from .models import predict_risk

def learned_estimates(model, device, obs):
    rows=[predict_risk(model,device,obs,a) for a in range(3)]
    return (np.array([r["success"] for r in rows]),
            np.array([r["cost"] for r in rows]),
            np.array([r["cvar"] for r in rows]),
            np.array([max(0,r["cvar"]-r["cost"]) for r in rows]),
            np.array([r["cat"] for r in rows]))

def choose(est, cfg, method):
    ps,mc,cv,ep,pcat=est
    scores=score_actions(ps,mc,cv,ep,cfg,method)
    if method=="safe_constraint":
        feasible=pcat<=cfg.safety_epsilon
        if feasible.any():
            scores=np.where(feasible,scores,-1e9)
        else:
            # If every action violates the estimated limit, choose the
            # least-catastrophic estimate rather than silently ignoring safety.
            return int(np.argmin(pcat))
    return int(np.argmax(scores))

def evaluate(name, n, seed, cfg, model, device, dqn=None, laya=None,
             shift=0.0, regime="mixed"):
    env=TransactionEnv(cfg.horizon,seed,shift,regime=regime)
    success=[]; costs=[]; cats=[]; rewards=[]; op_costs=[]; forced=[]; steps=[]
    for _ in range(n):
        obs=env.reset(); done=False; total_reward=0.0; final={}
        while not done:
            if name=="dqn":
                action=dqn.act(obs,explore=False)
            elif name=="dqn_risk":
                qvals=dqn.q_values(obs)
                risk=learned_estimates(model,device,obs)
                # DQN estimates expected scalar return; subtract learned
                # cost/tail penalties as an explicit hybrid ablation.
                ps,mc,cv,ep,pcat=risk
                hybrid=qvals - cfg.lambda_cost*mc - cfg.beta_cvar*cv - cfg.gamma_entropic*ep
                feasible=pcat<=cfg.safety_epsilon
                if feasible.any():
                    hybrid=np.where(feasible,hybrid,-1e9)
                else:
                    action=int(np.argmin(pcat))
                    hybrid=None
                if hybrid is not None:
                    action=int(np.argmax(hybrid))
            elif name.startswith("laya"):
                if laya is None: raise RuntimeError("Laya adapter not provided.")
                laya_rows,_=laya.predict_actions(obs)
                lps=np.asarray([r["success"] for r in laya_rows])
                lcat=np.asarray([r["cat"] for r in laya_rows])
                if name=="laya":
                    action=int(np.argmax(lps))
                else:
                    risk=learned_estimates(model,device,obs)
                    est=(lps,risk[1],risk[2],risk[3],np.maximum(lcat,risk[4]))
                    method={
                        "laya_expected_cost":"expected_cost",
                        "laya_cvar":"cvar",
                        "laya_entropic":"entropic",
                        "laya_cvar_entropic":"cvar_entropic",
                        "laya_safe":"safe_constraint",
                        "laya_risk":"cvar_entropic",
                    }.get(name,"cvar_entropic")
                    action=choose(est,cfg,method)
            else:
                action=choose(learned_estimates(model,device,obs),cfg,name)
            obs,reward,done,final=env.step(action); total_reward+=reward
        success.append(final["success"]); costs.append(final["cost"])
        cats.append(final["catastrophic"]); rewards.append(total_reward)
        op_costs.append(final["operational_cost"])
        forced.append(float(final["forced_execution"]))
        steps.append(final["steps"])
    c=np.asarray(costs)
    return {"policy":name,"success_rate":float(np.mean(success)),
        "mean_cost":float(c.mean()),"catastrophic_rate":float(np.mean(cats)),
        "cvar_5":upper_cvar(c,.05),
        "entropic_risk":entropic_risk(c,cfg.theta,cfg.cost_scale),
        "mean_operational_cost":float(np.mean(op_costs)),
        "forced_execution_rate":float(np.mean(forced)),
        "mean_steps":float(np.mean(steps)),
        "mean_reward":float(np.mean(rewards)),"episodes":n}

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

class RiskNet(nn.Module):
    """Shared encoder with heads for success, cost, catastrophe and tail risk.

    Cost-valued heads (cost, quantiles, cvar, entropic premium) are trained on
    targets divided by `cost_scale`; `predict_risk` maps them back to cost units.
    """
    def __init__(self, input_dim=19, hidden=128):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden), nn.ReLU())
        self.success_head = nn.Linear(hidden, 1)
        self.cost_head = nn.Linear(hidden, 1)
        self.cat_head = nn.Linear(hidden, 1)
        self.quantile_head = nn.Linear(hidden, 4)
        self.cvar_head = nn.Linear(hidden, 1)
        self.entropic_head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.encoder(x)
        sp = torch.nn.functional.softplus
        return {"success": torch.sigmoid(self.success_head(h)).squeeze(-1),
                "cost": sp(self.cost_head(h)).squeeze(-1),
                "cat": torch.sigmoid(self.cat_head(h)).squeeze(-1),
                "quantiles": sp(self.quantile_head(h)),
                "cvar": sp(self.cvar_head(h)).squeeze(-1),
                "entropic_premium": sp(self.entropic_head(h)).squeeze(-1)}

def pinball_loss(pred, target, taus=(0.50, 0.90, 0.95, 0.99)):
    t = torch.as_tensor(taus, dtype=pred.dtype, device=pred.device).view(1, -1)
    err = target-pred
    return torch.maximum(t*err, (t-1)*err).mean()

def train_risk_net(X, ys, yc, ycat, yq, ycv, yent, hidden=128, epochs=18,
                   batch_size=256, seed=0, cost_scale=100.0):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RiskNet(X.shape[1], hidden).to(device)
    model.cost_scale = float(cost_scale)
    s = float(cost_scale)
    ds = TensorDataset(*(torch.tensor(a) for a in
                         (X, ys, yc/s, ycat, yq/s, ycv/s, yent/s)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    bce, huber = nn.BCELoss(), nn.SmoothL1Loss()
    for _ in range(epochs):
        model.train()
        for xb, a, b, c, q, cv, ent in loader:
            xb, a, b, c, q, cv, ent = [t.to(device) for t in (xb, a, b, c, q, cv, ent)]
            out = model(xb)
            loss = bce(out["success"], a) + huber(out["cost"], b)
            loss += bce(out["cat"], c) + pinball_loss(out["quantiles"], q)
            loss += huber(out["cvar"], cv) + huber(out["entropic_premium"], ent)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
    model.eval()
    return model, device

@torch.no_grad()
def predict_risk(model, device, obs, action):
    x = np.concatenate([np.asarray(obs, np.float32),
                        np.eye(3, dtype=np.float32)[int(action)]])[None, :]
    out = model(torch.tensor(x, dtype=torch.float32, device=device))
    s = getattr(model, "cost_scale", 100.0)
    return {"success": float(out["success"].item()),
            "cost": float(out["cost"].item())*s,
            "cat": float(out["cat"].item()),
            "cvar": float(out["cvar"].item())*s,
            "entropic_premium": float(out["entropic_premium"].item())*s,
            "quantiles": out["quantiles"].cpu().numpy()[0]*s}

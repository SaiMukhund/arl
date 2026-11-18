import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

class RiskNet(nn.Module):
    def __init__(self, input_dim=11, hidden=128):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden), nn.ReLU())
        self.success_head = nn.Linear(hidden, 1)
        self.cost_head = nn.Linear(hidden, 1)
        self.cat_head = nn.Linear(hidden, 1)
        self.quantile_head = nn.Linear(hidden, 4)

    def forward(self, x):
        h = self.encoder(x)
        return {"success": torch.sigmoid(self.success_head(h)).squeeze(-1),
                "cost": torch.nn.functional.softplus(self.cost_head(h)).squeeze(-1),
                "cat": torch.sigmoid(self.cat_head(h)).squeeze(-1),
                "quantiles": torch.nn.functional.softplus(self.quantile_head(h))}

def pinball_loss(pred, target, taus=(0.50, 0.90, 0.95, 0.99)):
    t = torch.as_tensor(taus, dtype=pred.dtype, device=pred.device).view(1, -1)
    err = target-pred
    return torch.maximum(t*err, (t-1)*err).mean()

def train_risk_net(X, ys, yc, ycat, yq, hidden=128, epochs=18, batch_size=256, seed=0):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RiskNet(X.shape[1], hidden).to(device)
    ds = TensorDataset(*(torch.tensor(a) for a in (X, ys, yc, ycat, yq)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    bce, huber = nn.BCELoss(), nn.SmoothL1Loss()
    for _ in range(epochs):
        model.train()
        for xb, a, b, c, q in loader:
            xb, a, b, c, q = [t.to(device) for t in (xb, a, b, c, q)]
            out = model(xb)
            loss = bce(out["success"], a) + 0.01*huber(out["cost"], b)
            loss += bce(out["cat"], c) + 0.01*pinball_loss(out["quantiles"], q)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
    model.eval()
    return model, device

@torch.no_grad()
def predict_risk(model, device, obs, action):
    x = np.concatenate([np.asarray(obs, np.float32),
                        np.eye(3, dtype=np.float32)[int(action)]])[None, :]
    out = model(torch.tensor(x, dtype=torch.float32, device=device))
    q = out["quantiles"].cpu().numpy()[0]
    # Approximate worst-5% average using predicted upper quantiles.
    cvar5 = 0.5*q[1] + 0.3*q[2] + 0.2*q[3]
    return {"success": float(out["success"].item()), "cost": float(out["cost"].item()),
            "cat": float(out["cat"].item()), "cvar": float(cvar5),
            "quantiles": q}

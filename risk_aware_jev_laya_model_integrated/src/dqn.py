import random
from collections import deque
import numpy as np
import torch
from torch import nn

class QNetwork(nn.Module):
    def __init__(self, obs_dim=16, n_actions=3, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_actions))
    def forward(self, x): return self.net(x)

class DQNAgent:
    """Basic DQN: replay buffer, target network, epsilon-greedy, Huber loss."""
    def __init__(self, obs_dim=16, n_actions=3, lr=1e-3, gamma=0.97, replay_size=50000,
                 batch_size=128, warmup=500, target_update=250, eps_decay_steps=12000, seed=0):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        self.gamma, self.batch_size = gamma, batch_size
        self.warmup, self.target_update = warmup, target_update
        self.eps_decay_steps, self.steps, self.updates = eps_decay_steps, 0, 0
        self.memory = deque(maxlen=replay_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions
        self.q = QNetwork(obs_dim, n_actions).to(self.device)
        self.target = QNetwork(obs_dim, n_actions).to(self.device)
        self.target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)

    def epsilon(self):
        return max(0.05, 1.0 - 0.95*min(1.0, self.steps/max(1,self.eps_decay_steps)))

    def act(self, obs, explore=True):
        if explore:
            self.steps += 1
            if random.random() < self.epsilon(): return random.randrange(self.n_actions)
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32, device=self.device)[None, :]
            return int(self.q(x).argmax(1).item())

    @torch.no_grad()
    def q_values(self, obs):
        x = torch.tensor(obs, dtype=torch.float32, device=self.device)[None, :]
        return self.q(x).squeeze(0).cpu().numpy()

    def remember(self, s, a, r, ns, done):
        self.memory.append((np.array(s,copy=True), int(a), float(r),
                            np.array(ns,copy=True), float(done)))

    def learn(self):
        if len(self.memory) < max(self.warmup, self.batch_size): return
        batch = random.sample(self.memory, self.batch_size)
        s,a,r,ns,d = zip(*batch)
        s=torch.tensor(np.asarray(s),dtype=torch.float32,device=self.device)
        a=torch.tensor(a,dtype=torch.long,device=self.device)
        r=torch.tensor(r,dtype=torch.float32,device=self.device)
        ns=torch.tensor(np.asarray(ns),dtype=torch.float32,device=self.device)
        d=torch.tensor(d,dtype=torch.float32,device=self.device)
        qsa=self.q(s).gather(1,a[:,None]).squeeze(1)
        with torch.no_grad():
            target=r+self.gamma*(1-d)*self.target(ns).max(1).values
        loss=nn.functional.smooth_l1_loss(qsa,target)
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(),5.0); self.opt.step()
        self.updates += 1
        if self.updates % self.target_update == 0:
            self.target.load_state_dict(self.q.state_dict())

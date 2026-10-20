"""TabDDPM-style Gaussian diffusion over numeric and one-hot encoded columns."""
import math
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

class Denoiser(nn.Module):
    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim + 32, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, dim))
    def forward(self, x, t):
        half = 16
        freqs = torch.exp(torch.arange(half, device=x.device) * (-math.log(10000) / max(half-1, 1)))
        emb = torch.cat((torch.sin(t[:, None] * freqs), torch.cos(t[:, None] * freqs)), dim=1)
        return self.net(torch.cat((x, emb), dim=1))

class TabDDPM:
    def __init__(self, dim, steps=100, hidden_dim=256, device=None):
        self.dim, self.steps = dim, steps
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = Denoiser(dim, hidden_dim).to(self.device)
        self.betas = torch.linspace(1e-4, 0.02, steps, device=self.device)
        self.alphas = 1 - self.betas
        self.abars = torch.cumprod(self.alphas, dim=0)
        self.training_status = 'not trained'
        self.history = []

    def loss(self, x):
        n = x.shape[0]; t = torch.randint(0, self.steps, (n,), device=self.device)
        eps = torch.randn_like(x); abar = self.abars[t].unsqueeze(1)
        noisy = abar.sqrt() * x + (1-abar).sqrt() * eps
        return ((self.model(noisy, t.float() / self.steps) - eps) ** 2).mean()

    def fit(self, data, epochs=20, batch_size=256, lr=1e-3, seed=42, callback=None):
        torch.manual_seed(seed); np.random.seed(seed)
        loader = DataLoader(TensorDataset(torch.as_tensor(data, dtype=torch.float32)), batch_size=batch_size, shuffle=True, drop_last=False)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.training_status = 'training'; self.history=[]
        for epoch in range(epochs):
            losses=[]
            for (x,) in loader:
                x=x.to(self.device); opt.zero_grad(set_to_none=True); loss=self.loss(x); loss.backward(); opt.step(); losses.append(loss.item())
            avg=float(np.mean(losses)); self.history.append(avg)
            if callback: callback(epoch+1, avg)
        self.training_status='completed'; return self.history

    @torch.no_grad()
    def sample(self, n, condition=None, condition_mask=None, seed=None):
        if self.training_status != 'completed': raise RuntimeError('model must be trained before sampling')
        if seed is not None: torch.manual_seed(seed)
        x=torch.randn((n,self.dim), device=self.device)
        for i in reversed(range(self.steps)):
            t=torch.full((n,), i/self.steps, device=self.device)
            eps=self.model(x,t); beta=self.betas[i]; alpha=self.alphas[i]; abar=self.abars[i]
            x=(x-beta/torch.sqrt(1-abar)*eps)/torch.sqrt(alpha)
            if i: x=x+torch.sqrt(beta)*torch.randn_like(x)
            if condition is not None and condition_mask is not None:
                x=torch.where(condition_mask.to(self.device), condition.to(self.device), x)
        return x.cpu().numpy()

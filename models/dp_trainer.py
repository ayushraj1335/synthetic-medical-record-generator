"""DP-SGD training using Opacus PrivacyEngine."""
import torch
from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier

def train_private(diffusion, data, epochs=10, batch_size=256, lr=1e-3, noise_multiplier=None, target_epsilon=8.0, max_grad_norm=1.0, delta=1e-5, seed=42, callback=None):
    """Attach Opacus to the denoiser. The noise prediction loss is per-record."""
    torch.manual_seed(seed)
    model=diffusion.model
    optimizer=torch.optim.AdamW(model.parameters(), lr=lr)
    dataset=torch.utils.data.TensorDataset(torch.as_tensor(data, dtype=torch.float32))
    loader=torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    if noise_multiplier is None:
        noise_multiplier=get_noise_multiplier(target_epsilon=target_epsilon, target_delta=delta,
            sample_rate=min(batch_size/len(dataset),1.0), epochs=epochs, accountant='rdp')
    engine=PrivacyEngine(accountant='rdp')
    model, optimizer, loader=engine.make_private(module=model, optimizer=optimizer, data_loader=loader,
        noise_multiplier=noise_multiplier, max_grad_norm=max_grad_norm, poisson_sampling=True)
    diffusion.model=model
    diffusion.training_status='training'; diffusion.history=[]
    try:
        for epoch in range(epochs):
            losses=[]
            for (x,) in loader:
                x=x.to(diffusion.device); optimizer.zero_grad(set_to_none=True)
                # Poisson sampling may produce an empty batch; still execute a noisy
                # optimizer step so accountant steps match the actual mechanism.
                loss=diffusion.loss(x) if len(x) else sum(p.sum()*0 for p in model.parameters())
                loss.backward(); optimizer.step()
                if len(x): losses.append(float(loss.detach().cpu()))
            avg=sum(losses)/max(1,len(losses)); diffusion.history.append(avg)
            if callback: callback(epoch+1, avg)
        diffusion.training_status='completed'
        return engine
    except Exception:
        diffusion.training_status='failed'
        raise

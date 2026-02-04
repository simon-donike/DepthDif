"""

    This file contains the DDIM sampler class for a diffusion process

"""
import torch
from torch import nn

from ..beta_schedules import *
    
class DDIM_Sampler(nn.Module):
    
    def __init__(self,
                 num_timesteps=100,
                 train_timesteps=1000,
                 clip_sample=True,
                 schedule='linear',
                 beta_start=0.0001,
                 beta_end=0.02,
                 eta=0.0,
                 betas=None,
                ):
        
        super().__init__()
        
        self.num_timesteps = int(num_timesteps)
        self.train_timesteps = int(train_timesteps)
        self.clip_sample = bool(clip_sample)
        self.schedule = schedule
        self.eta = float(eta)
        self.final_alpha_cumprod = torch.tensor(1.0)
        
        if betas is not None:
            betas_tensor = torch.as_tensor(betas, dtype=torch.float32)
            if betas_tensor.ndim != 1:
                raise ValueError("DDIM_Sampler betas must be a 1D tensor.")
            self.train_timesteps = int(betas_tensor.numel())
            self.register_buffer('betas', betas_tensor)
        else:
            self.register_buffer(
                'betas',
                get_beta_schedule(
                    self.schedule,
                    self.train_timesteps,
                    beta_start=beta_start,
                    beta_end=beta_end,
                ),
            )
        self.register_buffer('betas_sqrt',self.betas.sqrt())
        self.register_buffer('alphas',1-self.betas)
        self.register_buffer('alphas_cumprod',torch.cumprod(self.alphas,0))
        self.register_buffer('alphas_cumprod_sqrt',self.alphas_cumprod.sqrt())
        self.register_buffer('alphas_one_minus_cumprod_sqrt',(1-self.alphas_cumprod).sqrt())
        self.register_buffer('alphas_sqrt',self.alphas.sqrt())
        self.register_buffer('alphas_sqrt_recip',1/(self.alphas_sqrt))
        # DDIM inference indices mapped into the training diffusion timeline.
        ddim_steps = torch.linspace(0, self.train_timesteps - 1, self.num_timesteps).round().long()
        self.register_buffer('ddim_train_steps', ddim_steps)
        ddim_prev = torch.cat([torch.tensor([-1], dtype=torch.long), ddim_steps[:-1]], dim=0)
        self.register_buffer('ddim_prev_steps', ddim_prev)
        
    @torch.no_grad()
    def forward(self,*args,**kwargs):   
        return self.step(*args,**kwargs)
    
    @torch.no_grad()
    def step(self, x_t, t, z_t):
        """
            Given approximation of noise z_t in x_t predict x_(t-1)
        """
        assert (t < self.num_timesteps).all()

        b = z_t.shape[0]
        device = z_t.device

        ddim_t = t.long()
        train_t = self.ddim_train_steps[ddim_t]
        train_t_prev = self.ddim_prev_steps[ddim_t]

        alpha_cumprod_t = self.alphas_cumprod[train_t].view(b, 1, 1, 1)
        alpha_cumprod_prev = torch.where(
            train_t_prev >= 0,
            self.alphas_cumprod[train_t_prev],
            self.final_alpha_cumprod.to(device),
        ).view(b, 1, 1, 1)

        # Estimate x_0 from x_t and predicted noise.
        x_0_pred = (x_t - self.alphas_one_minus_cumprod_sqrt[train_t].view(b, 1, 1, 1) * z_t) / alpha_cumprod_t.sqrt()
        if self.clip_sample:
            x_0_pred = torch.clamp(x_0_pred, -1, 1)

        sigma_t = self.eta * self.estimate_std(alpha_cumprod_t, alpha_cumprod_prev)
        dir_xt = torch.clamp(1 - alpha_cumprod_prev - sigma_t ** 2, min=0.0).sqrt() * z_t
        prev_sample = alpha_cumprod_prev.sqrt() * x_0_pred + dir_xt

        if self.eta > 0:
            prev_sample = prev_sample + sigma_t * torch.randn_like(x_t)

        return prev_sample
    
    def estimate_std(self, alpha_cumprod, alpha_cumprod_prev):
        one_minus_alpha_cumprod = 1 - alpha_cumprod
        one_minus_alpha_cumprod_prev = 1 - alpha_cumprod_prev

        var = (one_minus_alpha_cumprod_prev / one_minus_alpha_cumprod) * (1 - alpha_cumprod / alpha_cumprod_prev)

        return var.sqrt()

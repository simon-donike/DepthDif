# Sampling Process Diagnostics
Current validation sampling uses a cosine-guided noise schedule.

Logged diagnostics include:
- intermediate denoising frames  
- MAE vs reverse denoising step (using per-step `x0` prediction)  
- diffusion schedule profiles (`sqrt(alpha_bar_t)`, `sqrt(1-alpha_bar_t)`, `beta_tilde_t`, `log10(SNR+eps)`)  

Observed DDPM tradeoff:
- many early steps remain highly noisy  
- compute is spent on low-visual-information stages  

Potential improvement directions:
- DDIM sampling for faster useful denoising trajectory  
- alternate schedules  
- parameterization choices (`x0` vs `epsilon`)  

Intermediate reconstructions over the denoising path:  
![intermediate_steps](assets/intermediate_steps.png){ width="40%" }

MAE trend across intermediate denoising steps:  
![mae_vs_intermediate](assets/mae_vs_intermediate.png){ width="50%" }

Implemented schedule options in code:
- `linear`  
- `cosine`  
- `quadratic`  
- `sigmoid`  

Example schedule profile image:  
![noise_schedules](assets/noise_schedules.png){ width="85%" }

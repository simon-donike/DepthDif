<p align="center">
  <img src="assets/banner_depthdif.png" width="70%" style="border-radius: 12px;" />
</p>

# Densifying Sparse Ocean Depth Observations
DepthDif explores conditional diffusion for reconstructing dense subsurface ocean temperature fields from sparse, masked observations.

The repository currently supports:
- single-band corrupted-to-clean reconstruction
- EO-conditioned multi-band reconstruction (surface condition + deeper target bands)

## Environment & Dependencies
- Python: **3.12.3**
- Dependencies: `requirements.txt` in repository root
- Install:

```bash
pip install -r requirements.txt
```

## Documentation Map
- [Data](data.md): dataset source, export format, masking pipeline, split behavior
- [Model](model.md): architecture and diffusion conditioning flow
- [Date + Coordination Injection](date-coordination-injection.md): coordinate/date FiLM conditioning details
- [Training](training.md): CLI usage, run outputs, logging, checkpoints
- [Inference](inference.md): script and direct `predict_step` workflows
- [Experiments](experiments.md): qualitative test results
- [Model Settings](settings.md): key config knobs and where they are used
- [Development](development.md): known issues, TODOs, and roadmap
- [API Reference](api.md): auto-generated module reference via `mkdocstrings`

## Sampling Process Diagnostics
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

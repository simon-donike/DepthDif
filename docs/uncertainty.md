# Uncertainty

DepthDif estimates sampling uncertainty by running multiple stochastic reverse
diffusion trajectories for the same condition.

For samples \(\hat{x}^{(1)},\ldots,\hat{x}^{(N)}\), the implementation returns
the population standard deviation (`torch.std(..., unbiased=False)`):

\[
\sigma = \sqrt{\frac{1}{N}\sum_{i=1}^{N}
  \left(\hat{x}^{(i)}-\bar{x}\right)^2}.
\]

This is sample dispersion under the configured sampler and checkpoint. It is not
a calibrated predictive interval and does not include data or model uncertainty
outside those stochastic trajectories.

## Export modes

- Repository global exporters use 20 samples by default and preserve depth
  channels unless `--uncertainty-collapse-depth` is passed.
- The public weekly API uses 20 samples by default and exports one collapsed
  uncertainty raster for its legacy checkpoint.
- `uncertainty_only=True` or `--uncertainty-only` skips the ordinary prediction
  export and runs only the uncertainty ensemble.
- The uncertainty sampler and DDIM step count can be overridden independently
  from the main reconstruction sampler.

```python
from depth_recon import run_week_inference

run_week_inference(
    year=2015,
    iso_week=25,
    export_uncertainty=True,
    uncertainty_num_samples=20,
    uncertainty_sampler="ddim",
    uncertainty_ddim_num_timesteps=50,
)
```

For depth-resolved uncertainty and reliability data used by the spatial
dashboard, use the repository global exporters described in [Inference](inference.md).

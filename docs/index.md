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
- [Experiments](experiments.md): qualitative results and sampling diagnostics
- [Model Settings](settings.md): key config knobs and where they are used
- [Development](development.md): known issues, TODOs, and roadmap
- [API Reference](api.md): auto-generated module reference via `mkdocstrings`

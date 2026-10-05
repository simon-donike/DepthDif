<p align="center">
  <a href="https://pypi.org/project/depth-recon/"><img src="https://img.shields.io/pypi/v/depth-recon?style=for-the-badge&label=PyPI" alt="PyPI version" /></a>
  <img src="https://img.shields.io/badge/python-%3E%3D3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python >= 3.12" />
  <a href="https://depthdif.donike.net/"><img src="https://img.shields.io/badge/docs-online-0b2e4f?style=for-the-badge" alt="Documentation" /></a>
  <a href="https://colab.research.google.com/github/simon-donike/DepthDif/blob/main/Colab_Demo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" /></a>
</p>

<p align="center">
  <img src="docs/assets/branding/banner_depthdif.webp" width="65%" alt="DepthDif" />
</p>

# DepthDif

[![DOI](https://zenodo.org/badge/1148711247.svg)](https://doi.org/10.5281/zenodo.22111273)

DepthDif reconstructs dense subsurface ocean temperature or salinity fields from
sparse EN4/ARGO profiles with conditional diffusion. The maintained repository
workflow conditions on ordered surface SST, SSS, and ADT rasters, together with
sparse profile observations, ocean support, coordinates, and date context.

The downloadable public `depthdif_v1.ckpt` is an older, single-OSTIA model. It is
served by the stable PyPI API but is not architecture-compatible with current
three-surface checkpoints. See the [public inference guide](docs/public-inference-package.md)
before mixing package and repository assets.

## Install

DepthDif requires Python 3.12 or newer. For repository development:

```bash
/work/envs/depth/bin/python -m pip install -r requirements.txt
```

For public inference only:

```bash
python -m pip install depth-recon
```

## Public inference

```python
from depth_recon import run_week_inference

run_dir = run_week_inference(
    year=2015,
    iso_week=25,
    rectangle=(-20.0, 30.0, 10.0, 50.0),
    device="cuda",
)
print(run_dir)
```

This resolves the public config, checkpoint, and land mask; downloads EN4/ARGO
and OSTIA inputs when needed; and exports one ISO-week Wednesday. The default
public grid uses non-overlapping 128-pixel patches. GLORYS is optional and is
only used when ground-truth comparison is requested.

Equivalent commands are available as `depth-recon-infer-week`,
`depth-recon-download-argo`, and `depth-recon-download-ostia`.

## Repository training

For the two-stage mask-aware autoencoder and latent diffusion workflow, see
[the latent training guide](docs/autoencoder.md). It uses a fixed 12-channel
representation and trains the AE on dense fields and sparse observations.

The active dataset is `ArgoGeoTIFFGriddedPatchDataset`. Dense rasters provide
GLORYS targets and SST/SSS/ADT conditioning; a compact Zarr store supplies
depth-aligned sparse EN4/ARGO observations.

```bash
/work/envs/depth/bin/python train.py --scenario temperature
/work/envs/depth/bin/python train.py --scenario salinity
/work/envs/depth/bin/python train.py --scenario joint
```

The scenario resolver keeps fields and model channel counts consistent. The
default local preset trains the observation-supported ambient objective;
the HPC preset instead enables deterministic synthetic targets built from the
fitted monthly, spatial GLORYS-delta prior. Configuration overrides use repeated
`--set section.path=value` arguments.

The maintained diffusion presets also pass depth-wise observation and physical
wet-domain masks (configured with `data.dataset.wet_domain`) into the model. This changes the input-channel contract, so
those presets require fresh checkpoints. An optional climatology-residual mode
is available when a training-year-only climatology artifact is configured; it
does not alter the diffusion loss.

```bash
/work/envs/depth/bin/python train.py \
  --scenario temperature \
  --set training.wandb.run_name=temperature_local
```

Run artifacts are stored below `logs/<timestamp>/`, including checkpoints and
resolved data, model, and training configuration snapshots.
Best checkpoints use full reconstructions of 128 fixed validation patches across
all scenarios, rather than denoising loss. The regular validation loader stays
shuffled. See [checkpoint selection](docs/training.md#checkpoint-selection-from-full-reconstructions)
for the metric, sampling budget and GPU distribution.
An optional [archive-screened EN4 holdout benchmark](docs/training.md#held-out-en4-with-no-historical-archive-candidate)
evaluates up to 32 additional fixed patches once per epoch in its own W&B
namespace, without changing checkpoint selection or shared validation inputs.

## Documentation and validation

- [Quick start](docs/quickstart.md)
- [Training](docs/training.md)
- [Data pipeline](docs/data.md)
- [Inference and evaluation](docs/inference.md)
- [CLI reference](docs/cli.md)
- [Python API](docs/api.md)

Build the documentation and run the complete test suite with:

```bash
ENABLE_MKDOCSTRINGS=true /work/envs/depth/bin/python -m mkdocs build --strict
tests/run_tests.sh
```

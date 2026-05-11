# Public Inference Package
The PyPI package is published as `depth-recon`. It provides a small public
entry point around the DepthDif inference stack so users can run weekly ocean
temperature reconstruction without cloning the repository or preparing a GLORYS
target dataset.

Install it with:

```bash
python -m pip install depth-recon
```

The distribution exposes:
- import package: `depth_recon`
- main Python API: `run_week_inference(...)`
- asset resolver: `resolve_public_inference_assets(...)`
- console scripts: `depth-recon-infer-week`, `depth-recon-download-argo`, and `depth-recon-download-ostia`

The package still depends on the same model code used in this repository. The
wrapper solves the public-runtime problem: it resolves model artifacts, builds
one ISO-week inference grid, downloads or accepts public source files, and
writes the same GeoTIFF-style run directory produced by the repository exporter.

## Package vs Repository Usage
Use the package path when you want inference only:

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

Use the repository path when you are training, changing configs, running
experiments, exporting validation summaries, or packaging production globe
assets. The package installs the public inference helpers and the modules needed
to run the model, but the repository remains the source for development
workflows.

## What Happens During `run_week_inference`
`run_week_inference(...)` selects the public no-GLORYS path by default. The
branch is chosen when `glorys_dir` is omitted.

The default public workflow is:

1. Resolve model artifacts from Hugging Face.
2. Resolve the land-mask GeoTIFF used to create the inference patch grid.
3. Select the ISO-week Wednesday as the target date.
4. Build a land-mask-driven grid of overlapping patches and optionally filter it
   by `rectangle=(lon_min, lat_min, lon_max, lat_max)`.
5. Download EN4/ARGO monthly profile files unless `argo_dir` is supplied.
6. Download the matching OSTIA daily SST file unless `ostia_dir` is supplied or
   `auto_download_ostia=False`.
7. Rasterize ARGO profiles into sparse depth channels for each selected patch.
8. Load OSTIA as the EO surface-conditioning channel, or use an all-zero EO
   channel when OSTIA is disabled.
9. Run `PixelDiffusionConditional.predict_step(...)` in batches.
10. Stitch overlapping patch predictions into depth-level GeoTIFFs and write
    GeoJSON/CSV/YAML metadata.

The public path never requires GLORYS. That matters because GLORYS is the
training target and optional comparison product, but it is not needed to predict
from ARGO and OSTIA inputs.

## Artifact Resolution
By default, package inference resolves these files from
`simon-donike/DepthDif` at revision `main`:

| Artifact | Default filename |
| --- | --- |
| model config | `model_config.yaml` |
| data config | `data_config.yaml` |
| training config | `training_config.yaml` |
| checkpoint | `depthdif_v1.ckpt` |
| land mask | `world_land_mask_glorys_0p1.tif` |

Files are cached under `~/.cache/depthdif` unless `cache_dir` is provided.
Existing cached files are reused. Pass `force_download=True` to refresh them.
If the public data or training YAML files are not present in the Hugging Face
repository, the package writes built-in public defaults into the cache. The
model config and checkpoint must be resolvable.

You can prepare assets without running inference:

```python
from depth_recon import resolve_public_inference_assets

bundle = resolve_public_inference_assets()
print(bundle.assets.model_config)
print(bundle.assets.checkpoint)
print(bundle.land_mask_path)
```

For progress reporting, pass a callback receiving `(event, name, path)`. Events
include `cached`, `downloading`, `downloaded`, `builtin`, and `packaged`.

## Source Data Inputs
Public inference consumes EN4/ARGO profiles and can consume OSTIA SST.

EN4/ARGO downloads use the Met Office EN.4.2.2 yearly profile archive and
extract only the month files touched by the selected seven-day ISO-week window.
The selected target date is always the ISO-week Wednesday, while ARGO profiles
are queried across the configured temporal window around that date.

OSTIA downloads use the Copernicus Marine toolbox. Pass credentials directly:

```python
run_week_inference(
    year=2015,
    iso_week=25,
    copernicus_username="<username>",
    copernicus_token="<api-key>",
)
```

`copernicus_password` is still accepted for older callers. The Copernicus
toolbox accepts the API key through its password option, so `copernicus_token`
is the clearer package-facing name.

To use files you already downloaded, pass the source directories:

```python
run_week_inference(
    year=2015,
    iso_week=25,
    argo_dir="./en4_profiles",
    ostia_dir="./ostia",
    auto_download_argo=False,
    auto_download_ostia=False,
)
```

To run without OSTIA, omit `ostia_dir` and set `auto_download_ostia=False`.
DepthDif will keep the model input contract intact by filling the EO channel
with zeros.

## CLI Usage
The console script runs the same function:

```bash
depth-recon-infer-week \
  --year 2015 \
  --iso-week 25 \
  --rectangle -20 30 10 50 \
  --device cuda
```

Download helpers are available when you want to stage inputs first:

```bash
depth-recon-download-argo --year 2015 --iso-week 25 --output-dir ./en4_profiles
depth-recon-download-ostia --year 2015 --iso-week 25 --output-dir ./ostia
```

The equivalent module command is:

```bash
python -m inference.api infer-week --year 2015 --iso-week 25 --device cuda
```

## Outputs
The default public run directory is:

```text
inference/outputs/depthdif_argo_<YYYYMMDD>/
```

It contains:

| Output | Meaning |
| --- | --- |
| `depthdif_argo_<YYYYMMDD>_prediction_<depth>.tif` | stitched prediction raster for each exported depth level |
| `depthdif_argo_<YYYYMMDD>_argo_points.geojson` | observed ARGO point locations for the selected inference window |
| `depthdif_argo_<YYYYMMDD>_patch_splits.geojson` | selected inference patch polygons |
| `selected_patches.csv` | patch metadata used for the run |
| `run_summary.yaml` | artifact paths, date, grid settings, checkpoint/config paths, and output metadata |

The public path writes prediction rasters only. Ground-truth GLORYS rasters are
available from the repository exporter, or from `run_week_inference(...)` when a
valid `glorys_dir` is supplied.

## Important Options
| Option | Effect |
| --- | --- |
| `rectangle` / `--rectangle` | keeps patches intersecting `(lon_min, lat_min, lon_max, lat_max)`; antimeridian rectangles are supported |
| `device` / `--device` | `auto`, `cpu`, or `cuda` |
| `batch_size` / `--batch-size` | overrides the validation batch-size default |
| `cache_dir` / `--cache-dir` | changes where Hugging Face, EN4, and OSTIA files are cached |
| `checkpoint` / `--checkpoint` | uses a local checkpoint instead of the default public checkpoint |
| `revision` / `--revision` | selects a Hugging Face branch, tag, or commit |
| `min_ocean_fraction` / `--min-ocean-fraction` | controls how much ocean a selected patch must contain |
| `sigma` / `--sigma` | applies export-time Gaussian smoothing to prediction rasters; pass `0` to disable |
| `strict_load` / `--strict-load` | loads checkpoint weights with `strict=True` |
| `force_download` / `--force-download` | refreshes cached artifacts/source files |

## GLORYS-Backed Branch
When `glorys_dir` is supplied, `run_week_inference(...)` uses the repository's
global exporter path instead of the public ARGO-only path. That branch writes
prediction rasters and, by default, matching GLORYS ground-truth rasters. It can
also export sampled full-depth profiles and uses the source directories injected
into a temporary data config under the output root.

Use this branch for local research comparison runs where you have GLORYS,
ARGO/EN4, OSTIA, sea-level, and metadata-cache paths available. Use the default
no-GLORYS branch for public PyPI inference.

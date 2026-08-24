# Synthetic Vertical-Offset Prior

The optional Stage 1 target starts from observed SST and SSS surfaces and adds a
fitted GLORYS depth-minus-surface delta. It is a deterministic initialization
target, not an observation or subsurface truth.

## Schema v2

The committed prior uses schema version 2. For every month, depth, and 10°
latitude/longitude bin, the fitter accumulates GLORYS depth-minus-surface deltas.
Sparse cells shrink toward the global monthly profile, the spatial grid is
Gaussian-smoothed with periodic longitude handling, and the loader bilinearly
samples the result at each patch pixel.

For field \(v\) and depth \(z\):

\[
\hat{v}(z,y,x) = v_{surface}(y,x) +
\Delta v_{month,z}(y,x).
\]

The surface offset is exactly zero. Sparse EN4/ARGO observations overwrite the
synthetic values at their exact grid cells and depth levels. At depths without
adequate fitted support, the deepest supported offset is carried downward while
the supervision confidence decays with the configured extrapolation half-life.

## Committed artifact provenance

`vertical_offset_prior.npz` records its own metadata. The current artifact was
fitted from 2000 through 2024 source coverage while excluding 2016, using:

- monthly statistics and 10° spatial bins;
- Gaussian smoothing sigma of one grid cell;
- shrinkage scale of 4096 contributing pixels;
- 1000 m extrapolation half-life;
- at most 4000 selected patches, with 2202 available after filtering;
- seven-day surface aggregation windows;
- a maximum supervised depth of about 5727.9 m.

The fitter rejects any source window touching the held-out year, preventing the
2016 validation period from contributing to fitted statistics.

## Fit and inspect

```bash
/work/envs/depth/bin/python -m depth_recon.data.synthetic_dataset_creation.fit_vertical_offset_prior \
  --geotiff-root-dir /work/data/OceanVariableReconstruction \
  --output-path /work/data/OceanVariableReconstruction/priors/vertical_offset_prior.npz \
  --start-year 2000 --end-year 2024 --exclude-year 2016 \
  --spatial-bin-size-deg 10 --smoothing-sigma-cells 1 \
  --shrinkage-pixels 4096 --extrapolation-half-life-m 1000 \
  --max-patches 4000 --overwrite
```

```bash
/work/envs/depth/bin/python -m depth_recon.data.synthetic_dataset_creation.plot_vertical_offset_examples \
  --geotiff-root-dir /work/data/OceanVariableReconstruction \
  --statistics-path /work/data/OceanVariableReconstruction/priors/vertical_offset_prior.npz \
  --validation-year 2016
```

Enable the target with `data.dataset.synthetic_target.enabled=true` and point
`statistics_path` at the artifact. The loader requires the ordered
`[sst, sss, adt]` surface contract. Schema v1 scalar artifacts remain readable
for artifact compatibility, but they are not the maintained fitting workflow.

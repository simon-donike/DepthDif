# Vertical-offset pretraining

Stage 1 uses the deterministic targets

\[
T_z(x,y)=SST_{obs}(x,y)+\overline{T_z^{GLORYS}-T_0^{GLORYS}},
\]

\[
S_z(x,y)=SSS_{obs}(x,y)+\overline{S_z^{GLORYS}-S_0^{GLORYS}}.
\]

There is one scalar temperature and salinity offset for each of the 50 depths.
The first offsets are forced to zero. Thus the complete surface pattern—phase,
fronts, gradients, and frequencies—is copied unchanged to every depth.

Fit the coefficients with:

```bash
/work/envs/depth/bin/python -m depth_recon.data.synthetic_dataset_creation.fit_vertical_offset_prior \
  --geotiff-root-dir /work/data/OceanVariableReconstruction \
  --output-path /work/data/OceanVariableReconstruction/priors/vertical_offset_prior.npz \
  --metadata-cache-dir /work/data/OceanVariableReconstruction/depthdif_cache \
  --start-year 2000 --end-year 2024 --exclude-year 2018 \
  --tile-size 128 --patch-stride 128 --max-land-fraction 0.30 \
  --max-patches 4000 --max-supervised-depth-m 1000 \
  --random-seed 7 --overwrite --no-progress
```

The fitter rejects held-out years and surface-composite windows touching them.
The artifact contains only depth offsets, coverage weights, the depth axis, and
provenance.

For Stage 1, enable `data.dataset.pretraining_prior.enabled`, use
`priors/vertical_offset_prior.npz`, and disable ambient occlusion. Sparse ARGO
values remain exact overrides. This is a surface-duplication curriculum, not a
claim that observed fronts truly persist through the water column.

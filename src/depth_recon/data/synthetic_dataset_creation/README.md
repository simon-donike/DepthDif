# Smooth GLORYS-Delta Pretraining Targets

The synthetic Stage 1 target retains the observed EO surface exactly and adds an
offline-fitted GLORYS climatological depth delta:

```text
temperature(z,y,x) = observed_SST(y,x) + smooth_GLORYS_delta[month,z,y,x]
salinity(z,y,x) = observed_SSS(y,x) + smooth_GLORYS_delta[month,z,y,x]
```

Schema v2 fits the delta on a monthly 10° grid, shrinks sparse cells toward
the global monthly profile, smooths the grid with periodic longitude handling,
and bilinearly samples it at patch pixels. The loader never retrieves paired or
same-date GLORYS for this synthetic target. Sparse ARGO values still override
the target at their exact cells and levels.

Unsupported abyssal levels carry the deepest supported GLORYS delta while their
supervision confidence decays with depth. Surface offsets are exactly zero.
These targets remain a controlled initialization objective, not subsurface
truth. Schema-v1 scalar artifacts can still be loaded for compatibility, but the
v2 fitter is the maintained workflow.

```bash
/work/envs/depth/bin/python -m depth_recon.data.synthetic_dataset_creation.fit_vertical_offset_prior \
  --geotiff-root-dir /work/data/OceanVariableReconstruction \
  --output-path /work/data/OceanVariableReconstruction/priors/vertical_offset_prior.npz \
  --start-year 2000 --end-year 2024 --exclude-year 2016 \
  --spatial-bin-size-deg 10 --smoothing-sigma-cells 1 \
  --shrinkage-pixels 4096 --extrapolation-half-life-m 1000 \
  --max-patches 4000 --overwrite
```

The fitter rejects any source window that touches the excluded year.

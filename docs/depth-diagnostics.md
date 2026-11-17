# Fixed validation depth diagnostics

`evaluate_depth_baselines` scores several predictions on the same validation
patches and reports MAE, RMSE, signed bias, and valid support independently for
each depth. Configured evaluations include physical depth coordinates, sampled
indices, and the seed in their report. Invalid GLORYS cells are excluded;
configured evaluations fail if predictions are nonfinite on valid targets.
The normal validation dataloader remains shuffled; this command creates a
separate deterministic `Subset` using `--seed`.

For already exported arrays:

```bash
/work/envs/depth/bin/python -m depth_recon.scripts.evaluate_depth_baselines \
  --target-npz target.npz --mask-npz target_mask.npz \
  --method climatology=climatology.npz --method unet=unet.npz \
  --output-json depth_metrics.json
```

For configured checkpoints, use `NAME|CONFIG|CHECKPOINT` and the evaluator
builds the configured validation dataset and model without training:

```bash
/work/envs/depth/bin/python -m depth_recon.scripts.evaluate_depth_baselines \
  --configured-method 'unet|unet_training.yaml|unet.ckpt' \
  --configured-method 'depthdif|diffusion_training.yaml|depthdif.ckpt' \
  --sample-count 32 --seed 7 --batch-size 4 --device cuda --variable temperature \
  --output-json depth_metrics.json --output-npz depth_arrays.npz
```

The configured mode also scores the dataset's normalized climatology channel
when present. All methods use the same sampled validation indices and output
temperature metrics in degrees Celsius. Use `--variable salinity` for salinity
models or the salinity output of a joint model. Supply the saved training
super-config corresponding to each checkpoint; weights load strictly. Target
values and masks must match across methods. This command performs inference only.

Fit the optional background from the training years, explicitly excluding the
validation year:

```bash
/work/envs/depth/bin/python -m depth_recon.data.fit_climatology \
  --root /work/data/OceanVariableReconstruction \
  --output /absolute/path/to/climatology.npz --val-year 2016 \
  --spatial-stride 4 --field both
```

This streams dates into monthly spatial means. `--spatial-stride` reduces the
stored spatial resolution; nearest-neighbour sampling restores patch dimensions.
An unobserved monthly cell falls back to the same cell's annual training mean,
then the training mean at that depth. Entirely unsupported depths retain zero
observation counts and an explicitly recorded placeholder to preserve channel
alignment. The loader rejects valid targets at those depths, so placeholders
cannot silently become supervised climatology predictions.
Source dates, grid, depth coordinates, and excluded year are stored with the
artifact and validated when the dataset opens it. Temperature and salinity
artifacts are supported individually or together. Full-resolution fitting and
large joint artifacts require more memory; the fitter does not train a model.

For a fresh diffusion model, add:

```yaml
data:
  dataset:
    wet_domain:
      enabled: true
      reference_date: null  # earliest GLORYS reference mask; no target values used
    climatology:
      enabled: true
      path: /absolute/path/to/climatology.npz
model:
  climatology_residual: true
```

The model subtracts the background in the existing normalized units, conditions
on it, and adds it back to its predictions. There is no per-depth rescaling or
new loss term. Physical wet masks, observation masks, and supervised target masks
remain distinct. The nine-profile plot now uses the dataset's actual depths;
without physical coordinates, it explicitly labels the axis as depth bands.

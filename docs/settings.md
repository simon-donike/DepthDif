# Configuration Settings

Pixel workflows use one super-config containing `data`, `model`, `training`, and,
for inference, `inference`. The scenario resolver then derives fields and channel
counts. The commented YAML files are the authoritative reference for every key;
this page records the maintained presets and stable ownership of settings.

## Active presets

| Preset | Target/objective | Runtime defaults |
| --- | --- | --- |
| `training_super_config.yaml` | Ambient occlusion over all eligible patches; synthetic target and regional row sampling off. | 100 epochs, 2 devices, `strategy: auto`, W&B online, train batch 32/workers 2. |
| `training_super_config_standard.yaml` | Explicit copy of the standard local preset. | Same resource and objective defaults as the local preset. |
| `training_super_config_hpc.yaml` | Dense deterministic `synthetic_target`; ambient and hard-region sampling off. | 10,000-epoch ceiling, `devices: auto`, DDP, W&B offline, train batch 96/workers 48. |
| `training_super_config_spacehpc_glorys.yaml` | Direct paired-GLORYS supervision over all eligible patches; synthetic, ambient, and regional row sampling off. | 10,000-epoch ceiling, `devices: auto`, DDP, W&B offline, train batch 96/workers 48. |
| `inference_super_config.yaml` | No synthetic target or ambient objective; DDPM reconstruction default. | Grid stride 96, minimum ocean fraction 0.05, batch 64/workers 6. |

All pixel presets provide one scenario-derived surface channel to the model: SST
for temperature (and joint) runs or SSS for salinity runs. Training keeps patches
without ARGO profiles, retains ARGO observations regardless of QC flags, and does
not apply hard/easy regional row sampling. Coordinate conditioning, EMA, the
1,000-step training diffusion schedule, validation year 2016, 100-step DDIM
validation, and shuffled validation loaders remain configured.

## Scenario-derived contract

| Scenario | Generated channels | Condition channels | Fields |
| --- | ---: | ---: | --- |
| `temperature` | 50 | 53 | Temperature only; SST surface input. |
| `salinity` | 50 | 53 | Salinity only; SSS surface input. |
| `joint` | 100 | 103 | Temperature followed by salinity; SST surface input. |

The three non-generated condition channels are the scenario-derived dense surface,
sparse observation support, and the associated mask channel used by the pixel contract.
Do not set channel counts independently of the scenario resolver.

## Key ownership

- `data.dataset`: root paths, fields, surface sources, patch grid, selection,
  `synthetic_target`, and `finetune_sampling`.
- `data.split`: train/validation year policy. The maintained holdout year is 2016.
- `data.dataloader`: shared dataset-construction loader settings.
- `model`: architecture, checkpoint loading, EMA, ambient objective, coastal
  weighting, auxiliary losses, coordinate/date conditioning, and diffusion.
- `training.trainer`: Lightning epoch, accelerator, device, precision, strategy,
  validation, and logging controls.
- `training.dataloader`: training/validation batch and worker settings. These take
  precedence where the datamodule reads the training-specific section.
- `training.validation_sampling`: validation sampler and reconstruction cadence.
- `training.en4_candidate_eval`: optional patch-first EN4 callback; use
  `min_input_profiles` to set the post-holdout density floor and `image_depths_m`
  to choose full-reconstruction figure depths.
- `training.hard_region_eval`: optional hard-region callback. Both validation
  callbacks are enabled in current pixel presets.
- `inference.sampling`: reconstruction sampler overrides.
- `inference.grid`: stitched export stride and ocean-coverage filter.
- `inference.dataloader`: inference batch, workers, and prefetch settings.

## Important defaults

The local preset enables `finetune_sampling` with `hard_fraction: 0.5` for both
`train` and `val`, including the configured land-filter relaxation. Its polygons
are provisional hand-authored regions. It also enables ambient occlusion with a
0.25 observation-drop probability. Coastal loss is disabled.

Auxiliary timestep weighting is enabled locally, but all optional auxiliary loss
terms are disabled. The weighting therefore changes nothing until at least one
auxiliary term is enabled. Feature Gram remains reserved: enabling it raises
`NotImplementedError`.

## Overrides and snapshots

Use `--config` to select a super-config and repeated `--set` arguments for
intentional overrides:

```bash
/work/envs/depth/bin/python train.py \
  --config src/depth_recon/configs/px_space/training_super_config.yaml \
  --scenario temperature \
  --set training.trainer.max_epochs=2
```

Training stores the original super-config and resolved effective data, model,
and training snapshots beside checkpoints. Use those snapshots—not present-day
defaults—to reproduce an existing run.

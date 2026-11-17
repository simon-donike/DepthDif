"""Fixed autoencoder evaluation and training-only latent calibration."""

from pathlib import Path
from typing import Any, Iterable

import pytorch_lightning as pl
import torch
from torch.utils.data import default_collate

from depth_recon.configs.config_resolver_pixel import FULL_RECONSTRUCTION_MONITOR
from depth_recon.utils.normalizations import temperature_normalize, salinity_normalize
from depth_recon.utils.reconstruction_validation import (
    FullReconstructionValidation,
    _evaluation_rng,
)
from depth_recon.utils.validation_denoise import log_wandb_depth_errors

from .Autoencoder import DepthBandAutoencoderLightning


class AutoencoderReconstructionValidation(FullReconstructionValidation):
    """Score dense compression separately from sparse-input and held-profile errors."""

    @torch.no_grad()
    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: DepthBandAutoencoderLightning
    ) -> None:
        """Pool depth errors on fixed patches without advancing training randomness."""
        if trainer.sanity_checking:
            return
        self._select_patches(trainer)
        fields = pl_module.output_fields
        depth_count = pl_module.model.in_channels // len(fields)
        names = ("dense", "sparse_dense", "observed", "held_profiles")
        statistics = {
            name: {
                field: torch.zeros(
                    3, depth_count, dtype=torch.float64, device=pl_module.device
                )
                for field in fields
            }
            for name in names
        }
        for batch_index, indices in self.local_batches(
            trainer.global_rank, trainer.world_size
        ):
            with _evaluation_rng(pl_module.device, self.seed + batch_index):
                batch = default_collate([self.dataset[index] for index in indices])
                batch = pl_module.transfer_batch_to_device(batch, pl_module.device, 0)
                with trainer.precision_plugin.forward_context():
                    data = pl_module.physical_batch(batch)
                    cases = pl_module.reconstruction_cases(batch)
                    observed = data["x_valid_mask"]
                    held = torch.zeros_like(observed[:, :1])
                    # Hide entire horizontal profiles across every field/depth.
                    # Guarantee at least one held and one visible profile when possible.
                    for sample in range(observed.shape[0]):
                        positions = (
                            observed[sample].any(dim=0).flatten().nonzero().flatten()
                        )
                        if positions.numel() > 1:
                            count = min(
                                positions.numel() - 1,
                                max(
                                    1,
                                    round(
                                        positions.numel() * pl_module.holdout_fraction
                                    ),
                                ),
                            )
                            selected = positions[
                                torch.randperm(
                                    positions.numel(), device=positions.device
                                )[:count]
                            ]
                            held[sample].view(-1)[selected] = True
                    hidden = observed & held
                    visible = observed & ~held
                    cases["held_profiles"] = (
                        pl_module.model(data["x"], visible, data["wet_mask"]),
                        data["x"],
                        hidden,
                    )
                for name, (prediction, target, mask) in cases.items():
                    if "background" in data:
                        prediction, target = (
                            prediction + data["background"],
                            target + data["background"],
                        )
                    for field, pred, truth, valid in zip(
                        fields,
                        prediction.chunk(len(fields), 1),
                        target.chunk(len(fields), 1),
                        mask.chunk(len(fields), 1),
                    ):
                        target_key, mask_key = self._target_keys(field)
                        denorm = (
                            temperature_normalize
                            if field == "temperature"
                            else salinity_normalize
                        )
                        statistics[name][field] += self._batch_statistics(
                            {target_key: truth, mask_key: valid},
                            {
                                f"y_hat_{field}_denorm": denorm(
                                    mode="denorm", tensor=pred.float()
                                )
                            },
                            field,
                            single_field=len(fields) == 1,
                        )
        metrics = {}
        for name, field_stats in statistics.items():
            for field in fields:
                field_stats[field] = trainer.strategy.reduce(
                    field_stats[field], reduce_op="sum"
                )
                metrics[f"val/ae/{name}/{field}_valid_values"] = field_stats[field][
                    2
                ].sum()
            # Real sparse patches can have no held profiles. Report support counts
            # and skip their error metric rather than inventing a zero error.
            supported = {
                field: stats
                for field, stats in field_stats.items()
                if (stats[2] > 0).any()
            }
            if not supported:
                if name == "dense":
                    raise ValueError(
                        "Fixed AE validation subset has no valid dense targets."
                    )
                continue
            for key, value in self._metrics(supported).items():
                if key == FULL_RECONSTRUCTION_MONITOR:
                    if name == "dense":
                        metrics[key] = value
                else:
                    metrics[
                        key.replace("val/full_reconstruction/", f"val/ae/{name}/")
                    ] = value
            if trainer.is_global_zero:
                for field, stats in supported.items():
                    dataset = self.dataset
                    while hasattr(dataset, "dataset"):
                        dataset = dataset.dataset
                    log_wandb_depth_errors(
                        logger=trainer.logger,
                        statistics={"Prediction": stats},
                        depth_axis_m=getattr(dataset, "depth_axis_m", None),
                        prefix=f"val/ae/{name}",
                        image_key=f"{field}_error_by_depth",
                        title=f"AE {name}: {field}",
                    )
        pl_module.log_dict(
            metrics, on_step=False, on_epoch=True, sync_dist=False, batch_size=1
        )


@torch.no_grad()
def export_calibrated_autoencoder(
    model: DepthBandAutoencoderLightning,
    batches: Iterable[dict[str, Any]],
    output_path: str | Path,
    *,
    max_batches: int = 32,
) -> None:
    """Export frozen AE weights and latent moments fitted only on training batches."""
    if max_batches < 1:
        raise ValueError("Calibration max_batches must be positive.")
    model.eval()
    sums = torch.zeros(
        model.model.latent_channels, dtype=torch.float64, device=model.device
    )
    squares = torch.zeros_like(sums)
    count = 0
    with _evaluation_rng(model.device, 7):
        for index, batch in enumerate(batches):
            if index >= max_batches:
                break
            batch = model.transfer_batch_to_device(batch, model.device, 0)
            data = model.physical_batch(batch)
            encoded = model.model.encode(
                data["y"], data["y_valid_mask"], data["wet_mask"]
            ).double()
            support = data["y_valid_mask"].any(dim=1, keepdim=True)
            values = torch.where(support, encoded, 0.0)
            sums += values.sum(dim=(0, 2, 3))
            squares += values.square().sum(dim=(0, 2, 3))
            count += int(support.sum())
    if count == 0:
        raise ValueError("No valid training support for latent calibration.")
    mean = sums / count
    std = (squares / count - mean.square()).clamp_min(1e-8).sqrt()
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise ValueError("Nonfinite training-set latent statistics.")
    model.model.latent_mean.copy_(mean.reshape(1, -1, 1, 1))
    model.model.latent_std.copy_(std.reshape(1, -1, 1, 1))
    model.model.latent_calibrated.fill_(True)
    torch.save(
        {
            "state_dict": {
                key: value.cpu() for key, value in model.model.state_dict().items()
            },
            "ae_contract": model.model.contract(),
            "calibration": {
                "split": "train",
                "max_batches": max_batches,
                "valid_columns": count,
            },
        },
        output_path,
    )

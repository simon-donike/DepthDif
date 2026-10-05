"""Tests for isolated, bounded annual EN4 archive holdouts."""

from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from depth_recon.data.dataset_argo_geotiff_gridded import ArgoGeoTIFFGriddedPatchDataset
from depth_recon.inference.export_paper_metrics import (
    load_dataset_context,
    load_en4_candidate_profiles,
)
from depth_recon.utils.en4_archive_holdout import (
    AUDIT_STATUS,
    PREFIX,
    EN4ArchiveHoldoutValidation,
)
from depth_recon.utils.normalizations import salinity_normalize, temperature_normalize
from depth_recon.utils.reconstruction_validation import FullReconstructionValidation
from tests.test_argo_geotiff_gridded_dataset import _make_geotiff_dataset
from train import build_en4_archive_holdout_callback


class _ProfileStore:
    """Provide raw profiles and level QC independently of rasterized model inputs."""

    depth_axis_m = np.array([0.0, 100.0, 500.0])
    accepted_qc_flags = (1, 2)

    def load_temperature_profiles(self, indices):
        """Return temperature observations in physical units."""
        return np.full((len(indices), 3), 10.0, dtype=np.float32)

    def load_salinity_profiles(self, indices):
        """Return salinity observations in physical units."""
        return np.full((len(indices), 3), 30.0, dtype=np.float32)

    def _quality_mask_for_variable(self, variable, *, indices):
        """Reject one depth for alternating profiles to test pooled support."""
        mask = np.ones((len(indices), 3), dtype=bool)
        mask[np.asarray(indices) % 2 == 0, 2] = False
        return mask


class _Dataset(Dataset):
    """Overlapping patches on three dates, returning cached tensors to expose mutation."""

    tile_size = 4
    depth_axis_m = _ProfileStore.depth_axis_m

    def __init__(self):
        """Build dense conditioning and known-error reference fields."""
        self.argo_store = _ProfileStore()
        self.heldout_argo_location_keys = set()
        self._rows = pd.DataFrame(
            [
                {"date": date, "grid_y0": 0, "grid_x0": x0}
                for date in (20160108, 20160408, 20160908)
                for x0 in (0, 1, 2)
            ]
        )
        self.samples = []
        records = []
        for index, row in self._rows.iterrows():
            sample = {"sample_id": index, "land_mask": torch.ones(1, 4, 4)}
            for suffix, normalize, value in (
                ("", temperature_normalize, 10.0),
                ("_salinity", salinity_normalize, 30.0),
            ):
                sample[f"x{suffix}"] = normalize(
                    mode="norm", tensor=torch.full((3, 4, 4), value)
                )
                sample[f"x{suffix}_valid_mask"] = torch.ones(3, 4, 4, dtype=torch.bool)
                sample[f"x{suffix}_valid_mask_1d"] = torch.ones(
                    1, 4, 4, dtype=torch.bool
                )
                sample[f"y{suffix}"] = normalize(
                    mode="norm", tensor=torch.full((3, 4, 4), value + 2)
                )
                sample[f"y{suffix}_valid_mask"] = torch.ones(3, 4, 4, dtype=torch.bool)
            self.samples.append(sample)
        for date in self._rows["date"].unique():
            # The repeated location represents distinct colocated source profiles.
            for yy, xx in [(1, 1), (1, 1), (1, 2), (2, 2), (2, 3), (1, 4)]:
                index = len(records)
                records.append(
                    dict(
                        date=int(date),
                        grid_row=yy,
                        grid_col=xx,
                        profile_index=index,
                        lat=float(yy),
                        lon=float(xx),
                        profile_source_file=f"en4_{date}.nc",
                        source_profile_idx=index,
                        temperature_valid_depth_count=3,
                        salinity_valid_depth_count=3,
                    )
                )
        self.candidates = pd.DataFrame(records)
        self.candidates.attrs.update(
            audit_status_filter=AUDIT_STATUS,
            validation_year=2016,
            quality_filter_enabled=True,
        )

    def __len__(self):
        """Return patch count."""
        return len(self.samples)

    def __getitem__(self, index):
        """Return persistent tensors so leaked edits remain detectable."""
        return self.samples[index]


class _Model(pl.LightningModule):
    """Predict fields one unit closer to the observations than GLORYS."""

    def __init__(self):
        """Track every inferred patch and received conditioning mask."""
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.output_fields = ("temperature", "salinity")
        self.seen = []
        self.masks = []

    def validation_step(self, batch, batch_idx):
        """Keep ordinary validation cheap."""
        return None

    def predict_step(self, batch, batch_idx):
        """Consume RNGs and return exact known-error physical predictions."""
        self.seen.extend(batch["sample_id"].tolist())
        self.masks.append(batch["x_valid_mask"].clone())
        random.random()
        np.random.random()
        torch.rand(1)
        return {
            "y_hat_temperature_denorm": temperature_normalize(
                mode="denorm", tensor=batch["y"]
            )
            - 1
            + self.weight,
            "y_hat_salinity_denorm": salinity_normalize(
                mode="denorm", tensor=batch["y_salinity"]
            )
            - 1
            + self.weight,
        }


class _SaveResults(pl.Callback):
    """Persist per-rank metrics and actually evaluated patch identities."""

    def __init__(self, root):
        """Keep the temporary result destination."""
        self.root = Path(root)

    def on_validation_end(self, trainer, pl_module):
        """Write reduced metrics from each rank."""
        (self.root / f"rank{trainer.global_rank}.json").write_text(
            json.dumps(
                {
                    "seen": pl_module.seen,
                    "metrics": {
                        key: float(value)
                        for key, value in trainer.callback_metrics.items()
                    },
                }
            )
        )


class TestEN4ArchiveHoldout(unittest.TestCase):
    def _callback(self, root, dataset, **kwargs):
        """Build an explicitly labelled audit fixture and bounded callback."""
        path = root / "audit.parquet"
        path.write_bytes(b"fixed audit fixture")
        return EN4ArchiveHoldoutValidation(
            dataset=dataset,
            candidate_df=dataset.candidates,
            audit_path=path,
            output_dir=root,
            sample_count=kwargs.pop("sample_count", 5),
            batch_size=2,
            min_input_profiles=8,
            **kwargs,
        )

    def test_selection_is_seeded_annual_bounded_and_masks_copies_only(self):
        """All colocated observations disappear in every overlapping selected patch."""
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _Dataset()
            callback = self._callback(Path(tmp), dataset)
            callback._select_patches(("temperature", "salinity"))
            repeated = self._callback(Path(tmp), dataset)
            repeated._select_patches(("temperature", "salinity"))
            self.assertEqual(callback.selection, repeated.selection)
            self.assertEqual(len(callback.patch_indices), 5)
            self.assertEqual(callback.assignments["date"].nunique(), 3)
            self.assertEqual(
                callback.assignments["profile_index"].nunique(),
                len(callback.assignments),
            )
            self.assertEqual(
                callback.audit_sha256,
                hashlib.sha256(b"fixed audit fixture").hexdigest(),
            )
            for index in callback.patch_indices:
                masked = callback._masked_sample(index)
                row = dataset._rows.iloc[index]
                for suffix in ("", "_salinity"):
                    for date, yy, xx in callback.holdouts:
                        xx -= int(row.grid_x0)
                        if date == row.date and 0 <= xx < 4:
                            self.assertFalse(
                                masked[f"x{suffix}_valid_mask"][:, yy, xx].any()
                            )
                            self.assertFalse(
                                masked[f"x{suffix}_valid_mask_1d"][:, yy, xx].any()
                            )
                            self.assertFalse(masked[f"x{suffix}"][:, yy, xx].any())
                    self.assertTrue(dataset[index][f"x{suffix}_valid_mask"].all())
                    self.assertGreaterEqual(
                        int(masked[f"x{suffix}_valid_mask"].any(0).sum()), 8
                    )
                    torch.testing.assert_close(
                        masked[f"y{suffix}"], dataset[index][f"y{suffix}"]
                    )
            self.assertFalse(dataset.heldout_argo_location_keys)

    def test_existing_manifest_cannot_be_silently_replaced(self):
        """Resume preserves the exact cohort and rejects changed provenance."""
        with tempfile.TemporaryDirectory() as tmp:
            callback = self._callback(Path(tmp), _Dataset())
            callback._select_patches(("temperature", "salinity"))
            trainer = SimpleNamespace(is_global_zero=True, world_size=1)
            callback._record_selection(trainer)
            path = Path(tmp) / "en4_archive_holdout_selection.json"
            original = path.read_text()
            callback._record_selection(trainer)
            callback.selection["seed"] = 999
            with self.assertRaisesRegex(
                ValueError, "Existing EN4 holdout selection differs"
            ):
                callback._record_selection(trainer)
            self.assertEqual(path.read_text(), original)

    def test_once_per_epoch_sanity_resume_and_rng_isolation(self):
        """Repeated validation checks and resume cannot multiply reconstruction cost."""
        with tempfile.TemporaryDirectory() as tmp:
            callback = self._callback(Path(tmp), _Dataset())
            model = _Model()
            trainer = SimpleNamespace(
                current_epoch=0,
                sanity_checking=True,
                global_rank=0,
                world_size=1,
                is_global_zero=True,
                logger=None,
                strategy=SimpleNamespace(reduce=lambda value, **kwargs: value),
                precision_plugin=SimpleNamespace(forward_context=nullcontext),
            )
            with patch.object(model, "log_dict") as log:
                callback.on_validation_epoch_end(trainer, model)
                self.assertFalse(model.seen)
                trainer.sanity_checking = False
                rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
                callback.on_validation_epoch_end(trainer, model)
                callback.on_validation_epoch_end(trainer, model)
                self.assertEqual(len(model.seen), 5)
                self.assertEqual(random.getstate(), rng[0])
                np.testing.assert_equal(np.random.get_state(), rng[1])
                torch.testing.assert_close(torch.get_rng_state(), rng[2])
                self.assertEqual(log.call_count, 1)
                saved = callback.state_dict()
                resumed = self._callback(Path(tmp), _Dataset())
                resumed.load_state_dict(saved)
                resumed.on_validation_epoch_end(trainer, model)
                self.assertEqual(len(model.seen), 5)
                trainer.current_epoch = 1
                resumed.on_validation_epoch_end(trainer, model)
                self.assertEqual(len(model.seen), 10)
                metrics = log.call_args.args[0]
                for field in model.output_fields:
                    self.assertAlmostEqual(
                        float(metrics[f"{PREFIX}/{field}_prediction_rmse"]),
                        1.0,
                        places=4,
                    )
                    self.assertAlmostEqual(
                        float(metrics[f"{PREFIX}/{field}_glorys_rmse"]), 2.0, places=4
                    )
                    self.assertAlmostEqual(
                        float(metrics[f"{PREFIX}/{field}_skill_vs_glorys"]),
                        0.5,
                        places=4,
                    )
                self.assertNotIn("val/full_reconstruction_score", metrics)

    def test_ddp_has_exact_patch_identity_and_global_metrics(self):
        """Shard uneven batches without duplicated or mismatched reference profiles."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = _Dataset()
            callback = self._callback(root, dataset)
            trainer = pl.Trainer(
                accelerator="cpu",
                devices=2,
                strategy="ddp_fork",
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
                limit_val_batches=1,
                callbacks=[callback, _SaveResults(root)],
            )
            trainer.validate(
                _Model(), dataloaders=DataLoader(dataset, batch_size=1), verbose=False
            )
            ranks = [
                json.loads((root / f"rank{rank}.json").read_text()) for rank in range(2)
            ]
            selection = json.loads(
                (root / "en4_archive_holdout_selection.json").read_text()
            )
            seen = ranks[0]["seen"] + ranks[1]["seen"]
            self.assertEqual(len(seen), len(set(seen)))
            self.assertEqual(
                sorted(seen), sorted(row["index"] for row in selection["patches"])
            )
            for rank in ranks:
                for field in ("temperature", "salinity"):
                    self.assertAlmostEqual(
                        rank["metrics"][f"{PREFIX}/{field}_prediction_mae"],
                        1.0,
                        places=4,
                    )
                    self.assertEqual(
                        rank["metrics"][f"{PREFIX}/{field}_validated_profile_count"],
                        len(selection["profiles"]),
                    )
                self.assertEqual(rank["metrics"][f"{PREFIX}/patch_count"], 5)

    def test_checkpoint_subset_inputs_and_score_are_unchanged(self):
        """The archive callback must not alter the existing reconstruction evaluator."""
        with tempfile.TemporaryDirectory() as tmp:
            dataset = _Dataset()
            archive = self._callback(Path(tmp), dataset)
            reconstruction = FullReconstructionValidation(
                dataset=dataset, sample_count=5, batch_size=2
            )
            model = _Model()
            trainer = pl.Trainer(
                accelerator="cpu",
                devices=1,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
                limit_val_batches=1,
                callbacks=[archive, reconstruction],
            )
            trainer.validate(
                model, dataloaders=DataLoader(dataset, batch_size=1), verbose=False
            )
            self.assertEqual(len(model.seen), 10)
            self.assertTrue(all(mask.all() for mask in model.masks[-3:]))
            self.assertTrue(
                all(sample["x_valid_mask"].all() for sample in dataset.samples)
            )
            self.assertEqual(
                float(trainer.callback_metrics["val/full_reconstruction/sample_count"]),
                5,
            )

    def test_nonfinite_predictions_are_penalized_on_common_reference_support(self):
        """Model failures cannot improve either side's comparison by reducing support."""
        observed = np.array([[1.0, np.nan, 2.0], [1.0, 2.0, 3.0]])
        glorys = np.array([[2.0, 3.0, np.nan], [2.0, 3.0, 4.0]])
        predicted = np.array([[np.nan, 3.0, 2.0], [1.0, 2.0, 3.0]])
        stats = EN4ArchiveHoldoutValidation._statistics(predicted, glorys, observed)
        self.assertTrue(np.isinf(stats[0, 0, 0]))
        np.testing.assert_equal(stats[0, 2], [2.0, 1.0, 1.0])
        np.testing.assert_equal(stats[0, 2], stats[1, 2])
        np.testing.assert_equal(stats[1, 0], [2.0, 1.0, 1.0])

    def test_loader_filters_proxy_year_and_qc_even_if_training_is_permissive(self):
        """Reject proxy labels, wrong observation years and bad-quality source profiles."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output, cache, land = _make_geotiff_dataset(root)
            dataset = ArgoGeoTIFFGriddedPatchDataset(
                geotiff_root_dir=output,
                metadata_cache_dir=cache,
                split="all",
                tile_size=2,
                resolution_deg=1.0,
                land_mask_path=land,
                patch_stride=2,
                max_land_fraction=1.0,
                require_argo_for_all=False,
                include_salinity=True,
                filter_bad_argo_quality=False,
            )
            path = root / "audit.parquet"
            frame = pd.DataFrame(
                {
                    "profile_source_file": ["EN.4.2.2.f.profiles.g10.202401.nc"] * 2,
                    "source_profile_idx": [7, 8],
                    "datetime_utc": pd.to_datetime(
                        ["2024-01-08", "2024-01-08"], utc=True
                    ),
                    "audit_status": [AUDIT_STATUS, AUDIT_STATUS],
                }
            )
            frame.to_parquet(path)
            kwargs = dict(
                context=load_dataset_context(output),
                date_year=2024,
                candidate_profiles_path=path,
                profile_store=dataset.argo_store,
                audit_status=AUDIT_STATUS,
                require_quality=True,
            )
            selected = load_en4_candidate_profiles(**kwargs)
            self.assertEqual(selected["source_profile_idx"].tolist(), [7])
            self.assertFalse(dataset.argo_store.filter_bad_quality)
            frame.loc[0, "audit_status"] = "proxy_no_spatiotemporal_candidate"
            frame.to_parquet(path)
            with self.assertRaisesRegex(RuntimeError, "No valid"):
                load_en4_candidate_profiles(**kwargs)
            frame.loc[0, "audit_status"] = AUDIT_STATUS
            frame.loc[0, "datetime_utc"] = pd.Timestamp("2023-12-31", tz="UTC")
            frame.to_parquet(path)
            with self.assertRaisesRegex(RuntimeError, "No valid"):
                load_en4_candidate_profiles(**kwargs)
            frame.drop(columns="audit_status").to_parquet(path)
            with self.assertRaisesRegex(ValueError, "audit_status"):
                load_en4_candidate_profiles(**kwargs)
            dataset.argo_store.close()

    def test_builder_is_opt_in_and_rejects_shared_dataset_legacy_callback(self):
        """Do not load audit data when disabled, or allow legacy dataset mutation."""
        kwargs = dict(val_dataset=None, data_cfg={}, output_dir=Path("unused"))
        self.assertIsNone(build_en4_archive_holdout_callback(training_cfg={}, **kwargs))
        with self.assertRaisesRegex(ValueError, "legacy"):
            build_en4_archive_holdout_callback(
                training_cfg={
                    "training": {
                        "en4_archive_holdout": {"enabled": True},
                        "en4_candidate_eval": {"enabled": True},
                    }
                },
                **kwargs,
            )


if __name__ == "__main__":
    unittest.main()

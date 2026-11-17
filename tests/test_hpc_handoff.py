"""Tests for strict offline HPC handoff staging and verification."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from depth_recon.scripts.prepare_hpc_handoff import (
    REQUIRED_ASSET_PATHS,
    inventory_dataset,
    main,
    sha256_file,
)


class HpcHandoffTest(unittest.TestCase):
    """Exercise complete inventory, relocation, and strict preflight failures."""

    def setUp(self) -> None:
        self.tmp_path = Path(tempfile.mkdtemp())
        self.dataset = self.tmp_path / "source_dataset"
        self.dataset.mkdir()
        (self.dataset / "masks").mkdir()
        (self.dataset / "masks" / "mask.tif").write_bytes(b"mask")
        zarr = self.dataset / "argo" / "profiles.zarr"
        zarr.mkdir(parents=True)
        (zarr / ".zgroup").write_text("{}", encoding="utf-8")
        (zarr / "temperature" / "0.0").parent.mkdir()
        (zarr / "temperature" / "0.0").write_bytes(b"zarr chunk")
        salinity = self.dataset / "rasters" / "glorys" / "so" / "so_20160115.tif"
        salinity.parent.mkdir(parents=True)
        salinity.write_bytes(b"salinity")
        self.manifest = {
            "output_dir": str(self.dataset),
            "grid": {"source": "masks/mask.tif", "height": 1, "width": 1},
            "depth_axis_m": [1.0],
            "argo": {"path": "argo/profiles.zarr"},
            "rasters": {
                "glorys": {"thetao": [{"path": "rasters/glorys/so/so_20160115.tif"}]}
            },
        }
        (self.dataset / "manifest.yaml").write_text(
            __import__("yaml").safe_dump(self.manifest), encoding="utf-8"
        )
        self.assets = self._make_assets(salinity)
        self.bundle = self.tmp_path / "hpc_residual"

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_path)

    def _make_assets(self, salinity: Path) -> dict[str, Path]:
        """Create a compatible fixed asset roster with original Italian provenance."""
        manifest_sha = sha256_file(self.dataset / "manifest.yaml")
        profiles = self.tmp_path / "profiles.csv"
        profiles.write_text(
            "profile_id,observation_time,depth_m,observation\n1,2016-01-15T00:00:00Z,1,5\n",
            encoding="utf-8",
        )
        reference = self.tmp_path / "reference_manifest.json"
        reference.write_text(
            json.dumps(
                {
                    "dataset_manifest": str(self.dataset / "manifest.yaml"),
                    "dataset_manifest_sha256": manifest_sha,
                    "output_profiles_sha256": sha256_file(profiles),
                    "salinity_rasters": [
                        {"path": str(salinity), "sha256": sha256_file(salinity)}
                    ],
                }
            ),
            encoding="utf-8",
        )
        screening = self.tmp_path / "screening.parquet"
        screening.write_bytes(b"screening")
        archive = self.tmp_path / "focus.tar.gz"
        archive.write_bytes(b"archive")
        climatology = self.tmp_path / "climatology.npz"
        np.savez(
            climatology,
            temperature=np.zeros((12, 1, 1, 1), dtype=np.float32),
            count=np.ones((12, 1, 1, 1), dtype=np.int32),
            depth_axis_m=np.array([1.0], dtype=np.float32),
            metadata=json.dumps(
                {
                    "val_year_excluded": 2016,
                    "spatial_stride": 1,
                    "grid_shape": [1, 1],
                    "source_grid_shape": [1, 1],
                }
            ),
        )
        bias = self.tmp_path / "bias.npz"
        np.savez(
            bias,
            bias=np.zeros(1, dtype=np.float32),
            count=np.array([30], dtype=np.int64),
            variance=np.zeros(1, dtype=np.float32),
            supported=np.ones(1, dtype=bool),
            depth_axis_m=np.array([1.0], dtype=np.float32),
            metadata=json.dumps(
                {
                    "kind": "en4_minus_glorys_target_bias",
                    "units": "degree_C",
                    "sign_convention": "EN4_minus_GLORYS; target=GLORYS+bias",
                    "excluded_years": [2016],
                    "grid": self.manifest["grid"],
                    "depth_axis_m": [1.0],
                    "min_profiles": 30,
                    "source_temperature": "EN4 potential temperature raw compact profile",
                    "source_temperature_variable": "POTM_CORRECTED",
                    "source_temperature_standard_name": "sea_water_potential_temperature",
                    "input_hashes": {"manifest.yaml": manifest_sha},
                }
            ),
        )
        return {
            "profiles": profiles,
            "reference": reference,
            "screening": screening,
            "archive": archive,
            "climatology": climatology,
            "bias": bias,
        }

    def _stage(self) -> None:
        """Stage the test bundle using the public command-line interface."""
        self.assertEqual(
            main(
                [
                    "stage",
                    "--dataset-root",
                    str(self.dataset),
                    "--climatology",
                    str(self.assets["climatology"]),
                    "--target-bias",
                    str(self.assets["bias"]),
                    "--italian-profiles",
                    str(self.assets["profiles"]),
                    "--italian-manifest",
                    str(self.assets["reference"]),
                    "--italian-screening",
                    str(self.assets["screening"]),
                    "--source-archive",
                    str(self.assets["archive"]),
                    "--output-dir",
                    str(self.bundle),
                    "--hpc-destination",
                    str(self.dataset),
                ]
            ),
            0,
        )

    def _preflight(self) -> int:
        """Run strict preflight against the test dataset as its relocated copy."""
        return main(
            [
                "preflight",
                "--bundle-dir",
                str(self.bundle),
                "--dataset-root",
                str(self.dataset),
            ]
        )

    def test_inventory_streams_and_ignores_cache_trees(self) -> None:
        """Generated caches and repository metadata never enter the transfer inventory."""
        for name in (".cache", ".git", "depthdif_cache"):
            path = self.dataset / name / "ignored"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"ignored")
        entries = inventory_dataset(self.dataset)
        paths = {entry["path"] for entry in entries}
        self.assertNotIn(".cache/ignored", paths)
        self.assertNotIn(".git/ignored", paths)
        self.assertNotIn("depthdif_cache/ignored", paths)
        self.assertIn("argo/profiles.zarr/temperature/0.0", paths)

    def test_stage_and_preflight_preserve_original_italian_manifest(self) -> None:
        """Absolute local provenance remains bytes-identical and validates after relocation."""
        original = self.assets["reference"].read_bytes()
        self._stage()
        self.assertEqual(
            (self.bundle / "italian/reference_manifest.json").read_bytes(), original
        )
        path_map = json.loads(
            (self.bundle / "path_map.json").read_text(encoding="utf-8")
        )
        self.assertEqual(path_map["bundle_relative_root"], "hpc_residual")
        self.assertEqual(path_map["mappings"][0]["kind"], "dataset_tree")
        self.assertEqual(
            {
                record["path"]
                for record in json.loads(
                    (self.bundle / "dataset_inventory.json").read_text()
                )["staged_assets"]
            },
            set(REQUIRED_ASSET_PATHS),
        )
        self.assertEqual(self._preflight(), 0)

    def test_preflight_rejects_deleted_zarr_chunk_and_incomplete_inventory(
        self,
    ) -> None:
        """Zarr chunks must remain present and manifest-referenced trees must be inventoried."""
        self._stage()
        (self.dataset / "argo/profiles.zarr/temperature/0.0").unlink()
        self.assertEqual(self._preflight(), 1)
        (self.dataset / "argo/profiles.zarr/temperature/0.0").write_bytes(b"zarr chunk")
        inventory_path = self.bundle / "dataset_inventory.json"
        payload = json.loads(inventory_path.read_text(encoding="utf-8"))
        payload["files"] = [
            entry
            for entry in payload["files"]
            if entry["path"] != "argo/profiles.zarr/temperature/0.0"
        ]
        inventory_path.write_text(json.dumps(payload), encoding="utf-8")
        self.assertEqual(self._preflight(), 1)

    def test_preflight_rejects_tampered_relocated_italian_salinity(self) -> None:
        """The original Italian manifest's raw salinity digest is checked at its new root."""
        self._stage()
        (self.dataset / "rasters/glorys/so/so_20160115.tif").write_bytes(b"changed")
        self.assertEqual(self._preflight(), 1)

    def test_preflight_rejects_climatology_and_bias_grid_mismatches(self) -> None:
        """Climatology shape/grid and frozen target-bias provenance are semantic checks."""
        bad_climatology = self.tmp_path / "bad_climatology.npz"
        np.savez(
            bad_climatology,
            temperature=np.zeros((12, 1, 2, 1), dtype=np.float32),
            count=np.ones((12, 1, 2, 1), dtype=np.int32),
            depth_axis_m=np.array([1.0], dtype=np.float32),
            metadata=json.dumps(
                {
                    "val_year_excluded": 2016,
                    "spatial_stride": 1,
                    "grid_shape": [2, 1],
                    "source_grid_shape": [1, 1],
                }
            ),
        )
        self.assets["climatology"] = bad_climatology
        self._stage()
        self.assertEqual(self._preflight(), 1)
        self.assets["climatology"] = self.tmp_path / "climatology.npz"
        bad_bias = self.tmp_path / "bad_bias.npz"
        with np.load(self.assets["bias"], allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            metadata["grid"] = {"source": "wrong.tif"}
            np.savez(
                bad_bias,
                bias=data["bias"],
                count=data["count"],
                variance=data["variance"],
                supported=data["supported"],
                depth_axis_m=data["depth_axis_m"],
                metadata=json.dumps(metadata),
            )
        self.assets["bias"] = bad_bias
        self._stage()
        self.assertEqual(self._preflight(), 1)

    def test_inventory_only_is_explicitly_not_preflightable(self) -> None:
        """A quick stat-only readiness inventory cannot accidentally approve transfer."""
        output = self.tmp_path / "inventory_only"
        self.assertEqual(
            main(
                [
                    "stage",
                    "--dataset-root",
                    str(self.dataset),
                    "--output-dir",
                    str(output),
                    "--inventory-only",
                    "--no-full-hash",
                ]
            ),
            0,
        )
        self.assertEqual(
            main(
                [
                    "preflight",
                    "--bundle-dir",
                    str(output),
                    "--dataset-root",
                    str(self.dataset),
                ]
            ),
            1,
        )

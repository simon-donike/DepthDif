from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock
from zipfile import ZipFile

import numpy as np
import torch
import yaml

from inference.api import (
    InferenceAssets,
    _build_public_argo_sample,
    _en4_zip_url,
    _export_args_from_public_api,
    _ostia_filter_for_day,
    _write_public_data_config,
    download_argo_for_week,
    download_ostia_for_week,
    resolve_hf_assets,
    resolve_hf_land_mask,
    run_week_inference,
)


class TestPublicInferenceApi(unittest.TestCase):
    def test_resolve_hf_assets_downloads_expected_paths(self) -> None:
        calls: list[tuple[str, Path]] = []

        def fake_download(url: str, output_path: Path) -> Path:
            calls.append((url, output_path))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("artifact", encoding="utf-8")
            return output_path

        with tempfile.TemporaryDirectory() as tmpdir:
            assets = resolve_hf_assets(
                config_repo="owner/repo",
                revision="rev",
                cache_dir=tmpdir,
                model_config_path="configs/model.yaml",
                data_config_path="configs/data.yaml",
                train_config_path="configs/train.yaml",
                checkpoint_path="checkpoints/model.ckpt",
                downloader=fake_download,
            )

        self.assertEqual(len(calls), 4)
        self.assertEqual(
            calls[0][0],
            "https://huggingface.co/owner/repo/resolve/rev/configs/model.yaml",
        )
        self.assertTrue(str(assets.checkpoint).endswith("checkpoints/model.ckpt"))

    def test_resolve_hf_assets_uses_depthdif_release_defaults(self) -> None:
        calls: list[tuple[str, Path]] = []

        def fake_download(url: str, output_path: Path) -> Path:
            calls.append((url, output_path))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("artifact", encoding="utf-8")
            return output_path

        with tempfile.TemporaryDirectory() as tmpdir:
            resolve_hf_assets(cache_dir=tmpdir, downloader=fake_download)

        urls = [url for url, _path in calls]
        self.assertIn(
            "https://huggingface.co/simon-donike/DepthDif/resolve/main/model_config.yaml",
            urls,
        )
        self.assertIn(
            "https://huggingface.co/simon-donike/DepthDif/resolve/main/depthdif_v1.ckpt",
            urls,
        )

    def test_resolve_hf_land_mask_downloads_expected_path(self) -> None:
        calls: list[tuple[str, Path]] = []

        def fake_download(url: str, output_path: Path) -> Path:
            calls.append((url, output_path))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("mask", encoding="utf-8")
            return output_path

        with tempfile.TemporaryDirectory() as tmpdir:
            path = resolve_hf_land_mask(
                config_repo="owner/repo",
                revision="rev",
                cache_dir=tmpdir,
                land_mask_path="masks/land.tif",
                downloader=fake_download,
            )

        self.assertEqual(
            calls[0][0],
            "https://huggingface.co/owner/repo/resolve/rev/masks/land.tif",
        )
        self.assertTrue(str(path).endswith("masks/land.tif"))

    def test_export_args_from_public_api_keeps_cli_defaults(self) -> None:
        assets = InferenceAssets(
            model_config=Path("model.yaml"),
            data_config=Path("data.yaml"),
            train_config=Path("train.yaml"),
            checkpoint=Path("model.ckpt"),
        )

        args = _export_args_from_public_api(
            assets=assets,
            year=2026,
            iso_week=2,
            rectangle=(-20.0, 30.0, 10.0, 50.0),
            output_root="outputs",
            device="cpu",
            batch_size=None,
            export_ground_truth=False,
            full_sample_count=0,
            land_mask_path="mask.tif",
            min_ocean_fraction=0.2,
            sigma=0.0,
            strict_load=True,
        )

        self.assertEqual(args.year, 2026)
        self.assertEqual(args.iso_week, 2)
        self.assertEqual(args.split, "all")
        self.assertEqual(args.rectangle, [-20.0, 30.0, 10.0, 50.0])
        self.assertEqual(args.model_config, "model.yaml")
        self.assertEqual(args.checkpoint_path, "model.ckpt")
        self.assertFalse(args.export_ground_truth)
        self.assertEqual(args.full_sample_count, 0)
        self.assertEqual(args.min_ocean_fraction, 0.2)
        self.assertTrue(args.strict_load)

    def test_run_week_inference_uses_public_argo_ostia_path_without_glorys(
        self,
    ) -> None:
        with mock.patch("inference.api.run_argo_week_inference") as run_public:
            run_public.return_value = Path("outputs/public")

            result = run_week_inference(
                year=2024,
                iso_week=2,
                rectangle=(-20.0, 30.0, 10.0, 50.0),
                output_root="outputs",
                device="cpu",
                cache_dir="cache",
                copernicus_token="api-key",
            )

        self.assertEqual(result, Path("outputs/public"))
        kwargs = run_public.call_args.kwargs
        self.assertTrue(kwargs["auto_download_argo"])
        self.assertTrue(kwargs["auto_download_ostia"])
        self.assertEqual(kwargs["copernicus_token"], "api-key")
        self.assertEqual(kwargs["config_repo"], "simon-donike/DepthDif")

    def test_run_week_inference_can_skip_ostia_download(self) -> None:
        with mock.patch("inference.api.run_argo_week_inference") as run_public:
            run_public.return_value = Path("outputs/public")

            run_week_inference(
                year=2024,
                iso_week=2,
                rectangle=(-20.0, 30.0, 10.0, 50.0),
                output_root="outputs",
                device="cpu",
                cache_dir="cache",
                auto_download_ostia=False,
            )

        kwargs = run_public.call_args.kwargs
        self.assertFalse(kwargs["auto_download_ostia"])
        self.assertIsNone(kwargs["ostia_dir"])

    def test_build_public_argo_sample_uses_zero_eo_when_ostia_skipped(
        self,
    ) -> None:
        class EmptyArgoStore:
            """Minimal ARGO store with no profiles for sample construction."""

            depth_axis_m = np.asarray([0.0, 10.0], dtype=np.float32)

            def query_indices(self, **_kwargs: object) -> np.ndarray:
                """Return no matching ARGO profiles."""
                return np.asarray([], dtype=np.int64)

        row = {
            "date": 20240110,
            "lat0": 0.0,
            "lat1": 1.0,
            "lon0": 0.0,
            "lon1": 1.0,
        }

        sample = _build_public_argo_sample(
            argo_store=EmptyArgoStore(),
            ostia_store=None,
            row=row,
            tile_size=4,
            resolution_deg=0.25,
            temporal_window_days=7,
            ostia_var_name="analysed_sst",
        )

        self.assertTrue(torch.equal(sample["eo"], torch.zeros((1, 4, 4))))

    def test_en4_zip_url_matches_existing_download_script_pattern(self) -> None:
        self.assertEqual(
            _en4_zip_url("https://example.test/base/", 2024),
            "https://example.test/base/EN.4.2.2.profiles.g10.2024.zip",
        )

    def test_download_argo_for_week_extracts_requested_month_without_network(
        self,
    ) -> None:
        calls: list[str] = []

        def fake_download(url: str, output_path: Path) -> Path:
            calls.append(url)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with ZipFile(output_path, "w") as archive:
                archive.writestr(
                    "nested/EN.4.2.2.f.profiles.g10.202401.nc",
                    b"january",
                )
                archive.writestr(
                    "nested/EN.4.2.2.f.profiles.g10.202402.nc",
                    b"february",
                )
            return output_path

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            output_dir = download_argo_for_week(
                2024,
                2,
                tmp_path / "en4_profiles",
                base_url="https://example.test/en4",
                cache_dir=tmp_path / "cache",
                downloader=fake_download,
            )

            extracted = sorted(path.name for path in output_dir.glob("*.nc"))

        self.assertEqual(
            calls,
            ["https://example.test/en4/EN.4.2.2.profiles.g10.2024.zip"],
        )
        self.assertEqual(extracted, ["EN.4.2.2.f.profiles.g10.202401.nc"])

    def test_download_argo_for_week_skips_existing_month(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            output_dir = tmp_path / "en4_profiles"
            output_dir.mkdir()
            (output_dir / "EN.4.2.2.f.profiles.g10.202401.nc").write_text(
                "existing",
                encoding="utf-8",
            )

            result = download_argo_for_week(2024, 2, output_dir)

        self.assertEqual(result, output_dir)

    def test_ostia_filter_for_day_matches_existing_download_script_pattern(
        self,
    ) -> None:
        self.assertEqual(
            _ostia_filter_for_day("20240110"),
            "*/2024/01/*20240110120000-UKMO-L4_GHRSST-SSTfnd-"
            "OSTIA-GLOB_REP-v02.0-fv02.0.nc",
        )

    def test_download_ostia_for_week_runs_copernicus_and_detects_file(self) -> None:
        commands: list[list[str]] = []

        def fake_runner(cmd: list[str]) -> None:
            commands.append(list(cmd))
            output_dir = Path(cmd[cmd.index("-o") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "20240110120000-UKMO-L4_GHRSST-SSTfnd-OSTIA.nc").write_text(
                "ostia",
                encoding="utf-8",
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = download_ostia_for_week(
                2024,
                2,
                Path(tmpdir) / "ostia",
                dataset_candidates=("dataset-a",),
                username="user",
                password="pass",
                runner=fake_runner,
            )

        self.assertEqual(output_dir.name, "ostia")
        self.assertEqual(commands[0][commands[0].index("-i") + 1], "dataset-a")
        self.assertIn("--username", commands[0])
        self.assertIn("--password", commands[0])

    def test_download_ostia_for_week_accepts_token_alias(self) -> None:
        commands: list[list[str]] = []

        def fake_runner(cmd: list[str]) -> None:
            commands.append(list(cmd))
            output_dir = Path(cmd[cmd.index("-o") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "20240110120000-UKMO-L4_GHRSST-SSTfnd-OSTIA.nc").write_text(
                "ostia",
                encoding="utf-8",
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            download_ostia_for_week(
                2024,
                2,
                Path(tmpdir) / "ostia",
                dataset_candidates=("dataset-a",),
                username="user",
                token="api-key",
                runner=fake_runner,
            )

        password_index = commands[0].index("--password")
        self.assertEqual(commands[0][password_index + 1], "api-key")

    def test_write_public_data_config_applies_source_overrides(self) -> None:
        payload = {
            "dataset": {
                "core": {"argo_dir": "old"},
                "grid": {"land_mask_path": "old-mask"},
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "data.yaml"
            config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

            public_config = _write_public_data_config(
                source_config=config_path,
                output_dir=Path(tmpdir) / "out",
                argo_dir="new-argo",
                glorys_dir="new-glorys",
                ostia_dir=None,
                sealevel_dir=None,
                metadata_cache_dir="new-cache",
                land_mask_path="new-mask",
            )
            with public_config.open("r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)

        self.assertEqual(loaded["dataset"]["core"]["argo_dir"], "new-argo")
        self.assertEqual(loaded["dataset"]["core"]["glorys_dir"], "new-glorys")
        self.assertEqual(loaded["dataset"]["core"]["metadata_cache_dir"], "new-cache")
        self.assertEqual(loaded["dataset"]["grid"]["land_mask_path"], "new-mask")

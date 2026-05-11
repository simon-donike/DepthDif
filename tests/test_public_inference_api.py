from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

import yaml

from inference.api import (
    InferenceAssets,
    _en4_zip_url,
    _export_args_from_public_api,
    _write_public_data_config,
    download_argo_for_week,
    resolve_hf_assets,
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

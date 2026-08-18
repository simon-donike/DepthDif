from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from depth_recon.data.dataset_creation.data_download_packaged import (
    download_aligned_argo_zarr as downloader,
)


class TestDownloadAlignedArgoZarr(unittest.TestCase):
    def test_parse_hf_dataset_url_defaults_revision(self) -> None:
        endpoint, repo_id, revision, subdir = downloader._parse_hf_dataset_url(
            "https://huggingface.co/datasets/ESA-philab/OceanVariableReconstruction/",
            default_revision="main",
        )

        self.assertEqual(endpoint, "https://huggingface.co")
        self.assertEqual(repo_id, "ESA-philab/OceanVariableReconstruction")
        self.assertEqual(revision, "main")
        self.assertEqual(subdir, "")

    def test_parse_hf_dataset_url_keeps_tree_revision_and_subdir(self) -> None:
        endpoint, repo_id, revision, subdir = downloader._parse_hf_dataset_url(
            "https://huggingface.co/datasets/org/name/tree/v1/nested/package",
            default_revision="main",
        )

        self.assertEqual(endpoint, "https://huggingface.co")
        self.assertEqual(repo_id, "org/name")
        self.assertEqual(revision, "v1")
        self.assertEqual(subdir, "nested/package")

    def test_download_hf_package_mirrors_package_files_and_validates_zarr(self) -> None:
        repo_files = [
            "README.md",
            "data/argo_glors_ostia_ssh.zarr/.zgroup",
            "data/argo_glors_ostia_ssh.zarr/profile/.zarray",
            "indices/profiles.parquet",
            "unrelated.bin",
        ]
        downloaded: list[str] = []

        def fake_download(
            url: str,
            output_path: Path,
            *,
            force: bool,
            timeout_seconds: int,
            chunk_size_mb: int,
            token: str | None,
        ) -> Path:
            _ = (url, force, timeout_seconds, chunk_size_mb, token)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"data")
            downloaded.append(output_path.as_posix())
            return output_path

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "hf_argo"
            with (
                patch.object(
                    downloader,
                    "list_hf_dataset_files",
                    return_value=(
                        "https://huggingface.co",
                        "org/name",
                        "main",
                        repo_files,
                    ),
                ),
                patch.object(downloader, "download_file", side_effect=fake_download),
            ):
                written_paths = downloader.download_hf_package(
                    "https://huggingface.co/datasets/org/name",
                    output_dir,
                )

            self.assertEqual(len(written_paths), 4)
            self.assertTrue((output_dir / "data/argo_glors_ostia_ssh.zarr").exists())
            self.assertTrue((output_dir / "indices/profiles.parquet").exists())
            self.assertFalse((output_dir / "unrelated.bin").exists())
            self.assertFalse(any(path.endswith("unrelated.bin") for path in downloaded))

    def test_download_hf_package_can_mirror_full_depthdif_layout(self) -> None:
        repo_files = [
            "README.md",
            "data/argo_glors_ostia_ssh.zarr/.zgroup",
            "argo/argo_profiles_on_grid.zarr/.zgroup",
            "rasters/sss/sos/sos_20240102.tif",
            "masks/world_land_mask_glorys_0p1.tif",
            "manifest.yaml",
            "unrelated.bin",
        ]

        def fake_download(
            url: str,
            output_path: Path,
            *,
            force: bool,
            timeout_seconds: int,
            chunk_size_mb: int,
            token: str | None,
        ) -> Path:
            _ = (url, force, timeout_seconds, chunk_size_mb, token)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"data")
            return output_path

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "OceanVariableReconstruction"
            with (
                patch.object(
                    downloader,
                    "list_hf_dataset_files",
                    return_value=(
                        "https://huggingface.co",
                        "org/name",
                        "main",
                        repo_files,
                    ),
                ),
                patch.object(downloader, "download_file", side_effect=fake_download),
            ):
                written_paths = downloader.download_hf_package(
                    "https://huggingface.co/datasets/org/name",
                    output_dir,
                    package_prefixes=downloader.HF_FULL_PACKAGE_PREFIXES,
                )

            self.assertEqual(len(written_paths), 6)
            self.assertTrue((output_dir / "rasters/sss/sos/sos_20240102.tif").exists())
            self.assertTrue((output_dir / "argo/argo_profiles_on_grid.zarr").exists())
            self.assertTrue(
                (output_dir / "masks/world_land_mask_glorys_0p1.tif").exists()
            )
            self.assertTrue((output_dir / "manifest.yaml").exists())
            self.assertFalse((output_dir / "unrelated.bin").exists())

    def test_training_assets_exclude_large_raster_tree(self) -> None:
        """Training-asset pulls include compact inputs without dense rasters."""
        paths = (
            "data/argo_glors_ostia_ssh.zarr/.zgroup",
            "argo/argo_profiles_on_grid.zarr/.zgroup",
            "manifest.yaml",
            "masks/land_mask.tif",
            "rasters/glorys/thetao/thetao_20240102.tif",
        )

        selected = [
            path
            for path in paths
            if downloader._is_package_file(path, downloader.HF_TRAINING_ASSET_PREFIXES)
        ]

        self.assertIn("argo/argo_profiles_on_grid.zarr/.zgroup", selected)
        self.assertIn("masks/land_mask.tif", selected)
        self.assertNotIn("rasters/glorys/thetao/thetao_20240102.tif", selected)

    def test_training_assets_use_pinned_native_snapshot(self) -> None:
        """The selective mode delegates retries and filtering to HF Hub."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "training_assets"

            def fake_snapshot_download(**kwargs: object) -> str:
                expected = output_dir / downloader.DEFAULT_ZARR_PATH
                expected.mkdir(parents=True)
                (expected / ".zgroup").write_text("{}", encoding="utf-8")
                return str(output_dir)

            with patch.object(
                downloader,
                "snapshot_download",
                side_effect=fake_snapshot_download,
            ) as snapshot_mock:
                paths = downloader.download_hf_training_assets(
                    "https://huggingface.co/datasets/org/name",
                    output_dir,
                    revision="abc123",
                    max_workers=3,
                )

            snapshot_kwargs = snapshot_mock.call_args.kwargs
            self.assertEqual(snapshot_kwargs["revision"], "abc123")
            self.assertEqual(snapshot_kwargs["max_workers"], 3)
            self.assertIn("data/**", snapshot_kwargs["allow_patterns"])
            self.assertIn("argo/**", snapshot_kwargs["allow_patterns"])
            self.assertEqual(snapshot_kwargs["ignore_patterns"], ["rasters/**"])
            self.assertTrue(any(path.name == ".zgroup" for path in paths))


if __name__ == "__main__":
    unittest.main()

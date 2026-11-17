# Example: PYTHONPATH=src /work/envs/depth/bin/python -m depth_recon.data.preflight_climatology --root /work/data/OceanVariableReconstruction --output hpc_residual/climatology/temperature_monthly_stride4.npz --val-year 2016 --spatial-stride 4 --field temperature
"""Prepare a missing training-only climatology before constructing datasets."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any
import zipfile

import numpy as np
import yaml

from depth_recon.data.climatology import fit_monthly_climatology
from depth_recon.data.dataset_argo_geotiff_gridded import (
    DEFAULT_GEOTIFF_ROOT_DIR,
    _records_by_date,
)

DEFAULT_CLIMATOLOGY_PATH = "hpc_residual/climatology/temperature_monthly_stride4.npz"


def file_sha256(path: Path) -> str:
    """Hash a file without retaining its contents in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def climatology_sources(root: Path, field: str, val_year: int) -> list[Path]:
    """List exactly the manifest and uncorrected GLORYS files used by fitting."""
    manifest_path = root / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    fields = ("temperature", "salinity") if field == "both" else (field,)
    paths = {manifest_path}
    for name in fields:
        variable = "thetao" if name == "temperature" else "so"
        records = _records_by_date(manifest["rasters"]["glorys"][variable], root)
        training = [path for date, path in records.items() if date // 10000 != val_year]
        if not training:
            raise ValueError("No training-year GLORYS dates remain for climatology.")
        paths.update(training)
    return sorted(paths)


def validate_climatology(path: Path, root: Path, *, val_year: int, field: str) -> None:
    """Check compatibility without inflating the large field arrays into RAM."""
    manifest = yaml.safe_load((root / "manifest.yaml").read_text())
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        if not np.array_equal(
            data["depth_axis_m"], np.asarray(manifest["depth_axis_m"], dtype=np.float32)
        ):
            raise ValueError("Climatology depth axis does not match the dataset.")
    if metadata.get("val_year_excluded") != val_year:
        raise ValueError("Climatology excluded validation year does not match.")
    if any(int(date) // 10000 == val_year for date in metadata.get("dates", [])):
        raise ValueError("Climatology includes validation-year dates.")
    grid = manifest["grid"]
    if "grid" in metadata and metadata["grid"] != grid:
        raise ValueError("Climatology spatial grid does not match the dataset.")
    shape = (int(grid["height"]), int(grid["width"]))
    stride = int(metadata["spatial_stride"])
    if stride < 1 or tuple(metadata["source_grid_shape"]) != shape:
        raise ValueError("Climatology source grid/stride is invalid.")
    coarse_shape = tuple((size + stride - 1) // stride for size in shape)
    if tuple(metadata["grid_shape"]) != coarse_shape:
        raise ValueError("Climatology downsampled grid is invalid.")
    fields = ("temperature", "salinity") if field == "both" else (field,)
    available = metadata.get("fields", [metadata.get("field", "temperature")])
    if not set(fields) <= set(available):
        raise ValueError(f"Climatology does not support requested field {field}.")
    expected = (12, len(manifest["depth_axis_m"]), *coarse_shape)
    # Read NPY headers inside the archive; runtime sampling validates array values.
    with zipfile.ZipFile(path) as archive:
        for name in (*fields, "count"):
            with archive.open(f"{name}.npy") as stream:
                version = np.lib.format.read_magic(stream)
                reader = (
                    np.lib.format.read_array_header_1_0
                    if version == (1, 0)
                    else np.lib.format.read_array_header_2_0
                )
                array_shape, _, dtype = reader(stream)
            if array_shape != expected or dtype.hasobject:
                raise ValueError(f"Climatology {name} shape/type is invalid.")


def ensure_climatology(
    root: str | Path,
    output: str | Path = DEFAULT_CLIMATOLOGY_PATH,
    *,
    val_year: int,
    spatial_stride: int = 4,
    field: str = "temperature",
    verified_source_hashes: dict[str, str] | None = None,
) -> Path:
    """Reuse a compatible artifact, or atomically fit it once across launch ranks.

    ``verified_source_hashes`` may only be supplied by a preflight that has just
    hashed the complete source inventory. Other callers hash fitting inputs here.
    """
    if (
        val_year is None
        or spatial_stride < 1
        or field not in {"temperature", "salinity", "both"}
    ):
        raise ValueError(
            "Climatology requires val_year, a positive stride and a valid field."
        )
    root = Path(root).expanduser().resolve()
    path = Path(output).expanduser()
    if not path.is_absolute():
        path = root / path
    if path.exists():
        validate_climatology(path, root, val_year=val_year, field=field)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    # All torchrun/SLURM ranks may reach this before a process group exists.
    # A filesystem lock and same-filesystem rename prevent partial NPZ reads.
    with path.with_suffix(path.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            validate_climatology(path, root, val_year=val_year, field=field)
            return path
        source_hashes = {}
        for source in climatology_sources(root, field, int(val_year)):
            relative = os.path.relpath(source, root)
            source_hashes[relative] = (
                file_sha256(source)
                if verified_source_hashes is None
                else verified_source_hashes[relative]
            )
        print(
            f"Climatology preflight: fitting {field} at {path} (excluding {val_year}, stride {spatial_stride})",
            flush=True,
        )
        with tempfile.TemporaryDirectory(
            prefix=".climatology-", dir=path.parent
        ) as temp_dir:
            temporary = Path(temp_dir) / "climatology.npz"
            fit_monthly_climatology(
                root,
                temporary,
                val_year=int(val_year),
                spatial_stride=spatial_stride,
                field=field,
            )
            validate_climatology(temporary, root, val_year=int(val_year), field=field)
            receipt = {
                "schema_version": 1,
                "artifact_sha256": file_sha256(temporary),
                "recipe": {
                    "val_year": int(val_year),
                    "spatial_stride": spatial_stride,
                    "field": field,
                },
                "source_hashes": source_hashes,
            }
            receipt_path = Path(temp_dir) / "receipt.json"
            receipt_path.write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n"
            )
            # Publish provenance first: an interrupted write never leaves a complete
            # generated artifact without its checksummed generation record.
            os.replace(receipt_path, path.with_suffix(path.suffix + ".preflight.json"))
            os.replace(temporary, path)
        print(f"Climatology preflight: ready {path}", flush=True)
    return path


def prepare_training_climatology(
    data_cfg: dict[str, Any], model_cfg: dict[str, Any]
) -> Path | None:
    """Prepare the configured background before any dataset or model is built."""
    dataset = data_cfg.get("dataset", {})
    settings = dataset.get("climatology", {})
    if not settings.get("enabled", False):
        return None
    output = settings.get("path")
    if not output:
        raise ValueError("Enabled climatology requires a non-empty artifact path.")
    root = dataset.get("core", {}).get(
        "geotiff_root_dir", dataset.get("geotiff_root_dir", DEFAULT_GEOTIFF_ROOT_DIR)
    )
    fields = model_cfg.get("model", {}).get("output_fields", ["temperature"])
    fields = [fields] if isinstance(fields, str) else fields
    field = "both" if set(fields) == {"temperature", "salinity"} else fields[0]
    return ensure_climatology(
        root,
        output,
        val_year=data_cfg.get("split", {}).get("val_year"),
        spatial_stride=int(settings.get("spatial_stride", 4)),
        field=field,
    )


def main(argv: list[str] | None = None) -> int:
    """Run CPU preflight independently of a GPU training allocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(DEFAULT_GEOTIFF_ROOT_DIR))
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_CLIMATOLOGY_PATH))
    parser.add_argument("--val-year", type=int, required=True)
    parser.add_argument("--spatial-stride", type=int, default=4)
    parser.add_argument(
        "--field", choices=("temperature", "salinity", "both"), default="temperature"
    )
    args = parser.parse_args(argv)
    ensure_climatology(
        args.root,
        args.output,
        val_year=args.val_year,
        spatial_stride=args.spatial_stride,
        field=args.field,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import xarray as xr
import yaml
import torch
import random


def load_yaml(path: str) -> dict[str, Any]:
    """Load and return yaml data.

    Args:
        path (str): Path to an input or output file.

    Returns:
        dict[str, Any]: Dictionary containing computed outputs.
    """
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_default_nc(config_path: str) -> Path:
    """Compute find default nc and return the result.

    Args:
        config_path (str): Path to an input or output file.

    Returns:
        Path: Computed output value.
    """
    cfg = load_yaml(config_path)
    ds_cfg = cfg.get("dataset", {})
    source_cfg = ds_cfg.get("source", {})
    root_dir_value = source_cfg.get("root_dir", None)
    if root_dir_value is None:
        raise KeyError("Missing dataset source root dir in config (dataset.source.root_dir).")
    root_dir = Path(root_dir_value)
    nc_files = sorted(root_dir.glob("*.nc"))
    if not nc_files:
        raise FileNotFoundError(f"No .nc files found in {root_dir}")
    return nc_files[0]


def load_random_temperature_window(
    nc_path: Path,
    *,
    window_size: int = 128,
    depth_index: int = -1,  # -1 = deepest model level
    variable: str = "thetao",
) -> torch.Tensor:
    """Load and return random temperature window data.

    Args:
        nc_path (Path): Path to an input or output file.
        window_size (int): Input value.
        depth_index (int): Input value.
        variable (str): Input value.

    Returns:
        torch.Tensor: Tensor output produced by this call.
    """

    with xr.open_dataset(nc_path) as ds:
        var = ds[variable].isel(time=0, depth=depth_index)

        lat_size = var.sizes["latitude"]
        lon_size = var.sizes["longitude"]

        if window_size > lat_size or window_size > lon_size:
            raise ValueError("Window size larger than dataset dimensions")

        lat0 = random.randint(0, lat_size - window_size)
        lon0 = random.randint(0, lon_size - window_size)

        patch = var.isel(
            latitude=slice(lat0, lat0 + window_size),
            longitude=slice(lon0, lon0 + window_size),
        )

        # (H, W) → (1, H, W)
        tensor = torch.from_numpy(patch.values).float().unsqueeze(0)

        return tensor


def print_nc_info(nc_path: Path) -> None:
    """Print nc info for manual inspection.

    Args:
        nc_path (Path): Path to an input or output file.

    Returns:
        None: No value is returned.
    """
    lines: list[str] = [f"Opening: {nc_path}"]
    with xr.open_dataset(nc_path) as ds:
        lines.append("\n=== DATASET SUMMARY ===")
        lines.append(str(ds))

        lines.append("\n=== GLOBAL ATTRIBUTES (TAGS) ===")
        if ds.attrs:
            for k, v in ds.attrs.items():
                lines.append(f"- {k}: {v}")
        else:
            lines.append("(none)")

        lines.append("\n=== DIMENSIONS ===")
        for dim, size in ds.dims.items():
            lines.append(f"- {dim}: {size}")

        lines.append("\n=== COORDINATES ===")
        for name, coord in ds.coords.items():
            lines.append(
                f"- {name}: dims={coord.dims}, shape={coord.shape}, dtype={coord.dtype}"
            )
            if coord.attrs:
                lines.append("  attrs:")
                for k, v in coord.attrs.items():
                    lines.append(f"    - {k}: {v}")

        lines.append("\n=== DATA VARIABLES ===")
        for name, var in ds.data_vars.items():
            lines.append(
                f"- {name}: dims={var.dims}, shape={var.shape}, dtype={var.dtype}"
            )
            if var.attrs:
                lines.append("  attrs:")
                for k, v in var.attrs.items():
                    lines.append(f"    - {k}: {v}")

        lines.append("\n=== DEPTH LEVELS (METERS) ===")
        if "depth" in ds.coords:
            depth_values = ds["depth"].values
            for i, depth_m in enumerate(depth_values):
                lines.append(f"- level {i}: {float(depth_m):.3f} m")
        else:
            lines.append("No 'depth' coordinate found in this dataset.")

    output_path = Path("data/data_info.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run the script entry point.

    Args:
        None: This callable takes no explicit input arguments.

    Returns:
        None: No value is returned.
    """
    parser = argparse.ArgumentParser(description="Inspect a NetCDF (.nc) file.")
    parser.add_argument(
        "--nc-path",
        type=str,
        default=None,
        help="Path to a .nc file. If omitted, first file from data config root_dir is used.",
    )
    parser.add_argument(
        "--data-config",
        type=str,
        default="configs/data_config.yaml",
        help="Path to data config yaml (used when --nc-path is omitted).",
    )
    args = parser.parse_args()

    nc_path = Path(args.nc_path) if args.nc_path else find_default_nc(args.data_config)
    print_nc_info(nc_path)


if __name__ == "__main__":
    main()

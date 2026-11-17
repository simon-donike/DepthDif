# Example: /work/envs/depth/bin/python -m depth_recon.data.fit_climatology --root /work/data/OceanVariableReconstruction --output /tmp/climatology.npz --val-year 2022 --spatial-stride 4 --field temperature
"""Command-line fitting for training-only monthly climatology artifacts."""

from __future__ import annotations

import argparse

from depth_recon.data.climatology import fit_monthly_climatology


def main() -> None:
    """Parse options and write one climatology artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="GeoTIFF dataset root")
    parser.add_argument("--output", required=True, help="Output .npz artifact")
    parser.add_argument("--val-year", type=int, required=True)
    parser.add_argument("--spatial-stride", type=int, default=4)
    parser.add_argument(
        "--field", choices=("temperature", "salinity", "both"), default="temperature"
    )
    args = parser.parse_args()
    fit_monthly_climatology(
        args.root,
        args.output,
        val_year=args.val_year,
        spatial_stride=args.spatial_stride,
        field=args.field,
    )


if __name__ == "__main__":
    main()

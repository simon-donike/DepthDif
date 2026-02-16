from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def _window_key(row: dict[str, str], row_index: int) -> tuple[str, ...]:
    has_grid = all(k in row and row[k] != "" for k in ("y0", "x0", "edge_size"))
    if has_grid:
        return ("grid", row["y0"], row["x0"], row["edge_size"])

    has_geo = all(k in row and row[k] != "" for k in ("lat0", "lat1", "lon0", "lon1"))
    if has_geo:
        # Use raw strings from CSV to avoid floating-point formatting drift.
        return ("geo", row["lat0"], row["lat1"], row["lon0"], row["lon1"])

    return ("row", str(row_index))


def _val_count(n_windows: int, val_fraction: float) -> int:
    val_len = int(round(n_windows * val_fraction))
    if n_windows > 1:
        lower = 1 if val_fraction > 0.0 else 0
        val_len = min(max(val_len, lower), n_windows - 1)
    else:
        val_len = 0
    return val_len


def assign_split(
    input_csv: Path,
    output_csv: Path,
    *,
    val_fraction: float,
    seed: int,
) -> None:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    with input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError("CSV has no header row.")
        rows = list(reader)
        fieldnames = list(reader.fieldnames)

    if not rows:
        raise RuntimeError("CSV has no data rows.")

    key_by_row: list[tuple[str, ...]] = []
    unique_windows: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()

    for i, row in enumerate(rows):
        key = _window_key(row, i)
        key_by_row.append(key)
        if key not in seen:
            seen.add(key)
            unique_windows.append(key)

    n_windows = len(unique_windows)
    val_len = _val_count(n_windows, val_fraction)

    rng = random.Random(seed)
    val_windows = set(rng.sample(unique_windows, k=val_len)) if val_len > 0 else set()

    had_existing_split = "split" in fieldnames

    for i, row in enumerate(rows):
        row["split"] = "val" if key_by_row[i] in val_windows else "train"

    if "split" not in fieldnames:
        fieldnames.append("split")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_val_rows = sum(1 for r in rows if r["split"] == "val")
    print(f"Wrote: {output_csv}")
    if had_existing_split:
        print("Overrode existing 'split' column.")
    else:
        print("Added new 'split' column.")
    print(f"Rows: {len(rows)}")
    print(f"Unique windows: {n_windows}")
    print(f"Val windows: {val_len} ({(100.0 * val_len / n_windows):.2f}%)")
    print(f"Val rows: {n_val_rows} ({(100.0 * n_val_rows / len(rows)):.2f}%)")
    print(f"Train rows: {len(rows) - n_val_rows}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Assign a deterministic train/val split by unique spatial window and "
            "apply it consistently across all timestamps/files in the CSV."
        )
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        nargs="?",
        default=Path("/work/data/depth/4_bands/patch_index_with_paths.csv"),
        help=(
            "Path to input CSV index. "
            "Default: /work/data/depth/4_bands/patch_index_with_paths.csv"
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Path to output CSV. Defaults to overwriting input CSV.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of unique windows assigned to val. Default: 0.2",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic window selection. Default: 42",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    val_fraction = max(0.0, min(1.0, float(args.val_fraction)))
    output_csv = args.output_csv if args.output_csv is not None else args.input_csv

    assign_split(
        input_csv=args.input_csv,
        output_csv=output_csv,
        val_fraction=val_fraction,
        seed=int(args.seed),
    )


if __name__ == "__main__":
    main()
    # Example: python3 data/assign_window_split.py /work/data/depth/4_bands/patch_index_with_paths.csv --output-csv /work/data/depth/4_bands/patch_index_with_paths_split.csv --val-fraction 0.2 --seed 42
    # /work/data/depth/4_bands_v2/patch_index_with_paths.csv
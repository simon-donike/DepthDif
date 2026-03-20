from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.dataset_ostia_argo import OstiaArgoTileDataset


class _ParallelTiffExportDataset(Dataset):
    """Dataset wrapper that exports one OSTIA/Argo/GLORYS sample per item access."""

    def __init__(
        self,
        *,
        dataset: OstiaArgoTileDataset,
        export_indices: list[int],
        output_root: Path,
        manifest_path: Path,
        overwrite: bool,
    ) -> None:
        self.dataset = dataset
        self.export_indices = export_indices
        self.output_root = output_root
        self.manifest_path = manifest_path
        self.overwrite = bool(overwrite)

    def __len__(self) -> int:
        return len(self.export_indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        export_idx = int(self.export_indices[int(idx)])
        return self.dataset.save_to_disk(
            export_idx,
            output_root=self.output_root,
            manifest_path=self.manifest_path,
            overwrite=self.overwrite,
            write_manifest=False,
        )


def _single_record_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if len(batch) != 1:
        raise RuntimeError(f"Expected batch_size=1 for export, got batch of size {len(batch)}")
    return batch[0]


def _resolve_export_indices(total_len: int, *, start_index: int, limit: int | None) -> list[int]:
    if start_index < 0:
        raise ValueError("start_index must be >= 0.")
    if start_index >= total_len:
        return []
    stop_index = total_len if limit is None else min(total_len, start_index + max(int(limit), 0))
    return list(range(int(start_index), int(stop_index)))


def _shuffle_export_indices(
    export_indices: list[int],
    *,
    shuffle_seed: int | None,
    shuffle_block_size: int,
) -> list[int]:
    block_size = max(int(shuffle_block_size), 1)
    shuffled = list(export_indices)
    blocks = [
        shuffled[start : start + block_size]
        for start in range(0, len(shuffled), block_size)
    ]
    # Default to fresh entropy on each run so resumed exports do not replay one fixed order.
    rng = np.random.default_rng(None if shuffle_seed is None else int(shuffle_seed))
    rng.shuffle(blocks)
    return [idx for block in blocks for idx in block]


def _write_manifest(records: list[dict[str, Any]], manifest_path: Path) -> None:
    if manifest_path.exists():
        existing_records = pd.read_csv(manifest_path).to_dict(orient="records")
    else:
        existing_records = []

    merged_records: dict[int, dict[str, Any]] = {}
    for record in existing_records:
        merged_records[int(record["export_index"])] = record
    for record in records:
        # Keep previously exported samples in the manifest while letting the current
        # run refresh rows that were re-exported or resumed.
        merged_records[int(record["export_index"])] = record

    ordered_records = [
        merged_records[export_index]
        for export_index in sorted(merged_records)
    ]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(ordered_records).to_csv(manifest_path, index=False)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export OstiaArgoTileDataset samples as georeferenced OSTIA/ARGO/GLORYS "
            "GeoTIFF triplets using a multi-worker "
            "PyTorch DataLoader, then write one manifest CSV in the main process."
        )
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("/work/data/depth_v2/ostia_patch_index_daily_0p1.csv"),
        help="Merged daily CSV used by OstiaArgoTileDataset.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/work/data/depth_v3"),
        help="Directory containing the output argo/, glorys/, and ostia/ folders.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Optional explicit manifest CSV path. Defaults to <output-root>/ostia_argo_tiff_index.csv.",
    )
    parser.add_argument(
        "--root-path",
        type=Path,
        default=None,
        help="Optional dataset root passed through to OstiaArgoTileDataset.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=("all", "train", "val"),
        help="Dataset split filter.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=1,
        help="Temporal averaging window length passed to OstiaArgoTileDataset.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=128,
        help="Spatial tile size passed to OstiaArgoTileDataset.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="PyTorch DataLoader worker count for parallel TIFF export.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="DataLoader prefetch_factor when num_workers > 0.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Dataset index to start exporting from.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of samples to export.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite TIFF triplets even if they already exist.",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=100,
        help="Write the manifest CSV every N exported samples in the main process.",
    )
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shuffle export order so partially written datasets cover the full timespan better.",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help="Optional random seed for deterministic export-order shuffling.",
    )
    parser.add_argument(
        "--shuffle-block-size",
        type=int,
        default=100,
        help="Shuffle export order in contiguous blocks of this many samples.",
    )
    parser.add_argument(
        "--verbose-init",
        action="store_true",
        help="Enable verbose OstiaArgoTileDataset initialization logging.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    dataset = OstiaArgoTileDataset(
        args.csv_path,
        root_path=args.root_path,
        split=args.split,
        tile_size=args.tile_size,
        days=args.days,
        return_argo_profiles=True,
        verbose_init=bool(args.verbose_init),
    )
    manifest_path = (
        args.manifest_path
        if args.manifest_path is not None
        else args.output_root / "ostia_argo_tiff_index.csv"
    )
    export_indices = _resolve_export_indices(
        len(dataset),
        start_index=int(args.start_index),
        limit=args.limit,
    )
    if bool(args.shuffle):
        export_indices = _shuffle_export_indices(
            export_indices,
            shuffle_seed=args.shuffle_seed,
            shuffle_block_size=int(args.shuffle_block_size),
        )
    if not export_indices:
        raise RuntimeError("No samples selected for export.")

    export_dataset = _ParallelTiffExportDataset(
        dataset=dataset,
        export_indices=export_indices,
        output_root=args.output_root,
        manifest_path=manifest_path,
        overwrite=bool(args.overwrite),
    )

    loader_kwargs: dict[str, Any] = dict(
        dataset=export_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=False,
        collate_fn=_single_record_collate,
    )
    if int(args.num_workers) > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = int(args.prefetch_factor)
    loader = DataLoader(**loader_kwargs)

    records: list[dict[str, Any]] = []
    written_count = 0
    flush_every = max(int(args.flush_every), 1)
    with tqdm(total=len(export_dataset), desc="Exporting OSTIA/ARGO/GLORYS TIFF triplets") as pbar:
        for record in loader:
            records.append(record)
            written_count += int(record.get("files_written", 0))
            if len(records) % flush_every == 0:
                # Persist partial progress so long-running exports can be resumed or inspected.
                _write_manifest(records, manifest_path)
            pbar.update(1)

    _write_manifest(records, manifest_path)

    print(
        "Export complete: "
        f"samples={len(records)}, files_written={written_count}, "
        f"manifest={manifest_path}"
    )


if __name__ == "__main__":
    main()


"""

/work/envs/depth/bin/python data/export_ostia_argo_tiffs.py \
  --csv-path /data1/datasets/depth_v2/ostia_patch_index_daily.csv \
  --output-root /work/data/depth_prod \
  --days 7 \
  --num-workers 10 \
  --prefetch-factor 4 \
  --shuffle \
  --shuffle-block-size 500 \
  --flush-every 100 \
  --start-index 0


"""

# Example:
#   /work/envs/depth/bin/python src/depth_recon/scripts/benchmark_dataloader_settings.py --scenario temperature --workers 0,2,4,6,8 --prefetch-factors 2,4 --batches 80 --warmup-batches 10
"""Benchmark pixel training dataloader worker and prefetch settings."""

from __future__ import annotations

import argparse
import csv
import gc
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

import torch

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from depth_recon.configs.config_resolver_pixel import (  # noqa: E402
    DEFAULT_PIXEL_TRAINING_CONFIG_PATH,
    load_pixel_training_config,
)
from depth_recon.data.datamodule import DepthTileDataModule  # noqa: E402
from train import build_dataset  # noqa: E402


def parse_int_list(value: str) -> list[int]:
    """Parse a comma-separated list of integers."""
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    return [int(item) for item in items]


def move_to_device(value: Any, device: torch.device) -> Any:
    """Recursively move tensors in a batch to the selected device."""
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def tensor_bytes(value: Any) -> int:
    """Count tensor payload bytes in a nested batch."""
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(item) for item in value)
    return 0


def format_float(value: float | None) -> str:
    """Format optional floating-point metrics for terminal tables."""
    if value is None:
        return "-"
    return f"{value:.3f}"


def benchmark_loader(
    *,
    dataset: torch.utils.data.Dataset,
    base_dataloader_cfg: dict[str, Any],
    num_workers: int,
    prefetch_factor: int | None,
    device: torch.device,
    batches: int,
    warmup_batches: int,
    timeout: float,
    worker_start_method: str | None,
) -> dict[str, Any]:
    """Benchmark one DataLoader setting and return throughput metrics."""
    dataloader_cfg = dict(base_dataloader_cfg)
    dataloader_cfg["num_workers"] = int(num_workers)
    dataloader_cfg["persistent_workers"] = False
    if worker_start_method:
        dataloader_cfg["multiprocessing_context"] = str(worker_start_method)
    if prefetch_factor is not None:
        dataloader_cfg["prefetch_factor"] = int(prefetch_factor)

    # Validation is not used here, but the DataModule keeps the loader construction
    # identical to training and avoids a separate ad hoc DataLoader path.
    datamodule = DepthTileDataModule(
        dataset=dataset,
        val_dataset=dataset,
        dataloader_cfg=dataloader_cfg,
    )
    loader = datamodule.train_dataloader()
    if int(num_workers) > 0 and timeout > 0:
        loader.timeout = float(timeout)

    measured_batches = 0
    measured_samples = 0
    measured_bytes = 0
    start_time: float | None = None
    first_batch_seconds: float | None = None
    wall_start = time.perf_counter()
    error: str | None = None

    try:
        for batch_index, batch in enumerate(loader):
            if first_batch_seconds is None:
                first_batch_seconds = time.perf_counter() - wall_start

            if start_time is None:
                start_time = time.perf_counter()

            moved = move_to_device(batch, device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

            if batch_index + 1 <= warmup_batches:
                # Start measuring after warmup so startup and first-batch latency
                # are reported separately but excluded from steady-state timing.
                start_time = time.perf_counter()
                del moved
                del batch
                continue

            measured_batches += 1
            batch_size = int(dataloader_cfg.get("batch_size", 1))
            measured_samples += batch_size
            measured_bytes += tensor_bytes(moved)

            # Drop references before asking the next worker batch to arrive.
            del moved
            del batch

            if measured_batches >= batches:
                break
    except Exception as exc:  # noqa: BLE001 - benchmark should keep testing settings.
        error = f"{type(exc).__name__}: {exc}"
    finally:
        # Explicit shutdown avoids leaving worker processes alive between settings.
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None and hasattr(iterator, "_shutdown_workers"):
            iterator._shutdown_workers()
        del loader
        del datamodule
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    elapsed = 0.0 if start_time is None else max(time.perf_counter() - start_time, 0.0)
    return {
        "num_workers": int(num_workers),
        "prefetch_factor": prefetch_factor,
        "measured_batches": measured_batches,
        "samples_per_second": None if elapsed <= 0.0 else measured_samples / elapsed,
        "batches_per_second": None if elapsed <= 0.0 else measured_batches / elapsed,
        "mb_per_second_to_gpu": (
            None if elapsed <= 0.0 else (measured_bytes / 1048576.0) / elapsed
        ),
        "first_batch_seconds": first_batch_seconds,
        "error": error,
    }


def print_table(rows: list[dict[str, Any]], *, ddp_ranks: int) -> None:
    """Print benchmark results sorted by measured sample throughput."""
    headers = [
        "rank",
        "workers/rank",
        "total_workers",
        "prefetch",
        "batches",
        "samples/s",
        "batches/s",
        "MB/s to GPU",
        "first batch s",
        "error",
    ]
    sorted_rows = sorted(
        rows,
        key=lambda row: row["samples_per_second"] or -1.0,
        reverse=True,
    )
    table: list[list[str]] = []
    for rank, row in enumerate(sorted_rows, start=1):
        table.append(
            [
                str(rank),
                str(row["num_workers"]),
                str(row["num_workers"] * ddp_ranks),
                "-" if row["prefetch_factor"] is None else str(row["prefetch_factor"]),
                str(row["measured_batches"]),
                format_float(row["samples_per_second"]),
                format_float(row["batches_per_second"]),
                format_float(row["mb_per_second_to_gpu"]),
                format_float(row["first_batch_seconds"]),
                row["error"] or "",
            ]
        )

    widths = [
        max(len(header), *(len(row[index]) for row in table))
        for index, header in enumerate(headers)
    ]
    print(
        " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    )
    print("-|-".join("-" * width for width in widths))
    for row in table:
        print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)))

    if sorted_rows and sorted_rows[0]["samples_per_second"] is not None:
        best = sorted_rows[0]
        print(
            "\nBest setting: "
            f"num_workers={best['num_workers']}, "
            f"prefetch_factor={best['prefetch_factor']}, "
            f"persistent_workers=false"
        )


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark pixel training dataloader settings while moving batches to GPU."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_PIXEL_TRAINING_CONFIG_PATH,
        help="Pixel training super-config path.",
    )
    parser.add_argument("--scenario", default=None, help="Pixel scenario override.")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Config override in root.path=value form. Can be repeated.",
    )
    parser.add_argument(
        "--workers",
        type=parse_int_list,
        default=parse_int_list("0,2,4,6,8"),
        help="Comma-separated num_workers values per rank.",
    )
    parser.add_argument(
        "--prefetch-factors",
        type=parse_int_list,
        default=parse_int_list("2,4"),
        help="Comma-separated prefetch_factor values for worker runs.",
    )
    parser.add_argument("--batches", type=int, default=80, help="Measured batches.")
    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=10,
        help="Warmup batches excluded from timing.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device used for realistic batch transfer timing.",
    )
    parser.add_argument(
        "--ddp-ranks",
        type=int,
        default=max(1, torch.cuda.device_count()),
        help="Ranks used to report total DDP worker count.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="DataLoader timeout in seconds for worker runs.",
    )
    parser.add_argument(
        "--worker-start-method",
        default="spawn",
        choices=("spawn", "fork", "forkserver"),
        help="Multiprocessing start method for worker runs.",
    )
    parser.add_argument("--output-csv", type=Path, default=None, help="Optional CSV.")
    return parser


def main() -> None:
    """Run all requested dataloader benchmarks."""
    args = build_parser().parse_args()
    if args.batches <= 0:
        raise ValueError("--batches must be positive.")
    if args.warmup_batches < 0:
        raise ValueError("--warmup-batches must be non-negative.")

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    runtime_dir = Path(tempfile.mkdtemp(prefix="depthdif_dataloader_bench_"))
    bundle = load_pixel_training_config(
        config_path_value=args.config,
        scenario_override=args.scenario,
        overrides=args.overrides,
        runtime_config_dir=runtime_dir,
        write_snapshots=False,
    )
    ds_cfg = bundle.data_cfg.get("dataset", {})
    base_dataloader_cfg = dict(bundle.training_cfg.get("dataloader", {}))
    data_dataloader_cfg = bundle.data_cfg.get("dataloader", {})
    if isinstance(data_dataloader_cfg, dict):
        base_dataloader_cfg.update(data_dataloader_cfg)

    print(f"Config: {args.config}")
    print(f"Scenario: {args.scenario or 'config default'}")
    print(f"Device: {device}")
    print(f"Batch size: {base_dataloader_cfg.get('batch_size')}")
    print(f"DDP ranks for total-worker reporting: {args.ddp_ranks}")
    print(f"Runtime config dir: {runtime_dir}")
    print("Building train dataset...")
    dataset = build_dataset(
        data_config_path=bundle.effective_data_config_path,
        ds_cfg=ds_cfg,
        split="train",
    )
    print(f"Train rows: {len(dataset)}")

    rows: list[dict[str, Any]] = []
    for workers in args.workers:
        factors: list[int | None] = [None] if workers == 0 else args.prefetch_factors
        for prefetch_factor in factors:
            label = "-" if prefetch_factor is None else str(prefetch_factor)
            print(f"\nBenchmarking num_workers={workers}, prefetch_factor={label}...")
            rows.append(
                benchmark_loader(
                    dataset=dataset,
                    base_dataloader_cfg=base_dataloader_cfg,
                    num_workers=workers,
                    prefetch_factor=prefetch_factor,
                    device=device,
                    batches=args.batches,
                    warmup_batches=args.warmup_batches,
                    timeout=args.timeout,
                    worker_start_method=args.worker_start_method,
                )
            )

    print("\nResults")
    print_table(rows, ddp_ranks=args.ddp_ranks)

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote CSV: {args.output_csv}")


if __name__ == "__main__":
    main()

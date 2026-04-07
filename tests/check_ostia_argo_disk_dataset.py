import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.dataset_ostia_argo_disk import OstiaArgoTiffDataset


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Iterate the OSTIA/ARGO disk dataset and report sample-loading or "
            "date-conditioning errors before training."
        )
    )
    parser.add_argument(
        "--data-config",
        default="configs/px_space/data_ostia_argo_disk.yaml",
        help="Path to the dataset config used to build OstiaArgoTiffDataset.",
    )
    parser.add_argument(
        "--split",
        default="all",
        choices=("all", "train", "val"),
        help="Dataset split to iterate.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Enable synthetic sparse-profile mode while iterating.",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=20,
        help="Stop after reporting this many errors.",
    )
    return parser.parse_args()


def _is_valid_non_leap_yyyymmdd(date_value: int) -> bool:
    if int(date_value) == 20100101:
        return True
    month = (int(date_value) // 100) % 100
    day = int(date_value) % 100
    month_lengths = {
        1: 31,
        2: 28,
        3: 31,
        4: 30,
        5: 31,
        6: 30,
        7: 31,
        8: 31,
        9: 30,
        10: 31,
        11: 30,
        12: 31,
    }
    if month not in month_lengths:
        return False
    return 1 <= day <= month_lengths[month]


def main() -> int:
    args = _parse_args()
    dataset = OstiaArgoTiffDataset.from_config(args.data_config, split=args.split)
    if args.synthetic:
        dataset.synthetic_mode = True

    error_count = 0
    total = len(dataset)
    print(
        f"Checking dataset: split={args.split}, synthetic={dataset.synthetic_mode}, "
        f"length={total}, config={Path(args.data_config)}"
    , flush=True)

    for idx in range(total):
        row = dataset._rows[idx]
        try:
            sample = dataset[idx]
            date_value = int(sample["date"])
            if not _is_valid_non_leap_yyyymmdd(date_value):
                raise ValueError(
                    "date is incompatible with model date conditioning "
                    f"(fixed non-leap YYYYMMDD): {date_value}"
                )
        except Exception as exc:
            error_count += 1
            print(
                f"[ERROR] idx={idx} export_index={row.get('export_index', '')} "
                f"date={row.get('date', '')} error={exc}"
            , flush=True)
            if error_count >= args.max_errors:
                print(f"Stopping after {error_count} errors.", flush=True)
                return 1

        if (idx + 1) % 1000 == 0 or idx + 1 == total:
            print(f"Checked {idx + 1}/{total} samples, errors={error_count}", flush=True)

    print(f"Finished dataset check with errors={error_count}", flush=True)
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

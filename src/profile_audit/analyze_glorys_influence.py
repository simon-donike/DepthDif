from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from profile_audit.common import read_table, write_table


def analyze_glorys_influence(samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize observation-minus-GLORYS residuals by relative day.

    Input rows are depth-level samples and must include ``profile_id``,
    ``relative_day``, ``observation_temperature_c``, and
    ``glorys_temperature_c``. Supplying +/-14-day and pseudo-profile rows is
    intentionally left to the data acquisition step.
    """
    required = {"profile_id", "relative_day", "observation_temperature_c", "glorys_temperature_c"}
    missing = required - set(samples)
    if missing:
        raise ValueError(f"Influence samples are missing columns: {sorted(missing)}")
    rows = samples.copy()
    rows["residual_c"] = rows["observation_temperature_c"] - rows["glorys_temperature_c"]
    rows["absolute_residual_c"] = rows["residual_c"].abs()
    rows["squared_residual_c2"] = rows["residual_c"] ** 2
    group_columns = [column for column in ("relative_day", "is_pseudo", "cruise", "archive_match_class") if column in rows]
    summary = rows.groupby(group_columns, dropna=False).agg(
        sample_count=("residual_c", "size"),
        profile_count=("profile_id", "nunique"),
        mae_c=("absolute_residual_c", "mean"),
        rmse_c=("squared_residual_c2", lambda values: float(np.sqrt(values.mean()))),
        mean_bias_c=("residual_c", "mean"),
    ).reset_index()
    return rows, summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze weak public-data evidence of GLORYS influence.")
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--residuals-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    residuals, summary = analyze_glorys_influence(read_table(args.samples))
    print(write_table(residuals, args.residuals_output))
    print(write_table(summary, args.summary_output))


if __name__ == "__main__":
    main()

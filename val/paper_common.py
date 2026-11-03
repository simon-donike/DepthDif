"""Shared data contracts and publication exports for isolated validation stages."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/depthdif-val-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class Bundle:
    """Aligned physical observations and named sample matrices, in point row order."""

    points: pd.DataFrame
    members: dict[str, np.ndarray]
    metadata: dict[str, Any]
    path: Path | None = None

    def subset(self, mask: np.ndarray) -> "Bundle":
        """Select rows without breaking shared ensemble-member identities."""
        return Bundle(
            self.points.loc[mask].reset_index(drop=True).copy(),
            {name: values[mask].copy() for name, values in self.members.items()},
            dict(self.metadata),
            self.path,
        )


@dataclass
class Context:
    """Separate evaluation cases from the only cases permitted for calibration."""

    evaluation: Bundle
    calibration: Bundle | None
    config: dict[str, Any]
    seed: int


def point_crps(
    members: np.ndarray, truth: np.ndarray, fair: bool = False
) -> np.ndarray:
    """Compute marginal CRPS; fair correction assumes iid sampled members."""
    values = np.asarray(members, dtype=float)
    observed = np.asarray(truth, dtype=float).reshape(-1)
    if values.ndim != 2 or len(values) != len(observed) or not values.shape[1]:
        raise ValueError("members must have shape (number of observations, members)")
    count = values.shape[1]
    if fair and count < 2:
        raise ValueError("Fair CRPS needs at least two iid members")
    ordered = np.sort(values, axis=1)
    ranks = 2 * np.arange(1, count + 1) - count - 1
    denominator = count * (count - 1 if fair else count)
    return (
        np.abs(values - observed[:, None]).mean(axis=1)
        - (ordered @ ranks) / denominator
    )


def _tex(value: Any) -> str:
    """Escape cell content without requiring a LaTeX installation or Jinja."""
    substitutions = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    if isinstance(value, (float, np.floating)):
        return "--" if not np.isfinite(value) else f"{value:.4g}"
    return "".join(substitutions.get(char, char) for char in str(value))


def save_table(
    frame: pd.DataFrame, output_dir: Path, name: str, caption: str = ""
) -> list[str]:
    """Export identical machine-readable CSV and portable booktabs LaTeX tables."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path, tex_path = output_dir / f"{name}.csv", output_dir / f"{name}.tex"
    frame.to_csv(csv_path, index=False)
    lines = [r"\begin{table}[htbp]", r"\centering"]
    if caption:
        lines.append(r"\caption{" + _tex(caption) + "}")
    lines += [r"\begin{tabular}{" + "l" * len(frame.columns) + "}", r"\toprule"]
    lines.append(" & ".join(_tex(column) for column in frame.columns) + r" \\")
    lines.append(r"\midrule")
    lines += [
        " & ".join(_tex(value) for value in row) + r" \\"
        for row in frame.itertuples(index=False, name=None)
    ]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex_path.write_text("\n".join(lines) + "\n")
    return [str(csv_path), str(tex_path)]


def save_figure(figure: Any, output_dir: Path, name: str) -> list[str]:
    """Save vector PDF/SVG plus a 300 dpi raster preview and release memory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for extension in ("pdf", "svg", "png"):
        path = output_dir / f"{name}.{extension}"
        figure.savefig(path, dpi=300, bbox_inches="tight")
        artifacts.append(str(path))
    plt.close(figure)
    return artifacts


def stable_seed(value: str, seed: int) -> int:
    """Create process-independent random seeds for named evaluation strata."""
    return int.from_bytes(
        hashlib.sha256(f"{seed}:{value}".encode()).digest()[:4], "little"
    )


def profile_mean(points: pd.DataFrame, loss: np.ndarray) -> float:
    """Weight each profile equally rather than each available depth sample."""
    return float(
        pd.DataFrame({"profile_id": points.profile_id.to_numpy(), "loss": loss})
        .groupby("profile_id")
        .loss.mean()
        .mean()
    )

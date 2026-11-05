"""Sequential scientific validation suite and portable publication index."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import html
from importlib import import_module
import json
from pathlib import Path
import time
import traceback
from typing import Any

import numpy as np
import pandas as pd

from val.paper_common import Bundle, Context, save_table, stable_seed

STAGES = (
    "reference",
    "comparators",
    "baselines",
    "regional",
    "conditioning",
    "physical",
    "dependence",
    "extremes",
    "selective",
    "efficiency",
)


def _digest(path: Path) -> str:
    """Hash persisted inputs before calibration or scoring."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json(path: Path, value: Any) -> None:
    """Serialize status and provenance with portable nonfinite values."""

    def clean(item):
        if isinstance(item, dict):
            return {str(key): clean(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(val) for val in item]
        if isinstance(item, np.ndarray):
            return clean(item.tolist())
        if isinstance(item, (float, np.floating)) and not np.isfinite(item):
            return None
        if isinstance(item, np.generic):
            return item.item()
        return item

    path.write_text(
        json.dumps(clean(value), indent=2, default=str, allow_nan=False) + "\n"
    )


def load_bundle(path: Path) -> Bundle:
    """Read point-aligned samples and reject altered or invalid saved inputs."""
    path = Path(path).resolve()
    metadata = json.loads((path / "metadata.json").read_text())
    if not metadata.get("demo") and set(metadata.get("bundle_sha256", {})) != {
        "points.csv",
        "members.npz",
    }:
        raise ValueError(
            "Scientific bundles require recorded SHA256 hashes for points.csv and members.npz"
        )
    for name, digest in metadata.get("bundle_sha256", {}).items():
        if name not in {"points.csv", "members.npz"} or _digest(path / name) != digest:
            raise ValueError("Saved bundle contents differ from their recorded hashes")
    points = pd.read_csv(path / "points.csv")
    if points.point_id.isna().any() or points.point_id.duplicated().any():
        raise ValueError("Saved bundle point identities must be unique and populated")
    with np.load(path / "members.npz", allow_pickle=False) as saved:
        members = {name: saved[name].copy() for name in saved.files}
    for name, values in members.items():
        if (
            values.ndim != 2
            or values.shape[0] != len(points)
            or values.shape[1] < 1
            or not np.isfinite(values).all()
        ):
            raise ValueError(f"Invalid or unaligned ensemble: {name}")
    if "residual" not in members:
        raise ValueError("Saved bundle requires residual members")
    for column in points:
        if column.startswith("baseline_"):
            members.setdefault(
                column.removeprefix("baseline_"),
                points[column].to_numpy(float)[:, None],
            )
    return Bundle(points, members, metadata, path)


def _assert_disjoint(evaluation: Bundle, calibration: Bundle) -> None:
    """Block calibration leakage by point, profile, cluster and physical cast."""
    for key in ("point_id", "profile_id", "cluster_id"):
        overlap = set(evaluation.points[key].astype(str)) & set(
            calibration.points[key].astype(str)
        )
        if overlap:
            raise ValueError(
                f"Calibration/evaluation {key} overlap: {sorted(overlap)[:3]}"
            )
    # Differently named copies of a cast must not evade the identity checks.
    keys = ["observation_time", "latitude", "longitude", "variable", "depth_m"]
    if all(key in evaluation.points and key in calibration.points for key in keys):

        def fingerprints(points):
            frame = points[keys].copy()
            frame["observation_time"] = pd.to_datetime(
                frame.observation_time, utc=True, format="mixed"
            ).astype(str)
            return set(frame.astype(str).itertuples(index=False, name=None))

        if fingerprints(evaluation.points) & fingerprints(calibration.points):
            raise ValueError(
                "Calibration/evaluation contain a duplicate physical observation"
            )


def make_context(bundle: Bundle, config: dict[str, Any], output: Path) -> Context:
    """Freeze disjoint calibration units; never auto-fit on confirmation labels."""
    options = config.get("paper", {})
    seed = int(config.get("sampling", {}).get("seed", 20261103))
    calibration = None
    evaluation = bundle.subset(np.ones(len(bundle.points), dtype=bool))
    role = set(bundle.points.split_role.astype(str))
    if len(role) != 1:
        raise ValueError(
            "A bundle must have a single discovery or confirmation split role"
        )
    split_method = "none: calibration-dependent stages will be blocked"
    if options.get("calibration_bundle"):
        calibration = load_bundle(Path(options["calibration_bundle"]))
        _assert_disjoint(evaluation, calibration)
        comparison_keys = ("checkpoint_sha256", "config_sha256", "weight_variant")
        compatible = all(
            evaluation.metadata.get(key) is not None
            and evaluation.metadata.get(key) == calibration.metadata.get(key)
            for key in comparison_keys
        )
        if not compatible:
            # GLORYS/observation calibration remains useful across models, but a
            # different model's members cannot define this model's error scale.
            calibration.members = {}
            calibration.metadata["members_withheld_reason"] = (
                "Calibration model/checkpoint identity is missing or differs from evaluation"
            )
        split_method = "explicit external calibration bundle"
    elif role == {"discovery"} and options.get("calibration_fraction", 0.25) > 0:
        fraction = float(options.get("calibration_fraction", 0.25))
        if not 0 < fraction < 1:
            raise ValueError("paper.calibration_fraction must lie between zero and one")
        clusters = sorted(
            bundle.points.cluster_id.astype(str).unique(),
            key=lambda value: stable_seed(value, seed),
        )
        if len(clusters) >= 2:
            number = min(len(clusters) - 1, max(1, round(len(clusters) * fraction)))
            chosen = set(clusters[:number])
            calibration_mask = (
                bundle.points.cluster_id.astype(str).isin(chosen).to_numpy()
            )
            calibration, evaluation = bundle.subset(calibration_mask), bundle.subset(
                ~calibration_mask
            )
            _assert_disjoint(evaluation, calibration)
            split_method = "frozen cluster holdout within discovery data; not untouched confirmation"
    elif role - {"confirmation", "discovery"}:
        raise ValueError("Unknown split_role in saved bundle")
    split = []
    for label, data in (("evaluation", evaluation), ("calibration", calibration)):
        if data is not None:
            frame = data.points[
                ["point_id", "profile_id", "cluster_id", "variable"]
            ].copy()
            frame["partition"] = label
            split.append(frame)
    membership = pd.concat(split, ignore_index=True)
    membership.to_csv(output / "partition.csv", index=False)
    counts = membership.groupby(["partition", "variable"], as_index=False).agg(
        points=("point_id", "size"),
        profiles=("profile_id", "nunique"),
        clusters=("cluster_id", "nunique"),
    )
    save_table(
        counts,
        output,
        "partition_summary",
        "Frozen calibration and evaluation support; complete point membership is in partition.csv",
    )
    _json(
        output / "split.json",
        {
            "method": split_method,
            "seed": seed,
            "evaluation_points": len(evaluation.points),
            "calibration_points": 0 if calibration is None else len(calibration.points),
            "calibration_input": (
                str(calibration.path) if calibration and calibration.path else None
            ),
            "calibration_sha256": (
                {
                    name: _digest(calibration.path / name)
                    for name in ("points.csv", "members.npz", "metadata.json")
                }
                if calibration and calibration.path
                else None
            ),
        },
    )
    return Context(evaluation, calibration, config, seed)


def _index(output: Path, records: list[dict], context: Context) -> None:
    """Publish one local HTML index with linked tables and embedded figures."""
    demo = bool(context.evaluation.metadata.get("demo"))
    sections = [
        "<!doctype html><html><head><meta charset='utf-8'><title>DepthDif validation</title>",
        "<style>body{max-width:1200px;margin:3em auto;font:16px system-ui;padding:0 1em}table{border-collapse:collapse}td,th{padding:.4em;border-bottom:1px solid #ddd}img{max-width:100%}.blocked,.failed{color:#a22}.partial{color:#965a00}pre{white-space:pre-wrap}</style></head><body>",
        "<h1>" + ("SYNTHETIC DEMO — " if demo else "") + "Frozen-model validation</h1>",
        "<p>Scores use held-out evaluation profiles. Calibration labels are kept in a separate whole-cluster partition. Lower proper scores indicate better forecasts; uncertainty width alone does not establish benefit. Missing stages are listed explicitly.</p>",
        "<p><a href='stages.csv'>Stage status (CSV)</a> · <a href='manifest.json'>Provenance manifest</a> · <a href='split.json'>Calibration split</a> · <a href='partition.csv'>Point membership</a></p>",
    ]
    if demo:
        sections.append(
            "<p><strong>All observations and predictions are synthetic. This report is a software demonstration and supplies no scientific evidence.</strong></p>"
        )
    for record in records:
        sections += [
            f"<section><h2>{record['order']}. {html.escape(record['technique'])}</h2>",
            f"<p class='{html.escape(record['status'])}'><strong>{html.escape(record['status'])}</strong>: {html.escape(record['summary'])}</p>",
        ]
        seen_figures = set()
        for artifact in record.get("artifacts", []):
            path = Path(artifact)
            if not path.is_absolute():
                path = path.resolve()
            if not path.is_file() or output.resolve() not in path.parents:
                continue
            relative = path.relative_to(output.resolve()).as_posix()
            sections.append(
                f"<a href='{html.escape(relative)}'>{html.escape(path.name)}</a> "
            )
            if path.suffix == ".png" and path.stem not in seen_figures:
                sections.append(
                    f"<div><img loading='lazy' src='{html.escape(relative)}' alt='{html.escape(path.stem)}'></div>"
                )
                seen_figures.add(path.stem)
            if path.suffix == ".csv" and path.stat().st_size < 50000:
                sections.append(
                    pd.read_csv(path)
                    .head(25)
                    .to_html(index=False, float_format=lambda x: f"{x:.4g}")
                )
        sections.append(
            "<details><summary>Availability and scientific scope</summary><pre>"
            + html.escape(json.dumps(record.get("details", {}), indent=2, default=str))
            + "</pre></details></section>"
        )
    sections.append("</body></html>")
    (output / "index.html").write_text("\n".join(sections))


def run_suite(
    bundle_path: Path, output: Path, config: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Run every independently implemented technique in a fixed sequence."""
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Refusing to mix an existing publication report with new output: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    config = config or {}
    context = make_context(load_bundle(bundle_path), config, output)
    records = []
    for order, name in enumerate(STAGES, 1):
        stage_output = output / f"{order:02d}_{name}"
        stage_output.mkdir()
        options = dict(config.get("paper", {}).get(name, {}))
        if name == "comparators" and options.get("cavaiola2026"):
            options["_defer_cavaiola"] = True
        options.setdefault("seed", context.seed)
        print(f"Validation {order}/{len(STAGES)}: {name}", flush=True)
        started = time.monotonic()
        try:
            if options.get("enabled", True):
                result = import_module(f"val.techniques.{name}").run(
                    context, stage_output, options
                )
            else:
                result = {
                    "status": "blocked",
                    "summary": "Disabled explicitly in the frozen configuration",
                    "artifacts": [],
                    "details": {},
                }
            if result.get("status") not in {"complete", "partial", "blocked"}:
                raise ValueError(
                    f"Technique returned invalid status: {result.get('status')}"
                )
        except Exception as exc:
            (stage_output / "traceback.txt").write_text(traceback.format_exc())
            result = {
                "status": "failed",
                "summary": f"{type(exc).__name__}: {exc}",
                "artifacts": [],
                "details": {"traceback": str(stage_output / "traceback.txt")},
            }
        record = {
            "order": order,
            "technique": name,
            "seconds": time.monotonic() - started,
            **result,
        }
        records.append(record)
        _json(stage_output / "status.json", record)
        if name == "baselines" and config.get("paper", {}).get("comparators", {}).get(
            "cavaiola2026"
        ):
            from val.cavaiola import run as run_cavaiola

            # The diagnostic includes the calibration-only GLORYS/climatology
            # methods as well as all trained baselines, once fitting is complete.
            comparator = next(
                row for row in records if row["technique"] == "comparators"
            )
            directory = output / f"{comparator['order']:02d}_comparators"
            try:
                result = run_cavaiola(
                    context, directory, config["paper"]["comparators"]["cavaiola2026"]
                )
                comparator["artifacts"] += result["artifacts"]
                comparator["details"]["cavaiola2026"] = result
                for row in comparator["details"]["comparators"]:
                    if row["method"] == "cavaiola2026":
                        row.update(
                            status=(
                                "available"
                                if result["status"] == "complete"
                                else "blocked"
                            ),
                            information=result["summary"],
                        )
                comparator["status"] = (
                    (
                        "partial"
                        if any(
                            row["status"] != "available"
                            for row in comparator["details"]["comparators"]
                        )
                        else "complete"
                    )
                    if any(
                        row["status"] == "available"
                        for row in comparator["details"]["comparators"]
                    )
                    else "blocked"
                )
                comparator["summary"] = (
                    "Compared available methods; see explicit baseline availability and the Cavaiola diagnostic scorecard"
                )
                save_table(
                    pd.DataFrame(comparator["details"]["comparators"]),
                    directory,
                    "comparator_availability",
                    "AI comparators, matched support and declared information differences",
                )
            except Exception as exc:
                (directory / "cavaiola_traceback.txt").write_text(
                    traceback.format_exc()
                )
                comparator.update(
                    status="failed", summary=f"Cavaiola: {type(exc).__name__}: {exc}"
                )
            _json(directory / "status.json", comparator)
        _json(output / "stages.json", records)
    save_table(
        pd.DataFrame(
            [
                {
                    key: row[key]
                    for key in ("order", "technique", "status", "seconds", "summary")
                }
                for row in records
            ]
        ),
        output,
        "stages",
        "Sequential validation stage availability",
    )
    context.evaluation.points.to_csv(output / "evaluation_points.csv", index=False)
    np.savez_compressed(output / "evaluation_members.npz", **context.evaluation.members)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "demo": bool(context.evaluation.metadata.get("demo")),
        "source_bundle": str(Path(bundle_path).resolve()),
        "input_sha256": {
            name: _digest(Path(bundle_path) / name)
            for name in ("points.csv", "members.npz", "metadata.json")
        },
        "configuration": config,
        "metadata": context.evaluation.metadata,
        "workflow_sha256": {
            str(path.relative_to(Path(__file__).parent)): _digest(path)
            for path in Path(__file__).parent.rglob("*.py")
            if "vendor" not in path.parts and "results" not in path.parts
        },
        "stages": records,
        "claim": "Scientific benefit requires the paired comparisons and independence gates; stage completion alone is not evidence of superiority.",
    }
    _json(output / "manifest.json", manifest)
    _index(output, records, context)
    failed = [row["technique"] for row in records if row["status"] == "failed"]
    result = {
        "status": (
            "failed"
            if failed
            else (
                "partial"
                if any(row["status"] != "complete" for row in records)
                else "complete"
            )
        ),
        "failed_stages": failed,
        "report": str(output / "index.html"),
        "stages": len(records),
    }
    _json(output / "status.json", result)
    if failed:
        raise RuntimeError(
            "Validation stages failed: "
            + ", ".join(failed)
            + f"; inspect {output / 'index.html'}"
        )
    return result

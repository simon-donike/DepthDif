# Example: /work/envs/depth/bin/python -m depth_recon.scripts.prepare_hpc_handoff stage --dataset-root /work/data/OceanVariableReconstruction --climatology /work/code/DepthDif/outputs/checkpoint_selection_20261005/baselines/temperature_monthly_stride4.npz --target-bias /work/code/DepthDif/outputs/temperature_bias_excluding_2016.npz --italian-profiles /work/code/DepthDif/val/results/xbt_reference/profiles.csv --italian-manifest /work/code/DepthDif/val/results/xbt_reference/manifest.json --italian-screening /work/code/DepthDif/evaluations/outputs/italian_en4_no_match_profiles.parquet --source-archive /work/code/DepthDif/evaluations/focus_evaluation.tar.gz --output-dir /tmp/hpc_handoff/hpc_residual --hpc-destination /lustre/projects/1000/rubenc/ocean_data/ocean_data
"""Create and strictly verify the offline asset bundle for the corrected HPC run."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import yaml

from depth_recon.data.target_bias import TargetBias

DEFAULT_DATASET_ROOT = Path("/work/data/OceanVariableReconstruction")
DEFAULT_CLIMATOLOGY = Path(
    "/work/code/DepthDif/outputs/checkpoint_selection_20261005/baselines/"
    "temperature_monthly_stride4.npz"
)
DEFAULT_HPC_DEST = Path("/lustre/projects/1000/rubenc/ocean_data/ocean_data")
DEFAULT_PROFILES = Path("/work/code/DepthDif/val/results/xbt_reference/profiles.csv")
DEFAULT_MANIFEST = Path("/work/code/DepthDif/val/results/xbt_reference/manifest.json")
DEFAULT_SCREENING = Path(
    "/work/code/DepthDif/evaluations/outputs/italian_en4_no_match_profiles.parquet"
)
DEFAULT_ARCHIVE = Path("/work/code/DepthDif/evaluations/focus_evaluation.tar.gz")
EXPECTED_ITALIAN_REFERENCE_MANIFEST_SHA256 = (
    "cd4eade88dc458298632692863b414784f4c8cdb545a1e1c77ee31f576020771"
)

# These directories are generated metadata or repository state, never training inputs.
EXCLUDED_DATASET_PARTS = {
    ".cache",
    ".git",
    "__pycache__",
    "depthdif_cache",
    "hpc_residual",
}
REQUIRED_ASSET_PATHS = (
    "climatology/temperature_monthly_stride4.npz",
    "target_bias/temperature_bias_excluding_2016.npz",
    "italian/profiles.csv",
    "italian/reference_manifest.json",
    "italian/screened_profiles.parquet",
    "italian/focus_evaluation.tar.gz",
)


@dataclass(frozen=True)
class Asset:
    """An input file and its fixed relative destination in the HPC bundle."""

    source: Path
    relative: str


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest without loading a file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_under(path: Path, parent: Path) -> bool:
    """Return whether ``path`` is contained by ``parent`` after resolution."""
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _iter_dataset_files(
    root: Path, excluded_roots: Iterable[Path] = ()
) -> Iterator[Path]:
    """Yield relevant regular files while excluding known generated/cache trees."""
    resolved_excluded = tuple(path.resolve() for path in excluded_roots)
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in EXCLUDED_DATASET_PARTS for part in relative.parts):
            continue
        if any(_is_under(path, excluded) for excluded in resolved_excluded):
            continue
        if path.is_file():
            yield path


def _prior_entries(prior: Path | None) -> dict[str, dict[str, Any]]:
    """Read a prior inventory's file records when explicitly requested."""
    if prior is None or not prior.is_file():
        return {}
    payload = json.loads(prior.read_text(encoding="utf-8"))
    return {str(item["path"]): item for item in payload.get("files", [])}


def inventory_dataset(
    root: Path,
    *,
    prior_inventory: Path | None = None,
    full_hash: bool = True,
    reuse_prior_hashes: bool = False,
    excluded_roots: Iterable[Path] = (),
) -> list[dict[str, Any]]:
    """Inventory every relevant dataset file, with observable streaming progress.

    A full inventory hashes every byte by default. Reusing an old digest is an
    explicit staging shortcut and strict preflight always rehashes transferred bytes.
    """
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {root}")
    files = list(_iter_dataset_files(root, excluded_roots))
    prior = _prior_entries(prior_inventory) if reuse_prior_hashes else {}
    total_bytes = sum(path.stat().st_size for path in files)
    done_bytes = 0
    entries: list[dict[str, Any]] = []
    for number, path in enumerate(files, start=1):
        relative = path.relative_to(root).as_posix()
        stat = path.stat()
        old = prior.get(relative)
        reusable = (
            old is not None
            and old.get("size") == stat.st_size
            and old.get("mtime_ns") == stat.st_mtime_ns
            and isinstance(old.get("sha256"), str)
        )
        if full_hash and not reusable:
            digest, hash_status = sha256_file(path), "full"
        elif reusable:
            digest, hash_status = old["sha256"], "reused"
        else:
            digest, hash_status = None, "stat-only"
        done_bytes += stat.st_size
        entries.append(
            {
                "path": relative,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": digest,
                "hash_status": hash_status,
            }
        )
        if number % 1000 == 0 or number == len(files):
            print(
                f"inventory: {number}/{len(files)} files, "
                f"{done_bytes / 1024**3:.1f}/{total_bytes / 1024**3:.1f} GiB",
                file=sys.stderr,
                flush=True,
            )
    return entries


def _copy_asset(asset: Asset, output: Path) -> dict[str, Any]:
    """Copy one required bundle asset and return a checksummed record."""
    if not asset.source.is_file():
        raise FileNotFoundError(
            f"required handoff asset does not exist: {asset.source}"
        )
    destination = output / asset.relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(asset.source, destination)
    return {
        "path": asset.relative,
        "source": str(asset.source.resolve()),
        "size": destination.stat().st_size,
        "sha256": sha256_file(destination),
    }


def _assets(args: argparse.Namespace) -> list[Asset]:
    """Build the complete, fixed small-asset roster for this run."""
    return [
        Asset(args.climatology, REQUIRED_ASSET_PATHS[0]),
        Asset(args.target_bias, REQUIRED_ASSET_PATHS[1]),
        Asset(args.italian_profiles, REQUIRED_ASSET_PATHS[2]),
        Asset(args.italian_manifest, REQUIRED_ASSET_PATHS[3]),
        Asset(args.italian_screening, REQUIRED_ASSET_PATHS[4]),
        Asset(args.source_archive, REQUIRED_ASSET_PATHS[5]),
    ]


def _write_json(path: Path, value: Any) -> None:
    """Write stable, human-readable JSON metadata."""
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def stage(args: argparse.Namespace) -> int:
    """Stage required assets and a complete inventory for a copied dataset tree."""
    if not args.inventory_only and args.target_bias is None:
        raise ValueError("--target-bias is required for a corrected-target handoff")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset_root = args.dataset_root.resolve()
    if (
        not args.inventory_only
        and args.italian_manifest.resolve() == DEFAULT_MANIFEST.resolve()
        and sha256_file(args.italian_manifest)
        != EXPECTED_ITALIAN_REFERENCE_MANIFEST_SHA256
    ):
        raise ValueError(
            "the Italian reference manifest does not match the approved frozen artifact"
        )
    excluded = [output] if _is_under(output, dataset_root) else []
    inventory = inventory_dataset(
        dataset_root,
        prior_inventory=args.prior_inventory,
        full_hash=args.full_hash,
        reuse_prior_hashes=args.reuse_prior_hashes,
        excluded_roots=excluded,
    )
    if not args.inventory_only and not args.full_hash:
        raise ValueError("a corrected-target handoff requires --full-hash")
    staged = (
        []
        if args.inventory_only
        else [_copy_asset(asset, output) for asset in _assets(args)]
    )
    payload = {
        "schema_version": 2,
        "dataset_root_source": str(dataset_root),
        "dataset_hpc_destination": str(args.hpc_destination.resolve()),
        "bundle_relative_root": "hpc_residual",
        "dataset_copy_required": True,
        "dataset_hash_mode": "full" if args.full_hash else "stat-only",
        "files": inventory,
        "staged_assets": staged,
        "italian_reference_manifest_expected_sha256": (
            EXPECTED_ITALIAN_REFERENCE_MANIFEST_SHA256
            if not args.inventory_only
            and args.italian_manifest.resolve() == DEFAULT_MANIFEST.resolve()
            else None
        ),
    }
    _write_json(output / "dataset_inventory.json", payload)
    path_map = {
        "schema_version": 2,
        "bundle_relative_root": "hpc_residual",
        "dataset_source_root": str(dataset_root),
        "dataset_hpc_root": str(args.hpc_destination.resolve()),
        "mappings": [
            {
                "source": str(dataset_root),
                "hpc_destination": ".",
                "kind": "dataset_tree",
            },
            *[
                {
                    "source": record["source"],
                    "hpc_destination": f"hpc_residual/{record['path']}",
                    "kind": "bundle_asset",
                }
                for record in staged
            ],
        ],
        "climatology": REQUIRED_ASSET_PATHS[0],
        "target_bias": REQUIRED_ASSET_PATHS[1],
        "italian_profiles": REQUIRED_ASSET_PATHS[2],
        "italian_reference_manifest": REQUIRED_ASSET_PATHS[3],
        "italian_screening": REQUIRED_ASSET_PATHS[4],
        "italian_source_archive": REQUIRED_ASSET_PATHS[5],
    }
    _write_json(output / "path_map.json", path_map)
    print(f"staged {len(staged)} required assets into {output}")
    print(
        f"inventoried {len(inventory)} dataset files ({payload['dataset_hash_mode']} hash mode)"
    )
    return 0


def _check_hash(path: Path, expected: str) -> str | None:
    """Return an error string when a file is absent or has a wrong digest."""
    if not path.is_file():
        return "missing"
    actual = sha256_file(path)
    return None if actual == expected else f"sha256 mismatch: {actual} != {expected}"


def _relative_from_source(path_value: str, source_root: Path) -> str | None:
    """Resolve a manifest path against its recorded source root safely."""
    path = Path(path_value)
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.resolve().relative_to(source_root.resolve()).as_posix()
    except ValueError:
        return None


def _manifest_reference_paths(value: Any) -> set[str]:
    """Collect relative path values from exported dataset metadata recursively."""
    paths: set[str] = set()
    if isinstance(value, dict):
        for child_key, child in value.items():
            if child_key in {"path", "source"} and isinstance(child, str):
                candidate = Path(child)
                if child and not candidate.is_absolute():
                    paths.add(candidate.as_posix())
            paths.update(_manifest_reference_paths(child))
    elif isinstance(value, list):
        for child in value:
            paths.update(_manifest_reference_paths(child))
    return paths


def _required_manifest_files(
    dataset_root: Path, manifest: dict[str, Any]
) -> tuple[set[str], list[str]]:
    """Expand every manifest-referenced file/directory and report missing paths."""
    files: set[str] = {"manifest.yaml"}
    missing: list[str] = []
    references = _manifest_reference_paths(manifest)
    argo_path = manifest.get("argo", {}).get("path")
    if isinstance(argo_path, str) and not Path(argo_path).is_absolute():
        references.add(argo_path)
    grid_source = manifest.get("grid", {}).get("source")
    if isinstance(grid_source, str) and not Path(grid_source).is_absolute():
        references.add(grid_source)
    for relative in sorted(references):
        candidate = dataset_root / relative
        if candidate.is_file():
            files.add(relative)
        elif candidate.is_dir():
            regular = [path for path in candidate.rglob("*") if path.is_file()]
            if not regular:
                missing.append(f"dataset manifest referenced tree is empty: {relative}")
            files.update(path.relative_to(dataset_root).as_posix() for path in regular)
        else:
            missing.append(f"dataset manifest reference is missing: {relative}")
    return files, missing


def _validate_climatology(
    path: Path, depth: np.ndarray, grid: dict[str, Any]
) -> list[str]:
    """Validate the fitted climatology's physical and dataset compatibility metadata."""
    errors: list[str] = []
    try:
        with np.load(path, allow_pickle=False) as data:
            required = {"temperature", "count", "depth_axis_m", "metadata"}
            if not required <= set(data.files):
                return [
                    f"climatology missing arrays: {sorted(required - set(data.files))}"
                ]
            metadata = json.loads(str(data["metadata"]))
            artifact_depth = np.asarray(data["depth_axis_m"], dtype=np.float32).reshape(
                -1
            )
            if not np.array_equal(artifact_depth, depth.astype(np.float32)):
                errors.append("climatology depth axis does not match dataset manifest")
            if metadata.get("val_year_excluded") != 2016:
                errors.append("climatology does not record excluded year 2016")
            source_shape = (int(grid.get("height", 0)), int(grid.get("width", 0)))
            if tuple(metadata.get("source_grid_shape", ())) != source_shape:
                errors.append("climatology source grid does not match dataset manifest")
            stride = int(metadata.get("spatial_stride", 0))
            expected_grid = (
                tuple((size + stride - 1) // stride for size in source_shape)
                if stride
                else ()
            )
            if tuple(metadata.get("grid_shape", ())) != expected_grid:
                errors.append(
                    "climatology downsampled grid does not match its source grid"
                )
            expected_shape = (12, len(depth), *expected_grid)
            if np.asarray(data["temperature"]).shape != expected_shape:
                errors.append(
                    "climatology temperature shape does not match dataset grid/depth"
                )
            if np.asarray(data["count"]).shape != expected_shape:
                errors.append(
                    "climatology count shape does not match dataset grid/depth"
                )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        errors.append(f"climatology artifact cannot be read: {exc}")
    return errors


def _validate_target_bias(
    path: Path, depth: np.ndarray, grid: dict[str, Any], manifest_sha256: str
) -> list[str]:
    """Use the runtime loader and require exact provenance for frozen target bias."""
    try:
        bias = TargetBias(path, expected_depth_axis_m=depth)
        if bias.grid != grid:
            return ["target bias grid does not match dataset manifest"]
        if bias.metadata.get("excluded_years") != [2016]:
            return ["target bias does not record exactly excluded year 2016"]
        if (
            bias.metadata.get("input_hashes", {}).get("manifest.yaml")
            != manifest_sha256
        ):
            return [
                "target bias manifest provenance does not match the transferred dataset"
            ]
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return [f"target bias artifact is invalid: {exc}"]
    return []


def _validate_italian_manifest(
    bundle: Path,
    dataset_root: Path,
    source_dataset_root: Path,
    dataset_manifest: dict[str, Any],
    dataset_manifest_sha256: str,
    inventory_paths: set[str],
) -> list[str]:
    """Validate unchanged Italian provenance after its data paths are relocated."""
    errors: list[str] = []
    manifest_path = bundle / REQUIRED_ASSET_PATHS[3]
    profiles_path = bundle / REQUIRED_ASSET_PATHS[2]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            return ["Italian reference manifest must contain a JSON object"]
        expected_profiles = manifest.get("output_profiles_sha256")
        if not isinstance(expected_profiles, str) or _check_hash(
            profiles_path, expected_profiles
        ):
            errors.append("Italian prepared profiles do not match reference manifest")
        if manifest.get("dataset_manifest_sha256") != dataset_manifest_sha256:
            errors.append(
                "Italian reference manifest dataset manifest hash does not match relocated dataset"
            )
        # The exported dataset records its original root even after relocation.
        original_root = Path(str(dataset_manifest.get("output_dir", "")))
        if not original_root.is_absolute():
            original_root = Path(str(manifest.get("dataset_manifest", ""))).parent
        if not original_root.is_absolute():
            original_root = source_dataset_root
        for item in manifest.get("salinity_rasters", []):
            if (
                not isinstance(item, dict)
                or not item.get("path")
                or not item.get("sha256")
            ):
                errors.append(
                    "Italian reference manifest has an incomplete salinity raster record"
                )
                continue
            relative = _relative_from_source(str(item["path"]), original_root)
            if relative is None:
                errors.append(
                    f"Italian salinity path is outside original dataset root: {item['path']}"
                )
                continue
            if relative not in inventory_paths:
                errors.append(
                    f"Italian salinity reference is omitted from dataset inventory: {relative}"
                )
            error = _check_hash(dataset_root / relative, str(item["sha256"]))
            if error:
                errors.append(
                    f"relocated Italian salinity reference {relative}: {error}"
                )
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"Italian reference manifest is invalid: {exc}")
    return errors


def preflight(args: argparse.Namespace) -> int:
    """Strictly hash a copied HPC bundle and all relevant dataset files read-only."""
    bundle = args.bundle_dir.resolve()
    inventory_path = (
        args.inventory.resolve()
        if args.inventory
        else bundle / "dataset_inventory.json"
    )
    errors: list[str] = []
    try:
        payload = json.loads(inventory_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(
            f"ERROR: inventory is missing or invalid: {inventory_path}: {exc}",
            file=sys.stderr,
        )
        return 2
    if payload.get("schema_version") != 2:
        errors.append("inventory schema version is unsupported")
    if payload.get("dataset_hash_mode") != "full":
        errors.append("strict preflight requires a full dataset inventory")
    records = payload.get("files", [])
    if not isinstance(records, list) or not records:
        errors.append("dataset inventory is empty")
    try:
        path_map = json.loads((bundle / "path_map.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        path_map = {}
        errors.append(f"path map is missing or invalid: {exc}")
    dataset_root = (
        args.dataset_root.resolve()
        if args.dataset_root
        else Path(str(path_map.get("dataset_hpc_root", ""))).resolve()
    )
    source_root = Path(str(payload.get("dataset_root_source", "")))
    if not source_root.is_absolute():
        errors.append("inventory does not record an absolute source dataset root")
    staged = {
        str(item.get("path")): item
        for item in payload.get("staged_assets", [])
        if isinstance(item, dict)
    }
    for relative in REQUIRED_ASSET_PATHS:
        record = staged.get(relative)
        if record is None or not isinstance(record.get("sha256"), str):
            errors.append(f"required asset missing from inventory roster: {relative}")
            continue
        error = _check_hash(bundle / relative, record["sha256"])
        if error:
            errors.append(f"{relative}: {error}")
    frozen_manifest_hash = payload.get("italian_reference_manifest_expected_sha256")
    if frozen_manifest_hash is not None:
        if frozen_manifest_hash != EXPECTED_ITALIAN_REFERENCE_MANIFEST_SHA256:
            errors.append(
                "inventory records an unexpected Italian reference manifest identity"
            )
        elif _check_hash(bundle / REQUIRED_ASSET_PATHS[3], frozen_manifest_hash):
            errors.append(
                "Italian reference manifest is not the approved frozen artifact"
            )
    inventory_paths: set[str] = set()
    total = len(records)
    for number, entry in enumerate(records, start=1):
        relative = entry.get("path") if isinstance(entry, dict) else None
        digest = entry.get("sha256") if isinstance(entry, dict) else None
        if not isinstance(relative, str) or not isinstance(digest, str):
            errors.append(f"invalid inventory record at position {number}")
            continue
        inventory_paths.add(relative)
        path = dataset_root / relative
        if not path.is_file():
            errors.append(f"dataset file is missing: {relative}")
            continue
        if path.stat().st_size != entry.get("size"):
            errors.append(f"dataset file size mismatch: {relative}")
            continue
        error = _check_hash(path, digest)
        if error:
            errors.append(f"dataset file {relative}: {error}")
        if number % 1000 == 0 or number == total:
            print(
                f"preflight: {number}/{total} files hashed", file=sys.stderr, flush=True
            )
    manifest_path = dataset_root / "manifest.yaml"
    if not manifest_path.is_file():
        errors.append(f"dataset manifest is missing: {manifest_path}")
    else:
        try:
            manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                raise ValueError("not a mapping")
            required_files, reference_errors = _required_manifest_files(
                dataset_root, manifest
            )
            errors.extend(reference_errors)
            missing_inventory = sorted(required_files - inventory_paths)
            errors.extend(
                f"manifest-referenced file omitted from inventory: {item}"
                for item in missing_inventory
            )
            depth = np.asarray(
                manifest.get("depth_axis_m", ()), dtype=np.float32
            ).reshape(-1)
            grid = manifest.get("grid", {})
            manifest_hash = sha256_file(manifest_path)
            errors.extend(
                _validate_climatology(bundle / REQUIRED_ASSET_PATHS[0], depth, grid)
            )
            errors.extend(
                _validate_target_bias(
                    bundle / REQUIRED_ASSET_PATHS[1], depth, grid, manifest_hash
                )
            )
            if source_root.is_absolute():
                errors.extend(
                    _validate_italian_manifest(
                        bundle,
                        dataset_root,
                        source_root,
                        manifest,
                        manifest_hash,
                        inventory_paths,
                    )
                )
        except (OSError, ValueError, yaml.YAMLError) as exc:
            errors.append(f"dataset manifest is invalid: {exc}")
    if errors:
        print("HPC preflight FAILED:")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"HPC preflight OK: {bundle}")
    print(f"strictly hashed dataset inventory entries: {len(records)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for staging and strict preflight verification."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    stage_parser = subparsers.add_parser(
        "stage", help="stage assets and fully inventory the dataset"
    )
    stage_parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    stage_parser.add_argument("--climatology", type=Path, default=DEFAULT_CLIMATOLOGY)
    stage_parser.add_argument("--target-bias", type=Path)
    stage_parser.add_argument("--italian-profiles", type=Path, default=DEFAULT_PROFILES)
    stage_parser.add_argument("--italian-manifest", type=Path, default=DEFAULT_MANIFEST)
    stage_parser.add_argument(
        "--italian-screening", type=Path, default=DEFAULT_SCREENING
    )
    stage_parser.add_argument("--source-archive", type=Path, default=DEFAULT_ARCHIVE)
    stage_parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_HPC_DEST / "hpc_residual"
    )
    stage_parser.add_argument("--hpc-destination", type=Path, default=DEFAULT_HPC_DEST)
    stage_parser.add_argument("--prior-inventory", type=Path)
    stage_parser.add_argument("--reuse-prior-hashes", action="store_true")
    stage_parser.add_argument(
        "--full-hash", action=argparse.BooleanOptionalAction, default=True
    )
    stage_parser.add_argument("--inventory-only", action="store_true")
    stage_parser.set_defaults(handler=stage)
    preflight_parser = subparsers.add_parser(
        "preflight", help="strictly verify a staged bundle read-only"
    )
    preflight_parser.add_argument("--bundle-dir", type=Path, required=True)
    preflight_parser.add_argument("--inventory", type=Path)
    preflight_parser.add_argument("--dataset-root", type=Path)
    preflight_parser.set_defaults(handler=preflight)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    """Run the handoff CLI and return a shell-compatible exit status."""
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())

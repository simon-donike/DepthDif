# Example: PYTHONPATH=val/vendor:src /work/envs/depth/bin/python -m val.score_italian_ensemble --cohort val/results/temperature_parallel_20261117/italian_xbt_temperature --ensemble val/results/italian_ensemble_20261117 --spatial val/results/xbt_spatial_all_20261117 --output docs/assets/validation/italian_xbt_temperature/ensemble --device cuda:0
"""Score Italian ensembles against XBT and compare field structure with GLORYS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import binary_erosion

from depth_recon.inference.export_wavenumber_spectra import radial_wavenumber_spectrum
from val.paper import _json, load_bundle
from val.run import _sha256


def point_scores(observations, members):
    """Compute empirical/fair CRPS and central 90% interval diagnostics per point."""
    values = np.asarray(members, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 1 or not np.isfinite(values).all():
        raise ValueError("Require a finite point-by-member matrix")
    count = values.shape[1]
    first = np.abs(values - np.asarray(observations)[:, None]).mean(axis=1)
    ranks = 2 * np.arange(1, count + 1) - count - 1
    pairs = (np.sort(values, axis=1) * ranks).sum(axis=1)
    lower, upper = np.quantile(values, [0.05, 0.95], axis=1)
    return {
        "crps": first - pairs / count**2,
        "fair_crps": (
            first - pairs / (count * (count - 1))
            if count > 1
            else np.full(len(values), np.nan)
        ),
        "coverage90": (observations >= lower) & (observations <= upper),
        "width90": upper - lower,
        "mean_absolute_error": np.abs(values.mean(axis=1) - observations),
        "mean_squared_error": (values.mean(axis=1) - observations) ** 2,
    }


def score_profiles(points, predictions):
    """Retain equal cast weights overall, by mission and within 25 m depth bins."""
    rows, depth_rows, mission_rows = [], [], []
    for method, values in predictions.items():
        scores = points[["profile_id", "cluster_id", "depth_m"]].copy()
        scores = scores.assign(**point_scores(points.observation.to_numpy(), values))
        columns = [
            "crps",
            "fair_crps",
            "coverage90",
            "width90",
            "mean_absolute_error",
            "mean_squared_error",
        ]
        profile = scores.groupby("profile_id")[columns].mean()
        summary = {
            "method": method,
            "members": values.shape[1],
            "profiles": len(profile),
            "points": len(points),
            **profile.mean().to_dict(),
        }
        summary["ensemble_mean_rmse"] = np.sqrt(summary.pop("mean_squared_error"))
        rows.append(summary)
        scores["depth_lower_m"] = np.floor(scores.depth_m / 25) * 25
        for depth, group in scores.groupby("depth_lower_m"):
            depth_rows.append(
                {
                    "method": method,
                    "depth_lower_m": depth,
                    "profiles": group.profile_id.nunique(),
                    "points": len(group),
                    **group.groupby("profile_id")[columns].mean().mean().to_dict(),
                }
            )
        for mission, group in scores.groupby("cluster_id"):
            mission_rows.append(
                {
                    "method": method,
                    "mission": mission,
                    "profiles": group.profile_id.nunique(),
                    **group.groupby("profile_id")[columns].mean().mean().to_dict(),
                }
            )
    return rows, depth_rows, mission_rows


def lpips_layers(metric, values, reference, valid, device):
    """Repeat grayscale layers as RGB using fixed [-2, 35] °C scaling and wet interiors."""
    common = valid & np.isfinite(values) & np.isfinite(reference)
    # Erode the scoring mask to reduce coastal fill effects; convolutional
    # receptive fields can still cross masks, so this remains exploratory.
    interiors = np.stack([binary_erosion(mask, iterations=16) for mask in common])
    results = np.full(len(values), np.nan)
    for start in range(0, len(values), 8):
        stop = start + 8
        mask = common[start:stop]
        arrays = []
        for field in (values[start:stop], reference[start:stop]):
            scaled = np.where(
                mask, 2 * (np.clip(field, -2, 35) + 2) / 37 - 1, 0
            ).astype(np.float32)
            arrays.append(
                torch.from_numpy(scaled[:, None]).repeat(1, 3, 1, 1).to(device)
            )
        with torch.inference_mode():
            distance = metric(*arrays).cpu().numpy()[:, 0]
        for index, (layer, interior) in enumerate(zip(distance, interiors[start:stop])):
            if interior.sum() >= 32:
                results[start + index] = layer[interior].mean()
    return results


def spectral_error(power, reference):
    """Compare finite bins without dropping zero-power, over-smoothed forecasts."""
    valid = np.isfinite(power) & np.isfinite(reference)
    if not valid.any():
        return np.nan
    # A common physical floor makes zero-power bins comparable in log space.
    difference = np.log10(np.maximum(power[valid], 1e-12)) - np.log10(
        np.maximum(reference[valid], 1e-12)
    )
    return float(np.sqrt(np.mean(difference**2)))


def score(cohort_root, ensemble, spatial, output, device):
    """Require a complete verified ensemble, then write reproducible website metrics."""
    import lpips

    torch.set_num_threads(4)
    saved = load_bundle(cohort_root / "bundle")
    points = saved.points
    if not points.profile_id.str.startswith("italian-xbt:").all():
        raise ValueError("Italian cohort required")
    for shard in (0, 1):
        if not (ensemble / f"worker_{shard}" / "complete.json").is_file():
            raise ValueError("Both GPU workers must complete before publication")
    metric = lpips.LPIPS(net="vgg", spatial=True).to(device).eval()
    output.mkdir(parents=True, exist_ok=True)
    matrices, identities, spatial_rows, spectra_rows, manifests = [], [], [], [], []
    edges = np.geomspace(30, 1000, 33)
    centers = np.sqrt(edges[:-1] * edges[1:])
    common_patches = {
        int(path.stem.split("_")[1]) for path in spatial.glob("patch_*.npz")
    }
    if common_patches != set(points.patch_index):
        raise ValueError("Spatial archive must cover all Italian patches exactly")
    for number in range(len(saved.metadata["batch_records"])):
        directory = ensemble / f"batch_{number:02d}"
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["config_sha256"] != _sha256(cohort_root / "config.yaml"):
            raise ValueError("Frozen configuration changed after generation")
        if (
            manifest["input_sha256"]
            != saved.metadata["batch_records"][number]["input_sha256"]
            or manifest["source_bundle_sha256"] != saved.metadata["bundle_sha256"]
        ):
            raise ValueError("Ensemble source differs from frozen cohort")
        manifests.append(manifest)
        reference = np.load(directory / "reference.npz")
        members, ids, columns = [], None, []
        for member in range(manifest["members"]):
            path = directory / f"member_{member:03d}.npz"
            if (
                _sha256(path)
                != json.loads(path.with_suffix(".json").read_text())["sha256"]
            ):
                raise ValueError("Generated sample checksum differs")
            with np.load(path) as sample:
                if ids is None:
                    ids = sample["point_ids"].copy()
                elif not np.array_equal(ids, sample["point_ids"]):
                    raise ValueError("Member point order differs")
                columns.append(sample["predictions"].copy())
                members.append(sample["fields"].copy())
        matrices.append(np.stack(columns, axis=1))
        identities.extend(ids)
        mean = np.mean(members, axis=0)
        for geo in manifest["geography"]:
            patch, index = geo["patch"], geo["batch_index"]
            # Every method uses identical patches, depths and wet support.
            if patch not in common_patches:
                continue
            old = np.load(spatial / f"patch_{patch}.npz")
            truth, valid = reference["glorys"][index], reference["valid"][index].astype(
                bool
            )
            if not np.allclose(old["glorys"][valid], truth[valid], atol=1e-5):
                raise ValueError("Archived spatial reference differs")
            fields = {
                "residual_ensemble": [member[index] for member in members],
                "residual_mean": [mean[index]],
                "glorys": [truth],
                "climatology": [reference["climatology"][index]],
            }
            for method in ("lstm", "cnn", "unet3d", "unet2d"):
                fields[method] = [old[method]]
            common = valid.copy()
            for values in fields.values():
                for value in values:
                    common &= np.isfinite(value)
            ref_spectra = []
            for layer in np.where(common, truth, np.nan):
                result = radial_wavenumber_spectrum(
                    layer,
                    pixel_size_x_km=geo["dx_km"],
                    pixel_size_y_km=geo["dy_km"],
                    wavelength_edges_km=edges,
                    require_complete=False,
                )
                ref_spectra.append(result[0] if result else np.full(32, np.nan))
            for method, values in fields.items():
                distances, spectra = [], []
                for value in values:
                    distances.append(lpips_layers(metric, value, truth, common, device))
                    layers = []
                    for layer in np.where(common, value, np.nan):
                        result = radial_wavenumber_spectrum(
                            layer,
                            pixel_size_x_km=geo["dx_km"],
                            pixel_size_y_km=geo["dy_km"],
                            wavelength_edges_km=edges,
                            require_complete=False,
                        )
                        layers.append(result[0] if result else np.full(32, np.nan))
                    spectra.append(layers)
                average = np.mean(spectra, axis=0)
                distance = np.mean(distances, axis=0)
                for depth_index, depth in enumerate(manifest["depths_m"]):
                    ref_power = np.asarray(ref_spectra)[depth_index]
                    power = average[depth_index]
                    error = spectral_error(power, ref_power)
                    spatial_rows.append(
                        {
                            "patch": patch,
                            "method": method,
                            "depth_m": depth,
                            "lpips": distance[depth_index],
                            "spectral_log10_rmse": error,
                            "wet_pixels": int(common[depth_index].sum()),
                        }
                    )
                    for wavelength, energy in zip(centers, power):
                        spectra_rows.append(
                            {
                                "patch": patch,
                                "method": method,
                                "depth_m": depth,
                                "wavelength_km": wavelength,
                                "wavenumber_cycles_km": 1 / wavelength,
                                "power": energy,
                            }
                        )
            print(f"Scored spatial patch {patch}", flush=True)
    index = pd.Index(identities)
    if index.has_duplicates or set(index) != set(points.point_id):
        raise ValueError("Ensemble must cover every frozen point exactly once")
    matrix = np.concatenate(matrices)[index.get_indexer(points.point_id)]
    predictions = {"residual_ensemble": matrix}
    # Compare full-context forecasts only; saved bundles also contain ablations
    # and GLORYS-background corrections that are excluded from this website.
    for method in ("lstm", "cnn", "unet3d", "unet2d"):
        if method in saved.members:
            predictions[method] = saved.members[method]
    for method in ("glorys", "climatology"):
        predictions[method] = points[method].to_numpy()[:, None]
    overall, depth, missions = score_profiles(points, predictions)
    spatial_frame = pd.DataFrame(spatial_rows)
    spectra_frame = pd.DataFrame(spectra_rows)
    spatial_summary = (
        spatial_frame.groupby(["method", "patch"])[["lpips", "spectral_log10_rmse"]]
        .mean()
        .groupby("method")
        .mean()
        .reset_index()
    )
    spatial_summary["patches"] = (
        spatial_frame.groupby("method")
        .patch.nunique()
        .reindex(spatial_summary.method)
        .to_numpy()
    )
    averaged = spectra_frame.groupby(
        ["method", "depth_m", "wavelength_km", "wavenumber_cycles_km"], as_index=False
    ).power.mean()
    tables = {
        "ensemble_scores": pd.DataFrame(overall),
        "ensemble_depth_scores": pd.DataFrame(depth),
        "ensemble_mission_scores": pd.DataFrame(missions),
        "spatial_scores": spatial_summary,
        "spatial_by_depth_patch": spatial_frame,
        "spectra": averaged,
        "spectra_by_patch": spectra_frame,
    }
    for filename, frame in tables.items():
        frame.to_csv(output / f"{filename}.csv", index=False)
    np.savez_compressed(
        ensemble / "point_members.npz",
        point_ids=points.point_id.to_numpy(str),
        members=matrix,
    )
    metadata = {
        "members": matrix.shape[1],
        "profiles": points.profile_id.nunique(),
        "points": len(points),
        "spatial_patches": len(common_patches),
        "total_patches": points.patch_index.nunique(),
        "depths_m": manifests[0]["depths_m"],
        "overall": overall,
        "depth": depth,
        "missions": missions,
        "spatial": spatial_summary.to_dict("records"),
        "spectra": averaged.to_dict("records"),
        "methodology": {
            "crps": "Empirical ensemble CRPS; fair CRPS additionally corrects finite iid sample bias. Average within each cast, then equally across casts. No calibration; all 102 Italian XBT casts.",
            "lpips": "Pretrained LPIPS 0.1 VGG16, each depth repeated across RGB; fixed [-2,35] Celsius clipped/scaled to [-1,1]. Common wet mask, identical zero fill, spatial distances averaged over wet pixels eroded 16 cells. Exploratory natural-image metric against packaged GLORYS, not XBT accuracy. Average members, depths, then patches.",
            "spectra": "Physical temperature fields, plane detrending, 2D Hann window, window-energy-normalized FFT power, 32 logarithmic wavelength bins 30–1000 km. Physical spacing at patch center; common wet mask with zero residual fill. Average member power before patch averaging. Spectral score is log10-power RMSE against GLORYS with a common 1e-12 °C² power floor, then equal depth and patch weights; lower is better.",
            "support": "CRPS and spatial comparisons: all 14 Italian patches / 102 casts. Identical patch/depth/mask support across spatial methods. No EN4 cohort changes.",
            "uncertainty": "90% intervals are sample 5th–95th percentiles. Two missions do not support robust significance claims; discovery-year evaluation.",
        },
        "provenance": manifests,
        "downloads": [f"{filename}.csv" for filename in tables],
    }
    _json(output / "metrics.json", metadata)
    print(
        pd.DataFrame(overall)[
            ["method", "crps", "fair_crps", "coverage90", "width90"]
        ].to_string(index=False),
        flush=True,
    )


def main():
    """Read explicit input and publication destinations."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cohort", "ensemble", "spatial", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    score(args.cohort, args.ensemble, args.spatial, args.output, args.device)


if __name__ == "__main__":
    main()

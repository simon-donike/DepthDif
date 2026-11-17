"""Pixel config loading and scenario resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Any

import yaml

from depth_recon.paths import config_path, resolve_config_path

PIXEL_SCENARIOS: dict[str, list[str]] = {
    "temperature": ["temperature"],
    "salinity": ["salinity"],
    "joint": ["temperature", "salinity"],
}

PIXEL_SCENARIO_EO: dict[str, tuple[str, str]] = {
    "temperature": ("ostia", "analysed_sst"),
    "salinity": ("sss", "sos"),
    "joint": ("ostia", "analysed_sst"),
}
PIXEL_SCENARIO_SURFACE_SOURCES: dict[str, list[str]] = {
    "temperature": ["sst"],
    "salinity": ["sss"],
    # Joint runs retain the existing temperature-oriented EO choice while using
    # only one surface channel like the single-variable scenarios.
    "joint": ["sst"],
}

DEFAULT_PIXEL_TRAINING_CONFIG_PATH = str(
    config_path("px_space", "training_super_config.yaml")
)
DEFAULT_PIXEL_INFERENCE_CONFIG_PATH = str(
    config_path("px_space", "inference_super_config.yaml")
)
FULL_RECONSTRUCTION_MONITOR = "val/full_reconstruction_score"


def apply_reconstruction_checkpoint_contract(training_cfg: dict[str, Any]) -> None:
    """Require reconstruction-based selection, including for saved older configs."""
    trainer_cfg = training_cfg.setdefault("trainer", {})
    monitor = trainer_cfg.get("ckpt_monitor", FULL_RECONSTRUCTION_MONITOR)
    if monitor not in {FULL_RECONSTRUCTION_MONITOR, "val/loss", "val/loss_ckpt"}:
        raise ValueError(
            f"Best checkpoints must monitor {FULL_RECONSTRUCTION_MONITOR}, got {monitor!r}."
        )
    # Migrate old loss-based presets before writing effective run configurations.
    trainer_cfg["ckpt_monitor"] = FULL_RECONSTRUCTION_MONITOR
    evaluation = training_cfg.setdefault("training", {}).setdefault(
        "reconstruction_eval", {}
    )
    for key, default in (("sample_count", 128), ("batch_size", 4), ("seed", 7)):
        evaluation.setdefault(key, default)
        value = evaluation[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < (0 if key == "seed" else 1)
        ):
            raise ValueError(
                f"training.reconstruction_eval.{key} has an invalid value: {value!r}."
            )


@dataclass(frozen=True)
class PixelTrainingConfigBundle:
    """Resolved pixel training configs and materialized split config paths."""

    config_path: str
    scenario: str
    data_cfg: dict[str, Any]
    model_cfg: dict[str, Any]
    training_cfg: dict[str, Any]
    effective_data_config_path: str
    effective_model_config_path: str
    effective_training_config_path: str
    uploaded_config_paths: list[str]


@dataclass(frozen=True)
class PixelInferenceConfigBundle:
    """Resolved pixel inference configs and materialized split config paths."""

    config_path: str
    scenario: str
    data_cfg: dict[str, Any]
    model_cfg: dict[str, Any]
    training_cfg: dict[str, Any]
    inference_cfg: dict[str, Any]
    effective_data_config_path: str
    effective_model_config_path: str
    effective_training_config_path: str
    effective_inference_config_path: str
    uploaded_config_paths: list[str]


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load YAML config data from a package-relative or filesystem path."""
    with resolve_config_path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def dump_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    """Write YAML config data while preserving dictionary order."""
    with Path(path).open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def parse_config_override(raw_override: str) -> tuple[str, list[str], Any]:
    """Parse a strict CLI override of the form root.path=value."""
    if "=" not in raw_override:
        raise ValueError(
            f"Invalid override '{raw_override}'. Expected format: "
            "<data|training|model>.<path>=<yaml_value>."
        )

    lhs, rhs = raw_override.split("=", 1)
    lhs = lhs.strip()
    rhs = rhs.strip()
    if "." not in lhs:
        raise ValueError(
            f"Invalid override '{raw_override}'. Missing nested key path after root."
        )

    root, *keys = [part.strip() for part in lhs.split(".")]
    if root not in {"data", "training", "model", "inference"}:
        raise ValueError(
            f"Invalid override root '{root}' in '{raw_override}'. "
            "Allowed roots: data, training, model, inference."
        )
    if any(not key for key in keys):
        raise ValueError(
            f"Invalid override '{raw_override}'. Key path contains empty segment(s)."
        )

    return root, keys, yaml.safe_load(rhs)


def apply_config_overrides(
    overrides: list[str],
    configs_by_root: dict[str, dict[str, Any]],
) -> None:
    """Apply strict nested config overrides in-place."""
    for raw_override in overrides:
        root, keys, value = parse_config_override(raw_override)
        target: dict[str, Any] = configs_by_root[root]
        for key in keys[:-1]:
            if key not in target:
                raise KeyError(
                    f"Invalid override '{raw_override}': key '{key}' does not exist."
                )
            nested = target[key]
            if not isinstance(nested, dict):
                raise TypeError(
                    f"Invalid override '{raw_override}': '{key}' is not a mapping."
                )
            target = nested

        leaf_key = keys[-1]
        if leaf_key not in target:
            raise KeyError(
                f"Invalid override '{raw_override}': key '{leaf_key}' does not exist."
            )
        target[leaf_key] = value


def override_key_set(overrides: list[str] | None) -> set[str]:
    """Return normalized override key paths grouped by top-level root."""
    keys: set[str] = set()
    for raw_override in list(overrides or []):
        root, path_parts, _value = parse_config_override(raw_override)
        keys.add(".".join([root, *path_parts]))
    return keys


def apply_unet_baseline_condition_contract(
    model_cfg: dict[str, Any], override_keys: set[str]
) -> None:
    """Derive U-Net baseline condition channels after scenario overrides."""
    model_section = model_cfg.get("model", {})
    if not isinstance(model_section, dict):
        return
    model_type = str(model_section.get("model_type", "cond_px_dif")).strip()
    if model_type not in {"unet_baseline", "unet2d_baseline"}:
        return

    generated_channels = int(model_section.get("generated_channels", 1))
    if generated_channels < 1:
        raise ValueError("model.generated_channels must be >= 1.")
    output_fields = model_section.get("output_fields", ["temperature"])
    if isinstance(output_fields, str):
        output_fields = [output_fields]
    field_channels = len(list(output_fields))
    if field_channels < 1:
        raise ValueError("model.output_fields must contain at least one field.")
    if generated_channels % field_channels != 0:
        raise ValueError(
            "model.generated_channels must be divisible by the active output "
            "field count for U-Net baselines."
        )

    unet_cfg = model_section.get("unet_baseline", {})
    if unet_cfg is None:
        unet_cfg = {}
    if not isinstance(unet_cfg, dict):
        raise ValueError("model.unet_baseline must be a mapping when provided.")

    use_valid_mask = bool(model_section.get("condition_use_valid_mask", True))
    per_channel_valid_mask = bool(unet_cfg.get("per_channel_valid_mask", True))
    if model_type == "unet2d_baseline":
        # The 2D comparison model flattens depth into channels. Per-channel masks
        # therefore need one channel per generated field/depth band.
        per_channel_count = generated_channels
        condition_channels = generated_channels
    else:
        # The 3D U-Net keeps depth as a convolution axis, so condition channels
        # count field streams rather than every depth band.
        per_channel_count = field_channels
        condition_channels = field_channels

    if use_valid_mask and per_channel_valid_mask:
        default_mask_channels = per_channel_count
    elif use_valid_mask:
        default_mask_channels = int(model_section.get("condition_mask_channels", 1))
    else:
        default_mask_channels = 0

    if "model.condition_mask_channels" not in override_keys:
        model_section["condition_mask_channels"] = default_mask_channels

    mask_channels = (
        int(model_section.get("condition_mask_channels", default_mask_channels))
        if use_valid_mask
        else 0
    )
    if bool(model_section.get("condition_include_eo", False)):
        condition_channels += 1
    condition_channels += mask_channels
    if bool(model_section.get("condition_use_land_mask", False)):
        condition_channels += 1
    if "model.condition_channels" not in override_keys:
        model_section["condition_channels"] = condition_channels


def apply_cnn_baseline_condition_contract(
    model_cfg: dict[str, Any], override_keys: set[str]
) -> None:
    """Derive profile-CNN baseline feature count after scenario overrides."""
    model_section = model_cfg.get("model", {})
    if not isinstance(model_section, dict):
        return
    model_type = str(model_section.get("model_type", "cond_px_dif")).strip()
    if model_type != "cnn_baseline":
        return

    generated_channels = int(model_section.get("generated_channels", 1))
    if generated_channels < 1:
        raise ValueError("model.generated_channels must be >= 1.")
    output_fields = model_section.get("output_fields", ["temperature"])
    if isinstance(output_fields, str):
        output_fields = [output_fields]
    field_channels = len(list(output_fields))
    if field_channels < 1:
        raise ValueError("model.output_fields must contain at least one field.")
    if generated_channels % field_channels != 0:
        raise ValueError(
            "model.generated_channels must be divisible by the active output "
            "field count for cnn_baseline."
        )

    cnn_cfg = model_section.get("cnn_baseline", {})
    if cnn_cfg is None:
        cnn_cfg = {}
    if not isinstance(cnn_cfg, dict):
        raise ValueError("model.cnn_baseline must be a mapping when provided.")

    depth_channels = generated_channels // field_channels

    def optional_bool(key: str, default: bool) -> bool:
        value = cnn_cfg.get(key, None)
        return bool(default) if value is None else bool(value)

    include_valid_mask = optional_bool(
        "include_valid_mask",
        bool(model_section.get("condition_use_valid_mask", True)),
    )
    include_eo = optional_bool(
        "include_eo",
        bool(model_section.get("condition_include_eo", True)),
    )
    include_land_mask = optional_bool(
        "include_land_mask",
        bool(model_section.get("condition_use_land_mask", True)),
    )

    mask_channels = depth_channels if include_valid_mask else 0
    if "model.condition_mask_channels" not in override_keys:
        model_section["condition_mask_channels"] = mask_channels

    # The profile CNN consumes one vector per pixel, so condition_channels records
    # vector width: sparse profile, optional depth-wise mask, and scalar surfaces.
    condition_channels = depth_channels + mask_channels
    if include_eo:
        condition_channels += 1
    if include_land_mask:
        condition_channels += 1
    if "model.condition_channels" not in override_keys:
        model_section["condition_channels"] = condition_channels


def apply_surface_conditioning_contract(
    model_cfg: dict[str, Any],
    data_cfg: dict[str, Any],
    override_keys: set[str],
) -> None:
    """Derive dense surface and total denoiser channel counts."""
    model_section = model_cfg.get("model", {})
    dataset_section = data_cfg.get("dataset", {})
    surface_cfg = dataset_section.get("surface_conditioning", {})
    sources = surface_cfg.get("sources") if isinstance(surface_cfg, dict) else None
    if sources is None:
        eo_channels = int(model_section.get("condition_eo_channels", 1))
    else:
        if isinstance(sources, str):
            sources = [sources]
        eo_channels = len(list(sources))
        if eo_channels < 1:
            raise ValueError("data.dataset.surface_conditioning.sources is empty.")
    model_section["condition_eo_channels"] = eo_channels

    prior_cfg = dataset_section.get("synthetic_target", {})
    if isinstance(prior_cfg, dict) and bool(prior_cfg.get("enabled", False)):
        if not prior_cfg.get("statistics_path"):
            raise ValueError("synthetic_target.enabled=true requires statistics_path.")
        if not bool(model_section.get("condition_include_eo", False)):
            raise ValueError(
                "Online prior pretraining requires model.condition_include_eo=true."
            )
        if not bool(model_section.get("mask_loss_with_valid_pixels", False)):
            raise ValueError(
                "Online prior pretraining requires "
                "model.mask_loss_with_valid_pixels=true."
            )
        ambient_cfg = model_section.get("ambient_occlusion", {})
        if isinstance(ambient_cfg, dict) and bool(ambient_cfg.get("enabled", False)):
            raise ValueError(
                "Online prior pretraining requires "
                "model.ambient_occlusion.enabled=false."
            )

    if "model.condition_channels" in override_keys:
        return
    condition_channels = int(model_section.get("generated_channels", 1))
    if bool(model_section.get("condition_include_eo", False)):
        condition_channels += eo_channels
    if bool(model_section.get("condition_use_valid_mask", True)):
        condition_channels += int(model_section.get("condition_mask_channels", 1))
    if bool(model_section.get("condition_use_land_mask", False)):
        condition_channels += 1
    model_section["condition_channels"] = condition_channels


def apply_depth_aware_condition_contract(
    model_cfg: dict[str, Any],
    data_cfg: dict[str, Any],
    override_keys: set[str],
) -> None:
    """Resolve depth-wise masks and climatology channels for pixel diffusion.

    These options intentionally apply only to ``cond_px_dif``.  The baseline
    models have separate channel contracts and must remain comparable when a
    diffusion configuration happens to contain these keys.
    """
    model_section = model_cfg.get("model", {})
    if not isinstance(model_section, dict):
        return
    model_type = str(model_section.get("model_type", "cond_px_dif")).strip()
    if model_type != "cond_px_dif":
        return

    # Missing flags preserve the effective contract of old configs/checkpoints.
    has_new_contract = any(
        key in model_section
        for key in (
            "condition_per_depth_valid_mask",
            "condition_use_wet_mask",
            "mask_diffusion_with_wet_mask",
            "climatology_residual",
        )
    )
    if not has_new_contract:
        return

    generated_channels = int(model_section.get("generated_channels", 1))
    if generated_channels < 1:
        raise ValueError("model.generated_channels must be >= 1.")
    dataset_section = data_cfg.get("dataset", {})
    if not isinstance(dataset_section, dict):
        dataset_section = {}
    static_mask_cfg = dataset_section.get("wet_domain", {})
    static_mask_enabled = (
        bool(static_mask_cfg)
        if isinstance(static_mask_cfg, bool)
        else isinstance(static_mask_cfg, dict)
        and bool(static_mask_cfg.get("enabled", False))
    )
    use_wet_mask = bool(model_section.get("condition_use_wet_mask", False))
    mask_diffusion = bool(model_section.get("mask_diffusion_with_wet_mask", False))
    if (use_wet_mask or mask_diffusion) and not static_mask_enabled:
        raise ValueError(
            "Depth-aware wet-mask conditioning/masking requires "
            "data.dataset.wet_domain.enabled=true."
        )
    if mask_diffusion and not bool(
        model_section.get("mask_loss_with_valid_pixels", False)
    ):
        raise ValueError(
            "model.mask_diffusion_with_wet_mask=true requires "
            "model.mask_loss_with_valid_pixels=true."
        )

    if bool(model_section.get("condition_per_depth_valid_mask", False)):
        if "model.condition_mask_channels" not in override_keys:
            model_section["condition_mask_channels"] = generated_channels

    climatology_cfg = dataset_section.get("climatology", {})
    climatology_enabled = (
        isinstance(climatology_cfg, dict)
        and bool(climatology_cfg.get("enabled", False))
        and bool(climatology_cfg.get("path"))
    )
    residual = bool(model_section.get("climatology_residual", False))
    if residual and not climatology_enabled:
        raise ValueError(
            "model.climatology_residual=true requires "
            "data.dataset.climatology.enabled=true and a non-empty path."
        )

    if "model.condition_channels" in override_keys:
        return
    condition_channels = generated_channels
    if bool(model_section.get("condition_include_eo", False)):
        condition_channels += int(model_section.get("condition_eo_channels", 1))
    if bool(model_section.get("condition_use_valid_mask", True)):
        condition_channels += int(model_section.get("condition_mask_channels", 1))
    if bool(model_section.get("condition_use_land_mask", False)):
        condition_channels += 1
    if use_wet_mask:
        condition_channels += generated_channels
    if residual:
        condition_channels += generated_channels
    model_section["condition_channels"] = condition_channels


def apply_latent_condition_contract(
    model_cfg: dict[str, Any], data_cfg: dict[str, Any]
) -> None:
    """Keep physical field channels separate from encoded diffusion channels."""
    model = model_cfg["model"]
    if model.get("model_type") != "latent_cond_dif":
        return
    latent = model.get("latent", {})
    physical = int(model.get("depth_channels", 50)) * len(model["output_fields"])
    generated = int(latent.get("latent_channels", 12))
    if generated < 1 or int(latent.get("spatial_downsample", 1)) != 1:
        raise ValueError(
            "Latent workflow requires positive channels and spatial_downsample=1."
        )
    if model.get("ambient_occlusion", {}).get("enabled", False) or model.get(
        "clamp_known_pixels", False
    ):
        raise ValueError(
            "Latent workflow does not support ambient corruption or observation clamping."
        )
    dataset = data_cfg.get("dataset", {})
    if dataset.get("synthetic_target", {}).get("enabled", False):
        raise ValueError("Latent workflow currently requires dense GLORYS targets.")
    if not model.get("condition_use_valid_mask", True) or not model.get(
        "mask_loss_with_valid_pixels", True
    ):
        raise ValueError(
            "Latent workflow requires observation conditioning and valid-pixel loss masks."
        )
    if not dataset.get("wet_domain", {}).get("enabled", False):
        raise ValueError(
            "Latent workflow requires data.dataset.wet_domain.enabled=true."
        )
    if model.get("climatology_residual", False):
        climatology = dataset.get("climatology", {})
        if not climatology.get("enabled", False) or not climatology.get("path"):
            raise ValueError("Latent residuals require a fitted climatology path.")
    model["physical_channels"] = physical
    model["generated_channels"] = generated
    model["condition_mask_channels"] = physical
    model["condition_per_depth_valid_mask"] = True
    model["condition_use_wet_mask"] = True
    model["mask_diffusion_with_wet_mask"] = True
    model["condition_channels"] = generated + physical + physical
    if model.get("condition_include_eo", False):
        model["condition_channels"] += model["condition_eo_channels"]
    if model.get("condition_use_land_mask", False):
        model["condition_channels"] += 1
    if model.get("climatology_residual", False):
        model["condition_channels"] += physical


def resolve_pixel_scenario(
    super_cfg: dict[str, Any], scenario_override: str | None = None
) -> str:
    """Resolve the pixel scenario from CLI or super-config."""
    raw_scenario = scenario_override
    if raw_scenario is None:
        raw_scenario = super_cfg.get("scenario", None)
    if raw_scenario is None and isinstance(super_cfg.get("model"), dict):
        raw_scenario = super_cfg["model"].get("scenario", None)
    if raw_scenario is None or raw_scenario is False:
        raw_scenario = "temperature"

    scenario = str(raw_scenario).strip().lower()
    if scenario not in PIXEL_SCENARIOS:
        supported = ", ".join(sorted(PIXEL_SCENARIOS))
        raise ValueError(
            f"Unsupported pixel scenario '{raw_scenario}'. "
            f"Supported scenarios: {supported}."
        )
    return scenario


def apply_pixel_scenario(
    *,
    model_cfg: dict[str, Any],
    data_cfg: dict[str, Any],
    scenario: str,
) -> None:
    """Apply scenario-derived data/model settings in-place."""
    model_section = model_cfg.setdefault("model", {})
    if not isinstance(model_section, dict):
        raise ValueError("model config root must contain a 'model' mapping.")
    dataset_section = data_cfg.setdefault("dataset", {})
    if not isinstance(dataset_section, dict):
        raise ValueError("data config root must contain a 'dataset' mapping.")
    output_section = dataset_section.setdefault("output", {})
    if not isinstance(output_section, dict):
        raise ValueError("data.dataset.output must be a mapping.")
    sampling_section = dataset_section.setdefault("sampling", {})
    if not isinstance(sampling_section, dict):
        raise ValueError("data.dataset.sampling must be a mapping.")
    surface_section = dataset_section.setdefault("surface_conditioning", {})
    if not isinstance(surface_section, dict):
        raise ValueError("data.dataset.surface_conditioning must be a mapping.")

    output_fields = PIXEL_SCENARIOS[scenario]
    eo_source, eo_var_name = PIXEL_SCENARIO_EO[scenario]
    depth_channels = int(model_section.get("depth_channels", 50))
    if depth_channels < 1:
        raise ValueError("model.depth_channels must be >= 1 when scenario is set.")

    generated_channels = depth_channels * len(output_fields)
    condition_channels = generated_channels
    if bool(model_section.get("condition_include_eo", False)):
        condition_channels += int(model_section.get("condition_eo_channels", 1))
    if bool(model_section.get("condition_use_valid_mask", True)):
        condition_channels += int(model_section.get("condition_mask_channels", 1))
    if bool(model_section.get("condition_use_land_mask", False)):
        condition_channels += 1

    # These fields are derived together so the data/model tensor contract cannot drift.
    model_section["scenario"] = scenario
    model_section["output_fields"] = list(output_fields)
    model_section["generated_channels"] = generated_channels
    model_section["condition_eo_channels"] = int(
        model_section.get("condition_eo_channels", 1)
    )
    model_section["condition_channels"] = condition_channels
    output_section["fields"] = list(output_fields)
    output_section["include_salinity"] = "salinity" in output_fields
    sampling_section["eo_source"] = eo_source
    sampling_section["eo_var_name"] = eo_var_name
    surface_section["sources"] = (
        ["sst", "sss", "adt"]
        if model_section.get("model_type") == "latent_cond_dif"
        else list(PIXEL_SCENARIO_SURFACE_SOURCES[scenario])
    )


def _require_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Pixel super-config requires a top-level '{key}' mapping.")
    return value


def _split_super_config(
    super_cfg: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    data_cfg = _require_mapping(super_cfg, "data")
    model_section = _require_mapping(super_cfg, "model")
    training_cfg = _require_mapping(super_cfg, "training")
    return data_cfg, {"model": model_section}, training_cfg


def _split_inference_super_config(
    super_cfg: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    data_cfg, model_cfg, training_cfg = _split_super_config(super_cfg)
    inference_cfg = _require_mapping(super_cfg, "inference")
    return data_cfg, model_cfg, training_cfg, {"inference": inference_cfg}


def _materialize_effective_configs(
    *,
    data_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    training_cfg: dict[str, Any],
    output_dir: str | Path,
) -> tuple[str, str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    data_path = output_path / "data_config_effective.yaml"
    model_path = output_path / "model_config_effective.yaml"
    training_path = output_path / "training_config_effective.yaml"
    dump_yaml(data_path, data_cfg)
    dump_yaml(model_path, model_cfg)
    dump_yaml(training_path, training_cfg)
    return str(data_path), str(model_path), str(training_path)


def _materialize_effective_inference_config(
    *,
    inference_cfg: dict[str, Any],
    output_dir: str | Path,
) -> str:
    """Write an effective inference config and return its path."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    inference_path = output_path / "inference_config_effective.yaml"
    dump_yaml(inference_path, inference_cfg)
    return str(inference_path)


def load_pixel_training_config(
    *,
    config_path_value: str | Path = DEFAULT_PIXEL_TRAINING_CONFIG_PATH,
    scenario_override: str | None = None,
    overrides: list[str] | None = None,
    runtime_config_dir: str | Path,
    snapshot_dir: str | Path | None = None,
    write_snapshots: bool = True,
) -> PixelTrainingConfigBundle:
    """Load one pixel super-config and materialize effective split configs."""
    resolved_config_path = resolve_config_path(config_path_value)
    super_cfg = load_yaml(resolved_config_path)
    data_cfg, model_cfg, training_cfg = _split_super_config(super_cfg)
    scenario = resolve_pixel_scenario(super_cfg, scenario_override=scenario_override)

    apply_reconstruction_checkpoint_contract(training_cfg)
    apply_pixel_scenario(model_cfg=model_cfg, data_cfg=data_cfg, scenario=scenario)
    override_keys = override_key_set(overrides)
    apply_config_overrides(
        list(overrides or []),
        configs_by_root={
            "data": data_cfg,
            "model": model_cfg["model"],
            "training": training_cfg,
        },
    )
    apply_surface_conditioning_contract(model_cfg, data_cfg, override_keys)
    apply_reconstruction_checkpoint_contract(training_cfg)
    apply_depth_aware_condition_contract(model_cfg, data_cfg, override_keys)
    apply_latent_condition_contract(model_cfg, data_cfg)
    apply_unet_baseline_condition_contract(model_cfg, override_keys)
    apply_cnn_baseline_condition_contract(model_cfg, override_keys)
    effective_data, effective_model, effective_training = (
        _materialize_effective_configs(
            data_cfg=data_cfg,
            model_cfg=model_cfg,
            training_cfg=training_cfg,
            output_dir=runtime_config_dir,
        )
    )

    uploaded_config_paths = [str(resolved_config_path)]
    if snapshot_dir is not None and write_snapshots:
        snapshot_path = Path(snapshot_dir)
        snapshot_path.mkdir(parents=True, exist_ok=True)
        super_snapshot = snapshot_path / Path(resolved_config_path).name
        shutil.copy2(resolved_config_path, super_snapshot)
        snapshot_data, snapshot_model, snapshot_training = (
            _materialize_effective_configs(
                data_cfg=data_cfg,
                model_cfg=model_cfg,
                training_cfg=training_cfg,
                output_dir=snapshot_path,
            )
        )
        uploaded_config_paths = [
            str(super_snapshot),
            snapshot_data,
            snapshot_model,
            snapshot_training,
        ]

    return PixelTrainingConfigBundle(
        config_path=str(resolved_config_path),
        scenario=scenario,
        data_cfg=data_cfg,
        model_cfg=model_cfg,
        training_cfg=training_cfg,
        effective_data_config_path=effective_data,
        effective_model_config_path=effective_model,
        effective_training_config_path=effective_training,
        uploaded_config_paths=uploaded_config_paths,
    )


def load_pixel_inference_config(
    *,
    config_path_value: str | Path = DEFAULT_PIXEL_INFERENCE_CONFIG_PATH,
    scenario_override: str | None = None,
    overrides: list[str] | None = None,
    runtime_config_dir: str | Path,
    snapshot_dir: str | Path | None = None,
    write_snapshots: bool = True,
) -> PixelInferenceConfigBundle:
    """Load one pixel inference super-config and materialize effective configs."""
    resolved_config_path = resolve_config_path(config_path_value)
    super_cfg = load_yaml(resolved_config_path)
    data_cfg, model_cfg, training_cfg, inference_cfg = _split_inference_super_config(
        super_cfg
    )
    scenario = resolve_pixel_scenario(super_cfg, scenario_override=scenario_override)

    apply_pixel_scenario(model_cfg=model_cfg, data_cfg=data_cfg, scenario=scenario)
    override_keys = override_key_set(overrides)
    apply_config_overrides(
        list(overrides or []),
        configs_by_root={
            "data": data_cfg,
            "model": model_cfg["model"],
            "training": training_cfg,
            "inference": inference_cfg["inference"],
        },
    )
    apply_surface_conditioning_contract(model_cfg, data_cfg, override_keys)
    apply_depth_aware_condition_contract(model_cfg, data_cfg, override_keys)
    apply_latent_condition_contract(model_cfg, data_cfg)
    apply_unet_baseline_condition_contract(model_cfg, override_keys)
    apply_cnn_baseline_condition_contract(model_cfg, override_keys)
    effective_data, effective_model, effective_training = (
        _materialize_effective_configs(
            data_cfg=data_cfg,
            model_cfg=model_cfg,
            training_cfg=training_cfg,
            output_dir=runtime_config_dir,
        )
    )
    effective_inference = _materialize_effective_inference_config(
        inference_cfg=inference_cfg,
        output_dir=runtime_config_dir,
    )

    uploaded_config_paths = [str(resolved_config_path)]
    if snapshot_dir is not None and write_snapshots:
        snapshot_path = Path(snapshot_dir)
        snapshot_path.mkdir(parents=True, exist_ok=True)
        super_snapshot = snapshot_path / Path(resolved_config_path).name
        shutil.copy2(resolved_config_path, super_snapshot)
        snapshot_data, snapshot_model, snapshot_training = (
            _materialize_effective_configs(
                data_cfg=data_cfg,
                model_cfg=model_cfg,
                training_cfg=training_cfg,
                output_dir=snapshot_path,
            )
        )
        snapshot_inference = _materialize_effective_inference_config(
            inference_cfg=inference_cfg,
            output_dir=snapshot_path,
        )
        uploaded_config_paths = [
            str(super_snapshot),
            snapshot_data,
            snapshot_model,
            snapshot_training,
            snapshot_inference,
        ]

    return PixelInferenceConfigBundle(
        config_path=str(resolved_config_path),
        scenario=scenario,
        data_cfg=data_cfg,
        model_cfg=model_cfg,
        training_cfg=training_cfg,
        inference_cfg=inference_cfg,
        effective_data_config_path=effective_data,
        effective_model_config_path=effective_model,
        effective_training_config_path=effective_training,
        effective_inference_config_path=effective_inference,
        uploaded_config_paths=uploaded_config_paths,
    )

"""Run one-off inference with a configured checkpoint.

This script loads the model and dataset configuration, restores a checkpoint,
and generates predictions either from the dataloader or from a random tensor
batch depending on the in-file mode settings.

Typical CLI:
    /work/envs/depth/bin/python -m depth_recon.inference.run_single
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from depth_recon.inference.core import (
    build_datamodule,
    build_dataset,
    build_model,
    build_random_batch,
    choose_device,
    ds_cfg_value,
    load_checkpoint_weights,
    pretty_shape,
    resolve_checkpoint_path,
    resolve_model_type,
    run_predict_once,
    to_device,
)
from depth_recon.configs.config_resolver_pixel import (
    DEFAULT_PIXEL_INFERENCE_CONFIG_PATH,
    load_pixel_inference_config,
)

# ----------------------------
# In-script settings
# ----------------------------
CONFIG_PATH = DEFAULT_PIXEL_INFERENCE_CONFIG_PATH
SCENARIO: str | None = None
CONFIG_OVERRIDES: list[str] = []

# Optional explicit checkpoint path. If None, uses model.resume_checkpoint from model config.
CHECKPOINT_PATH: str | None = None

# "dataloader" or "random"
MODE = "dataloader"  # or random

# Used when MODE == "dataloader": "train" or "val"
LOADER_SPLIT = "val"

# "auto", "cpu", or "cuda"
DEVICE = "auto"

SEED = 7
STRICT_LOAD = False
INCLUDE_INTERMEDIATES = False

# Used when MODE == "random"
RANDOM_BATCH_SIZE = 2
RANDOM_HEIGHT: int | None = None
RANDOM_WIDTH: int | None = None


def main() -> None:
    """Run the script entry point."""
    torch.manual_seed(int(SEED))

    config_bundle = load_pixel_inference_config(
        config_path_value=CONFIG_PATH,
        scenario_override=SCENARIO,
        overrides=CONFIG_OVERRIDES,
        runtime_config_dir=Path("/tmp/depthdif_inference_configs") / "run_single",
        write_snapshots=False,
    )
    model_cfg = config_bundle.model_cfg
    data_cfg = config_bundle.data_cfg
    training_cfg = config_bundle.training_cfg
    resolve_model_type(model_cfg)

    device = choose_device(DEVICE)

    if MODE not in {"dataloader", "random"}:
        raise ValueError(f"MODE must be 'dataloader' or 'random' (got '{MODE}').")
    if LOADER_SPLIT not in {"train", "val"}:
        raise ValueError(
            f"LOADER_SPLIT must be 'train' or 'val' (got '{LOADER_SPLIT}')."
        )

    dataset = build_dataset(
        config_bundle.effective_data_config_path, data_cfg.get("dataset", {})
    )
    datamodule = build_datamodule(
        dataset=dataset, data_cfg=data_cfg, training_cfg=training_cfg
    )
    datamodule.setup("fit")

    model = build_model(
        model_config_path=config_bundle.effective_model_config_path,
        data_config_path=config_bundle.effective_data_config_path,
        training_config_path=config_bundle.effective_training_config_path,
        model_cfg=model_cfg,
        datamodule=datamodule,
    )

    ckpt_path = resolve_checkpoint_path(CHECKPOINT_PATH, model_cfg)
    if ckpt_path is not None:
        weight_source = load_checkpoint_weights(
            model,
            ckpt_path,
            strict=bool(STRICT_LOAD),
        )
        print(f"Loaded checkpoint: {ckpt_path} ({weight_source} weights)")
    else:
        print("No checkpoint provided/found. Running with current model weights.")

    model = model.to(device)
    model.eval()

    if MODE == "dataloader":
        loader = (
            datamodule.train_dataloader()
            if LOADER_SPLIT == "train"
            else datamodule.val_dataloader()
        )
        batch = next(iter(loader))
        batch = to_device(batch, device)
    else:
        edge_size = int(
            ds_cfg_value(
                data_cfg.get("dataset", {}),
                "source.edge_size",
                "edge_size",
                default=128,
            )
        )
        h = int(RANDOM_HEIGHT) if RANDOM_HEIGHT is not None else edge_size
        w = int(RANDOM_WIDTH) if RANDOM_WIDTH is not None else edge_size
        batch = build_random_batch(
            model=model,
            data_cfg=data_cfg,
            batch_size=int(RANDOM_BATCH_SIZE),
            height=h,
            width=w,
            device=device,
        )

    print("Input batch keys/shapes:")
    for k, v in batch.items():
        print(f"  - {k}: {pretty_shape(v)}")

    pred = run_predict_once(
        model, batch, include_intermediates=bool(INCLUDE_INTERMEDIATES)
    )

    print("Output keys/shapes:")
    for k, v in pred.items():
        print(f"  - {k}: {pretty_shape(v)}")


if __name__ == "__main__":
    main()

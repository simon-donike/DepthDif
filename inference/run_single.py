"""Run one-off inference with a configured checkpoint.

This script loads the model and dataset configuration, restores a checkpoint,
and generates predictions either from the dataloader or from synthetic random
inputs depending on the in-file mode settings.

Typical CLI:
    /work/envs/depth/bin/python inference/run_single.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference.core import (
    build_datamodule,
    build_dataset,
    build_model,
    build_random_batch,
    choose_device,
    ds_cfg_value,
    load_yaml,
    pretty_shape,
    resolve_checkpoint_path,
    resolve_model_type,
    run_predict_once,
    to_device,
)

# ----------------------------
# In-script settings
# ----------------------------
MODEL_CONFIG_PATH = "configs/px_space/model_config.yaml"
DATA_CONFIG_PATH = "configs/px_space/data_ostia.yaml"
TRAIN_CONFIG_PATH = "configs/px_space/training_config.yaml"

# Optional explicit checkpoint path. If None, uses model.load_checkpoint then model.resume_checkpoint from model config.
CHECKPOINT_PATH: str | None = None

# "dataloader" or "random"
MODE = "dataloader" # or random

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

    model_cfg = load_yaml(MODEL_CONFIG_PATH)
    data_cfg = load_yaml(DATA_CONFIG_PATH)
    training_cfg = load_yaml(TRAIN_CONFIG_PATH)
    resolve_model_type(model_cfg)

    device = choose_device(DEVICE)

    if MODE not in {"dataloader", "random"}:
        raise ValueError(f"MODE must be 'dataloader' or 'random' (got '{MODE}').")
    if LOADER_SPLIT not in {"train", "val"}:
        raise ValueError(
            f"LOADER_SPLIT must be 'train' or 'val' (got '{LOADER_SPLIT}')."
        )

    dataset = build_dataset(DATA_CONFIG_PATH, data_cfg.get("dataset", {}))
    datamodule = build_datamodule(dataset=dataset, data_cfg=data_cfg, training_cfg=training_cfg)
    datamodule.setup("fit")

    model = build_model(
        model_config_path=MODEL_CONFIG_PATH,
        data_config_path=DATA_CONFIG_PATH,
        training_config_path=TRAIN_CONFIG_PATH,
        model_cfg=model_cfg,
        datamodule=datamodule,
    )

    ckpt_path = resolve_checkpoint_path(CHECKPOINT_PATH, model_cfg)
    if ckpt_path is not None:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=bool(STRICT_LOAD))
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print("No checkpoint provided/found. Running with current model weights.")

    model = model.to(device)
    model.eval()

    if MODE == "dataloader":
        loader = datamodule.train_dataloader() if LOADER_SPLIT == "train" else datamodule.val_dataloader()
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

    pred = run_predict_once(model, batch, include_intermediates=bool(INCLUDE_INTERMEDIATES))

    print("Output keys/shapes:")
    for k, v in pred.items():
        print(f"  - {k}: {pretty_shape(v)}")


if __name__ == "__main__":
    main()

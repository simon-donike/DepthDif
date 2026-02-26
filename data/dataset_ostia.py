from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from data.dataset_utils import SurfaceTempPatchBaseLightDataset


class SurfaceTempPatchOstiaLightDataset(SurfaceTempPatchBaseLightDataset):
    """Dataset that loads OSTIA-conditioned multi-band depth patches."""

    DEFAULT_CONFIG_PATH = "configs/data_ostia.yaml"
    FORCE_DISABLE_EO_DEGRADATION = True
    ENABLE_EO_DROPOUT = False
    PLOT_COLUMNS = ("x", "eo", "y")
    PLOT_TITLES = {
        "x": "Input X",
        "eo": "EO (OSTIA Surface Temp)",
        "y": "Y",
    }
    PLOT_OUTPUT_PATH = "temp/example_depth_tile_ostia.png"
    PLOT_FIG_WIDTH = 11.0

    def _load_eo_np(self, row: dict[str, Any], y_np_all: np.ndarray) -> np.ndarray:
        _ = y_np_all
        if "ostia_npy_path" not in row:
            raise RuntimeError("Index is missing required column 'ostia_npy_path'.")
        eo_abs_path = self._resolve_index_path(row["ostia_npy_path"])
        eo_np_all = np.load(eo_abs_path).astype(np.float32, copy=False)
        if eo_np_all.ndim == 2:
            return eo_np_all[None, ...]
        if eo_np_all.ndim == 3 and eo_np_all.shape[0] == 1:
            return eo_np_all
        raise RuntimeError(
            f"Unexpected EO shape at {eo_abs_path}: {tuple(eo_np_all.shape)} "
            "(expected (H,W) or (1,H,W))."
        )

    def _postprocess_eo_tensor(self, eo: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Always align OSTIA EO to the exact target grid before further processing.
        return F.interpolate(
            eo.unsqueeze(0),
            size=(int(y.shape[-2]), int(y.shape[-1])),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = super().__getitem__(idx)
        # Use the depth-derived land mask so OSTIA EO is zeroed at land locations.
        ocean_mask = (sample["land_mask"] > 0.5).any(dim=0, keepdim=True)
        sample["eo"] = sample["eo"] * ocean_mask.to(dtype=sample["eo"].dtype)
        return sample


if __name__ == "__main__":
    dataset = SurfaceTempPatchOstiaLightDataset.from_config("configs/data_ostia.yaml")
    dataset._plot_example_image()

    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    print(
        f"eo shape: {sample['eo'].shape}, x shape: {sample['x'].shape}, "
        f"y shape: {sample['y'].shape}"
    )
    print(
        "Valid mask sum: "
        f"{sample['valid_mask'].sum().item()}, "
        f"Land mask sum: {sample['land_mask'].sum().item()}"
    )
    print(f"Coords: {sample.get('coords', 'N/A')}")

    # testing images
    if True:
        import time

        for _ in range(5):
            dataset._plot_example_image()
            time.sleep(15)

    # looking at values
    if False:
        # count 0s in x: pixels are 0s after norm!
        zero_count = (sample["x"] == 0.0).sum().item()
        total_count = sample["x"].numel()
        print(
            f"Zero count in x: {zero_count} / {total_count} ({100 * zero_count / total_count:.2f}%)"
        )
        # count 0s in mask
        mask_zero_count = (sample["valid_mask"] == 0.0).sum().item()
        mask_total_count = sample["valid_mask"].numel()
        print(
            f"Zero count in valid_mask: {mask_zero_count} / {mask_total_count} ({100 * mask_zero_count / mask_total_count:.2f}%)"
        )

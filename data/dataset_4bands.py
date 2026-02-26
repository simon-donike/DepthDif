from __future__ import annotations

from data.dataset_utils import SurfaceTempPatchBaseLightDataset


class SurfaceTempPatch4BandsLightDataset(SurfaceTempPatchBaseLightDataset):
    """Dataset that loads EO-conditioned multi-band depth patches."""

    DEFAULT_CONFIG_PATH = "configs/data.yaml"
    FORCE_DISABLE_EO_DEGRADATION = False
    ENABLE_EO_DROPOUT = True
    PLOT_COLUMNS = ("eo", "x", "y", "valid_mask", "land_mask")
    PLOT_TITLES = {
        "eo": "EO (band 0)",
        "x": "Input x",
        "y": "Target y",
        "valid_mask": "Valid mask",
        "land_mask": "Land mask",
    }
    PLOT_OUTPUT_PATH = "temp/example_depth_tile_4bands.png"
    PLOT_FIG_WIDTH = 17.0


if __name__ == "__main__":
    dataset = SurfaceTempPatch4BandsLightDataset.from_config("configs/data.yaml")
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
    if False:
        import time

        for _ in range(5):
            dataset._plot_example_image()
            time.sleep(4)

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

__all__ = [
    "DEFAULT_ENRICHED_ARGO_ZARR",
    "DEFAULT_OUTPUT_DIR",
    "decode_stretched_uint8",
    "encode_stretched_uint8",
    "export_training_geotiff_dataset",
]


def __getattr__(name: str):
    """Lazily expose GeoTIFF export helpers without preloading the CLI module."""
    if name in __all__:
        from depth_recon.data.dataset_creation.export_dataset_geotiff import (
            export_dataset_geotiff,
        )

        return getattr(export_dataset_geotiff, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

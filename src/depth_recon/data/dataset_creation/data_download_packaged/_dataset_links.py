"""Helpers for packaged dataset download links."""

from __future__ import annotations

from pathlib import Path

import yaml

DATASET_LINKS_PATH = Path(__file__).with_name("dataset_links.yaml")


def load_dataset_url(dataset_key: str) -> str:
    """Load a packaged dataset URL from the bundled dataset links YAML."""
    # Keep the link file next to the downloader modules so installed packages
    # and source-tree runs resolve the same editable YAML file.
    with DATASET_LINKS_PATH.open("r", encoding="utf-8") as f:
        links = yaml.safe_load(f) or {}

    if not isinstance(links, dict):
        raise ValueError(
            f"Expected mapping in dataset links file: {DATASET_LINKS_PATH}"
        )

    url = links.get(dataset_key)
    if not isinstance(url, str) or not url.strip():
        raise ValueError(
            f"Missing dataset link '{dataset_key}' in {DATASET_LINKS_PATH}"
        )
    return url.strip()

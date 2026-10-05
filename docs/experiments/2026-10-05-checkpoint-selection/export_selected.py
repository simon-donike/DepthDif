# Example:
# /work/envs/depth/bin/python docs/experiments/2026-10-05-checkpoint-selection/export_selected.py \
#   --record-dir docs/experiments/2026-10-05-checkpoint-selection \
#   --output outputs/checkpoint_selection_20261005/selected_inference.ckpt
"""Freeze the winning diffusion weight variant without an ambiguous EMA fallback."""

import argparse
from pathlib import Path

import torch
import yaml

from depth_recon.inference.core import extract_ema_state_dict
from depth_recon.scripts.compare_checkpoints import file_sha256


def main():
    """Export inference-only weights, preserving original checkpoints separately."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = yaml.safe_load((args.record_dir / "results.yaml").read_text())
    selected = next(
        key.removesuffix("/field")
        for key in report["field_ranking_by_equal_depth_rmse"]
        if not key.startswith("climatology/")
    )
    identity = report["checkpoint_identity"][selected]
    if file_sha256(identity["checkpoint"]) != identity["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint no longer matches the evaluated copy.")
    original = torch.load(
        identity["checkpoint"], map_location="cpu", weights_only=False
    )
    weights = (
        extract_ema_state_dict(original)
        if identity["weights"] == "ema"
        else original["state_dict"]
    )
    if weights is None:
        raise ValueError("Selected weight variant is missing.")
    if args.output.exists():
        raise FileExistsError("Refusing to replace an existing selected export.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # No EMA callback or optimizer state: default inference loads precisely these
    # selected tensors, while resumable training still uses the archived original.
    torch.save(
        {
            "state_dict": weights,
            "epoch": original["epoch"],
            "global_step": original["global_step"],
            "variable_scenario": "temperature",
            "selection_provenance": identity,
        },
        args.output,
    )
    exported = torch.load(args.output, map_location="cpu", weights_only=False)
    if exported["state_dict"].keys() != weights.keys() or any(
        not torch.equal(tensor, exported["state_dict"][key])
        for key, tensor in weights.items()
    ):
        raise ValueError("Exported tensors differ from the selected weights.")
    if extract_ema_state_dict(exported) is not None:
        raise ValueError(
            "Inference export unexpectedly contains competing EMA weights."
        )
    args.output.chmod(0o444)
    record = {
        "selected_diffusion_method": selected,
        "selection_metric": "field equal_depth_rmse_c",
        "source": identity,
        "inference_export": str(args.output),
        "inference_export_sha256": file_sha256(args.output),
        "exact_tensor_equality_verified": True,
        "resume_training_from_export": False,
        "overall_field_winner": report["field_ranking_by_equal_depth_rmse"][0],
        "status": (
            "provisional_diffusion_reference; not promoted over climatology"
            if report["field_ranking_by_equal_depth_rmse"][0] == "climatology/field"
            else "selected_diffusion_reference_on_this_benchmark"
        ),
    }
    (args.record_dir / "selection.yaml").write_text(
        yaml.safe_dump(record, sort_keys=False)
    )
    print(yaml.safe_dump(record, sort_keys=False))


if __name__ == "__main__":
    main()

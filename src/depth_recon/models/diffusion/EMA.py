# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os.path
import warnings
from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import STEP_OUTPUT

try:
    import amp_C

    apex_available = True
except Exception:
    apex_available = False


class EMA(Callback):
    """Callback that maintains exponential moving-average model weights."""

    def __init__(
        self,
        decay: float,
        apply_ema_every_n_steps: int = 1,
        start_step: int = 0,
        save_ema_weights_in_callback_state: bool = False,
        evaluate_ema_weights_instead: bool = False,
    ) -> None:
        """Initialize EMA with configured parameters.

        Args:
            decay (float): Input value.
            apply_ema_every_n_steps (int): Step or timestep value.
            start_step (int): Step or timestep value.
            save_ema_weights_in_callback_state (bool): Boolean flag controlling behavior.
            evaluate_ema_weights_instead (bool): Boolean flag controlling behavior.

        Returns:
            None: No value is returned.
        """
        if not apex_available:
            rank_zero_warn(
                "EMA has better performance when Apex is installed: https://github.com/NVIDIA/apex#installation."
            )
        if not (0 <= decay <= 1):
            raise MisconfigurationException("EMA decay value must be between 0 and 1")
        if apply_ema_every_n_steps < 1:
            raise MisconfigurationException("EMA apply_ema_every_n_steps must be >= 1")
        if start_step < 0:
            raise MisconfigurationException("EMA start_step must be >= 0")
        self._ema_model_weights: Optional[Dict[str, torch.Tensor]] = None
        self._overflow_buf: Optional[torch.Tensor] = None
        self._cur_step: Optional[int] = None
        self._weights_buffer: Optional[Dict[str, torch.Tensor]] = None
        self.apply_ema_every_n_steps = apply_ema_every_n_steps
        self.start_step = start_step
        self.save_ema_weights_in_callback_state = save_ema_weights_in_callback_state
        self.evaluate_ema_weights_instead = evaluate_ema_weights_instead
        self.decay = decay

    def _setup_ema_weights(self, pl_module: "pl.LightningModule") -> None:
        """Initialize EMA state and move it to the module device."""
        model_state = pl_module.state_dict()
        if self._ema_model_weights is None:
            self._ema_model_weights = {
                key: value.detach().clone() for key, value in model_state.items()
            }

        missing_keys = [
            key for key in model_state.keys() if key not in self._ema_model_weights
        ]
        unexpected_keys = [
            key for key in self._ema_model_weights.keys() if key not in model_state
        ]
        if missing_keys or unexpected_keys:
            raise MisconfigurationException(
                "EMA state does not match model state_dict keys: "
                f"missing={missing_keys[:5]}, unexpected={unexpected_keys[:5]}."
            )

        # Keep EMA tensors on the active device so updates and eval swaps are cheap.
        self._ema_model_weights = {
            key: value.to(pl_module.device)
            for key, value in self._ema_model_weights.items()
        }
        self._overflow_buf = torch.IntTensor([0]).to(pl_module.device)

    def on_fit_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """Initialize EMA before sanity validation can run."""
        self._setup_ema_weights(pl_module)

    def on_train_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """Compute on train start and return the result.

        Args:
            trainer ('pl.Trainer'): Input value.
            pl_module ('pl.LightningModule'): Input value.

        Returns:
            None: No value is returned.
        """
        self._setup_ema_weights(pl_module)

    def ema(self, pl_module: "pl.LightningModule") -> None:
        """Compute ema and return the result.

        Args:
            pl_module ('pl.LightningModule'): Input value.

        Returns:
            None: No value is returned.
        """
        if self._ema_model_weights is None:
            self._setup_ema_weights(pl_module)
        if apex_available and pl_module.device.type == "cuda":
            return self.apply_multi_tensor_ema(pl_module)
        return self.apply_ema(pl_module)

    def apply_multi_tensor_ema(self, pl_module: "pl.LightningModule") -> None:
        """Compute apply multi tensor ema and return the result.

        Args:
            pl_module ('pl.LightningModule'): Input value.

        Returns:
            None: No value is returned.
        """
        ema_weights = []
        model_weights = []
        for key, orig_weight in pl_module.state_dict().items():
            ema_weight = self._ema_model_weights[key]
            if orig_weight.shape != ema_weight.shape:
                continue
            if not (
                torch.is_floating_point(orig_weight)
                and torch.is_floating_point(ema_weight)
            ):
                # Non-floating buffers cannot be averaged, but should stay current.
                ema_weight.copy_(orig_weight.detach().to(ema_weight.device))
                continue
            ema_weights.append(ema_weight)
            model_weights.append(
                orig_weight.detach().to(
                    device=ema_weight.device, dtype=ema_weight.dtype
                )
            )
        if not ema_weights:
            return None
        amp_C.multi_tensor_axpby(
            65536,  # todo (sean): chunk size, should we expose?
            self._overflow_buf,
            [ema_weights, model_weights, ema_weights],
            self.decay,
            1 - self.decay,
            -1,
        )

    def apply_ema(self, pl_module: "pl.LightningModule") -> None:
        """Compute apply ema and return the result.

        Args:
            pl_module ('pl.LightningModule'): Input value.

        Returns:
            None: No value is returned.
        """
        for key, orig_weight in pl_module.state_dict().items():
            ema_weight = self._ema_model_weights[key]
            if orig_weight.shape != ema_weight.shape:
                continue
            if not (
                torch.is_floating_point(orig_weight)
                and torch.is_floating_point(ema_weight)
            ):
                # Non-floating buffers cannot be averaged, but should stay current.
                ema_weight.copy_(orig_weight.detach().to(ema_weight.device))
                continue
            current = orig_weight.detach().to(
                device=ema_weight.device,
                dtype=ema_weight.dtype,
            )
            ema_weight.mul_(self.decay).add_(current, alpha=1.0 - self.decay)

    def should_apply_ema(self, step: int) -> bool:
        """Compute should apply ema and return the result.

        Args:
            step (int): Step or timestep value.

        Returns:
            bool: Computed scalar output.
        """
        return (
            step != self._cur_step
            and step >= self.start_step
            and step % self.apply_ema_every_n_steps == 0
        )

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Compute on train batch end and return the result.

        Args:
            trainer ('pl.Trainer'): Input value.
            pl_module ('pl.LightningModule'): Input value.
            outputs (STEP_OUTPUT): Input value.
            batch (Any): Input value.
            batch_idx (int): Zero-based index for selecting a sample or batch.

        Returns:
            None: No value is returned.
        """
        if self.should_apply_ema(trainer.global_step):
            self._cur_step = trainer.global_step
            self.ema(pl_module)

    def state_dict(self) -> Dict[str, Any]:
        """Return the serializable state for this object.

        Args:
            None: This callable takes no explicit input arguments.

        Returns:
            Dict[str, Any]: Computed output value.
        """
        if self.save_ema_weights_in_callback_state:
            ema_weights = None
            if self._ema_model_weights is not None:
                ema_weights = {
                    key: value.detach().cpu()
                    for key, value in self._ema_model_weights.items()
                }
            return dict(cur_step=self._cur_step, ema_weights=ema_weights)
        return dict(cur_step=self._cur_step)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load checkpoint weights into the current module.

        Args:
            state_dict (Dict[str, Any]): Input value.

        Returns:
            None: No value is returned.
        """
        self._cur_step = state_dict.get("cur_step")
        # when loading using NeMo, ema weights will be loaded by the experiment manager separately.
        if self._ema_model_weights is None:
            ema_weights = state_dict.get("ema_weights")
            if isinstance(ema_weights, dict):
                self._ema_model_weights = ema_weights

    def on_load_checkpoint(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        checkpoint: Dict[str, Any],
    ) -> None:
        """Compute on load checkpoint and return the result.

        Args:
            trainer ('pl.Trainer'): Input value.
            pl_module ('pl.LightningModule'): Input value.
            checkpoint (Dict[str, Any]): Input value.

        Returns:
            None: No value is returned.
        """
        checkpoint_callback = trainer.checkpoint_callback

        if (
            trainer.ckpt_path
            and checkpoint_callback is not None
            and "NeMo" in type(checkpoint_callback).__name__
        ):
            ext = checkpoint_callback.FILE_EXTENSION
            if trainer.ckpt_path.endswith(f"-EMA{ext}"):
                return
            ema_path = trainer.ckpt_path.replace(ext, f"-EMA{ext}")
            if os.path.exists(ema_path):
                ema_state_dict = torch.load(ema_path, map_location=torch.device("cpu"))
                self._ema_model_weights = dict(ema_state_dict["state_dict"])
                del ema_state_dict
            else:
                warnings.warn(
                    "we were unable to find the associated EMA weights when re-loading, "
                    "training will start with new EMA weights.",
                    UserWarning,
                )

    def replace_model_weights(self, pl_module: "pl.LightningModule") -> None:
        """Compute replace model weights and return the result.

        Args:
            pl_module ('pl.LightningModule'): Input value.

        Returns:
            None: No value is returned.
        """
        if self._weights_buffer is not None:
            raise MisconfigurationException("EMA weights are already applied.")
        if self._ema_model_weights is None:
            self._setup_ema_weights(pl_module)
        state_dict = pl_module.state_dict()
        self._weights_buffer = {
            key: value.detach().clone().to("cpu") for key, value in state_dict.items()
        }
        new_state_dict = {}
        for key, current_weight in state_dict.items():
            ema_weight = self._ema_model_weights[key]
            if current_weight.shape == ema_weight.shape:
                new_state_dict[key] = ema_weight.detach().to(
                    device=current_weight.device,
                    dtype=current_weight.dtype,
                )
            else:
                new_state_dict[key] = current_weight
        pl_module.load_state_dict(new_state_dict)

    def restore_original_weights(self, pl_module: "pl.LightningModule") -> None:
        """Compute restore original weights and return the result.

        Args:
            pl_module ('pl.LightningModule'): Input value.

        Returns:
            None: No value is returned.
        """
        if self._weights_buffer is None:
            raise MisconfigurationException("EMA weights have not been applied.")
        state_dict = pl_module.state_dict()
        new_state_dict = {
            key: value.to(device=state_dict[key].device, dtype=state_dict[key].dtype)
            for key, value in self._weights_buffer.items()
        }
        pl_module.load_state_dict(new_state_dict)
        self._weights_buffer = None

    @property
    def weights_are_applied(self) -> bool:
        """Return whether EMA weights are currently loaded into the module."""
        return self._weights_buffer is not None

    def _raw_state_dict_for_metrics(
        self, pl_module: "pl.LightningModule"
    ) -> Dict[str, torch.Tensor]:
        """Return raw model state even when EMA weights are temporarily applied."""
        if self._weights_buffer is not None:
            return self._weights_buffer
        return pl_module.state_dict()

    @torch.no_grad()
    def compute_weight_delta_metrics(
        self, pl_module: "pl.LightningModule"
    ) -> Dict[str, torch.Tensor]:
        """Compute raw-vs-EMA weight distance metrics."""
        if self._ema_model_weights is None:
            self._setup_ema_weights(pl_module)

        device = pl_module.device
        total_elements = torch.zeros((), device=device, dtype=torch.float64)
        total_abs_delta = torch.zeros((), device=device, dtype=torch.float64)
        total_delta_sq = torch.zeros((), device=device, dtype=torch.float64)
        total_raw_sq = torch.zeros((), device=device, dtype=torch.float64)
        max_abs_delta = torch.zeros((), device=device, dtype=torch.float64)
        floating_tensors = torch.zeros((), device=device, dtype=torch.float64)

        raw_state = self._raw_state_dict_for_metrics(pl_module)
        for key, raw_weight in raw_state.items():
            ema_weight = self._ema_model_weights.get(key)
            if ema_weight is None or raw_weight.shape != ema_weight.shape:
                continue
            if not (
                torch.is_floating_point(raw_weight)
                and torch.is_floating_point(ema_weight)
            ):
                continue

            raw = raw_weight.detach().to(device=device, dtype=torch.float64)
            ema = ema_weight.detach().to(device=device, dtype=torch.float64)
            delta = ema - raw
            if int(delta.numel()) == 0:
                continue
            abs_delta = delta.abs()
            total_elements += float(delta.numel())
            total_abs_delta += abs_delta.sum()
            total_delta_sq += delta.square().sum()
            total_raw_sq += raw.square().sum()
            max_abs_delta = torch.maximum(max_abs_delta, abs_delta.max())
            floating_tensors += 1.0

        safe_elements = torch.clamp(total_elements, min=1.0)
        rms_delta = torch.sqrt(total_delta_sq / safe_elements)
        raw_rms = torch.sqrt(total_raw_sq / safe_elements)
        return {
            "decay": torch.tensor(float(self.decay), device=device),
            "weight_mean_abs_delta": (total_abs_delta / safe_elements).float(),
            "weight_rms_delta": rms_delta.float(),
            "weight_relative_rms_delta": (
                rms_delta / torch.clamp(raw_rms, min=1.0e-12)
            ).float(),
            "weight_max_abs_delta": max_abs_delta.float(),
            "tracked_floating_tensors": floating_tensors.float(),
        }

    def log_weight_delta_metrics(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """Log EMA scalar diagnostics for the current validation epoch."""
        if self._ema_model_weights is None:
            return

        should_sync = False
        if hasattr(pl_module, "_should_sync_dist"):
            should_sync = bool(pl_module._should_sync_dist())
        elif trainer is not None:
            should_sync = int(getattr(trainer, "world_size", 1)) > 1

        metrics = self.compute_weight_delta_metrics(pl_module)
        for name, value in metrics.items():
            pl_module.log(
                f"ema/{name}",
                value,
                on_step=False,
                on_epoch=True,
                logger=True,
                sync_dist=should_sync,
                batch_size=1,
            )

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """Log EMA weight diagnostics once per validation epoch."""
        self.log_weight_delta_metrics(trainer, pl_module)

    @property
    def ema_initialized(self) -> bool:
        """Compute ema initialized and return the result.

        Args:
            None: This callable takes no explicit input arguments.

        Returns:
            bool: Computed scalar output.
        """
        return self._ema_model_weights is not None

    def on_validation_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """Compute on validation start and return the result.

        Args:
            trainer ('pl.Trainer'): Input value.
            pl_module ('pl.LightningModule'): Input value.

        Returns:
            None: No value is returned.
        """
        if self.ema_initialized and self.evaluate_ema_weights_instead:
            self.replace_model_weights(pl_module)

    def on_validation_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """Compute on validation end and return the result.

        Args:
            trainer ('pl.Trainer'): Input value.
            pl_module ('pl.LightningModule'): Input value.

        Returns:
            None: No value is returned.
        """
        if self.ema_initialized and self.evaluate_ema_weights_instead:
            self.restore_original_weights(pl_module)

    def on_test_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """Compute on test start and return the result.

        Args:
            trainer ('pl.Trainer'): Input value.
            pl_module ('pl.LightningModule'): Input value.

        Returns:
            None: No value is returned.
        """
        if self.ema_initialized and self.evaluate_ema_weights_instead:
            self.replace_model_weights(pl_module)

    def on_test_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """Compute on test end and return the result.

        Args:
            trainer ('pl.Trainer'): Input value.
            pl_module ('pl.LightningModule'): Input value.

        Returns:
            None: No value is returned.
        """
        if self.ema_initialized and self.evaluate_ema_weights_instead:
            self.restore_original_weights(pl_module)

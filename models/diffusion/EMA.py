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
from typing import Any, Dict, List, Optional

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
        self._ema_model_weights: Optional[List[torch.Tensor]] = None
        self._overflow_buf: Optional[torch.Tensor] = None
        self._cur_step: Optional[int] = None
        self._weights_buffer: Optional[List[torch.Tensor]] = None
        self.apply_ema_every_n_steps = apply_ema_every_n_steps
        self.start_step = start_step
        self.save_ema_weights_in_callback_state = save_ema_weights_in_callback_state
        self.evaluate_ema_weights_instead = evaluate_ema_weights_instead
        self.decay = decay

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
        if self._ema_model_weights is None:
            self._ema_model_weights = [
                p.detach().clone() for p in pl_module.state_dict().values()
            ]
        # ensure that all the weights are on the correct device
        self._ema_model_weights = [
            p.to(pl_module.device) for p in self._ema_model_weights
        ]
        self._overflow_buf = torch.IntTensor([0]).to(pl_module.device)

    def ema(self, pl_module: "pl.LightningModule") -> None:
        """Compute ema and return the result.

        Args:
            pl_module ('pl.LightningModule'): Input value.

        Returns:
            None: No value is returned.
        """
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
        model_weights = list(pl_module.state_dict().values())
        amp_C.multi_tensor_axpby(
            65536,  # todo (sean): chunk size, should we expose?
            self._overflow_buf,
            [self._ema_model_weights, model_weights, self._ema_model_weights],
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
        for orig_weight, ema_weight in zip(
            list(pl_module.state_dict().values()), self._ema_model_weights
        ):
            if orig_weight.data.shape == ema_weight.data:
                # (only if same shape, ignores gammas for diffusion models)
                diff = ema_weight.data - orig_weight.data
                diff.mul_(1.0 - self.decay)
                ema_weight.sub_(diff)

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
            return dict(cur_step=self._cur_step, ema_weights=self._ema_model_weights)
        return dict(cur_step=self._cur_step)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load checkpoint weights into the current module.

        Args:
            state_dict (Dict[str, Any]): Input value.

        Returns:
            None: No value is returned.
        """
        self._cur_step = state_dict["cur_step"]
        # when loading using NeMo, ema weights will be loaded by the experiment manager separately.
        if self._ema_model_weights is None:
            self._ema_model_weights = state_dict.get("ema_weights")

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
                self._ema_model_weights = ema_state_dict["state_dict"].values()
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
        self._weights_buffer = [
            p.detach().clone().to("cpu") for p in pl_module.state_dict().values()
        ]
        new_state_dict = {
            k: v for k, v in zip(pl_module.state_dict().keys(), self._ema_model_weights)
        }
        pl_module.load_state_dict(new_state_dict)

    def restore_original_weights(self, pl_module: "pl.LightningModule") -> None:
        """Compute restore original weights and return the result.

        Args:
            pl_module ('pl.LightningModule'): Input value.

        Returns:
            None: No value is returned.
        """
        state_dict = pl_module.state_dict()
        new_state_dict = {k: v for k, v in zip(state_dict.keys(), self._weights_buffer)}
        pl_module.load_state_dict(new_state_dict)
        del self._weights_buffer

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

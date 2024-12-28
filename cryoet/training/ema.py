import copy
import collections
from typing import Any

import numpy as np
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn
from lightning import Callback, Trainer, LightningModule
from torch.optim import Optimizer


# -------------------------------------------------------------------------
# EMA Decay Functions
# -------------------------------------------------------------------------
class EMADecay:
    """Abstract interface for decay schedule."""

    def __call__(self, step: int, total_steps: int):
        raise NotImplementedError


class ConstantDecay(EMADecay):
    def __init__(self, decay: float):
        self.decay = decay

    def __call__(self, step: int, total_steps: int):
        return self.decay

    def __repr__(self):
        return f"ConstantDecay(decay={self.decay})"


class ThresholdDecay(EMADecay):
    def __init__(self, decay: float):
        self.decay = decay

    def __call__(self, step: int, total_steps: int):
        # Example: step-based decay that saturates to self.decay
        dynamic_decay = (step + 1) / (step + 1000)
        return float(np.minimum(self.decay, dynamic_decay))

    def __repr__(self):
        return f"ThresholdDecay(decay={self.decay})"


class ExpEMADecay(EMADecay):
    def __init__(self, decay: float, beta: float):
        self.decay = decay
        self.beta = beta

    def __call__(self, step: int, total_steps: int):
        # Example: exponential schedule
        p = step / total_steps
        return self.decay * (1 - np.exp(-p * self.beta))

    def __repr__(self):
        return f"ExpEMADecay(decay={self.decay}, beta={self.beta})"


class BetaDecay(EMADecay):
    def __init__(self, beta: float, max_decay: float = 0.999):
        self.beta = beta
        self.max_decay = max_decay

    def __call__(self, step: int, total_steps: int):
        # Example: saturating exponential with param `beta`
        p = step / total_steps
        decay = 1 - (np.exp(-p) ** self.beta)
        return min(decay, self.max_decay)

    def __repr__(self):
        return f"BetaDecay(beta={self.beta}, max_decay={self.max_decay})"


# -------------------------------------------------------------------------
# ExponentialMovingAverage container
# -------------------------------------------------------------------------
class ExponentialMovingAverage:
    """
    Maintains (exponential) moving average of a model’s parameters.
    """

    def __init__(self, state_dict: collections.OrderedDict):
        """
        Args:
            state_dict: Copy of model.state_dict() at initialization.
        """
        # Store a copy of the model’s state dict
        self.state_dict = state_dict

    @torch.no_grad()
    def update(self, current_state_dict: collections.OrderedDict, decay: float):
        """
        Update the stored EMA parameters in-place.

        Args:
            current_state_dict: The current model’s parameters (model.state_dict()).
            decay: The fraction for EMA. If `decay` = 0.99, then
                   new_ema = old_ema * 0.99 + current_params * 0.01
        """
        if self.state_dict.keys() != current_state_dict.keys():
            raise RuntimeError("Keys in EMA state and current model do not match!")
        for key in self.state_dict:
            if self.state_dict[key].dtype.is_floating_point:
                # EMA update
                self.state_dict[key].mul_(decay).add_(current_state_dict[key], alpha=(1.0 - decay))
            else:
                # For non-float params (e.g., buffers like int), just copy
                self.state_dict[key].copy_(current_state_dict[key])


# -------------------------------------------------------------------------
# EMA Callback for PyTorch Lightning
# -------------------------------------------------------------------------
class EMACallback(Callback):
    """
    Weight Averaging Callback for Exponential Moving Averages in PyTorch Lightning.

    This callback:
      - Stores an internal copy of model weights (EMA).
      - After each optimizer step, it updates the EMA copy.
      - Temporarily switches model weights to EMA for validation.
      - (Optionally) reverts to the non-EMA weights for continued training.

    Usage:
      trainer = Trainer(callbacks=[EMACallback(decay=ConstantDecay(0.999))])
    """

    def __init__(
        self,
        decay: EMADecay,
    ):
        super().__init__()
        self.decay_fn = decay
        self.ema_storage = None
        self.non_ema_state_dict = None
        self.total_steps = None

    def __repr__(self):
        return f"EMACallback(decay={self.decay_fn})"

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
        Called at the very start of fit (training).
        We create a copy of the model’s weights for our EMA container
        and determine the total number of gradient steps for scheduling.
        """
        # 1) Initialize the total number of gradient steps for scheduling the decay
        #    In Lightning, `trainer.estimated_stepping_batches` is the total # of optimizer steps
        #    across all epochs. Alternatively, you could do:
        #    total_steps = trainer.num_training_batches * trainer.max_epochs
        #    but estimated_stepping_batches handles grad_accum as well.
        self.total_steps = trainer.estimated_stepping_batches

        # 2) Copy the model’s current state into the EMA container
        model_state_dict = copy.deepcopy(pl_module.state_dict())
        self.ema_storage = ExponentialMovingAverage(model_state_dict)

        # 3) We have not yet cached the “normal” (non-EMA) state
        self.non_ema_state_dict = None

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
        Called at the start of each training epoch.
        If we previously loaded EMA weights (for validation),
        we restore the model’s original weights here to continue training normally.
        """
        if self.non_ema_state_dict is not None:
            pl_module.load_state_dict(self.non_ema_state_dict)
            self.non_ema_state_dict = None

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs, batch: Any, batch_idx: int) -> None:
        """
        Called after each optimizer step (i.e., gradient update).
        We compute the “decay” fraction and update the EMA.
        """
        # 1) Determine the current step (# of completed optimizer steps).
        step = trainer.global_step  # In Lightning, global_step is # of optimizer steps by default
        if step > self.total_steps:
            # Edge case: if you exceed estimated stepping batches
            step = self.total_steps

        # 2) Compute the decay fraction from our decay schedule
        decay = self.decay_fn(step=step, total_steps=self.total_steps)

        # 3) Update the EMA with the current model’s parameters
        self.ema_storage.update(pl_module.state_dict(), decay)

        # (Optional) log or track the decay
        pl_module.log("train/ema", decay, on_step=True, on_epoch=False)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule, outputs) -> None:
        """
        Called after each training epoch ends.
        Save the normal model weights into a private storage so we can safely swap to EMA
        for validation without losing the "normal" weights.
        """
        self.non_ema_state_dict = copy.deepcopy(pl_module.state_dict())

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
        Called at the start of validation.
        We load the EMA weights into the model so validation metrics reflect the EMA version.
        """
        if self.ema_storage is not None:
            pl_module.load_state_dict(self.ema_storage.state_dict)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
        Called at the end of validation.
        We do *not* revert to the non-EMA weights here so that any checkpoints saved at epoch_end
        will contain the EMA weights.

        If you want your training to resume from normal weights, you can revert here.
        But typically we let training resume from normal weights at on_train_epoch_start.
        """
        pass

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
        Called at the very end of fit.
        Optionally load the EMA weights so the final model is EMA.
        Or revert to the non-EMA model if that’s desired.

        We’ll default to leaving the model in the EMA state.
        """
        if self.ema_storage is not None:
            pl_module.load_state_dict(self.ema_storage.state_dict)

        self.ema_storage = None
        self.non_ema_state_dict = None


# -------------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------------
# class MyModel(LightningModule):
#     ...
#
# model = MyModel()
# decay = ConstantDecay(decay=0.99)
# ema_callback = EMACallback(decay=decay)
#
# trainer = Trainer(
#     max_epochs=10,
#     callbacks=[ema_callback],
#     ...
# )
# trainer.fit(model)

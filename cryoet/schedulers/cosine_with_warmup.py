import math
import warnings
from typing import List

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineScheduler(_LRScheduler):
    """
    PyTorch-compatible LR scheduler with warmup and cosine decay.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_steps (int): Number of warmup steps.
        total_steps (int): Total steps for the entire schedule (warmup + decay).
        warmup_learning_rate (float): Initial learning rate for warmup.
            This will linearly increase to each param group’s base_lr over `warmup_steps`.
        decay_factor (float): Final factor for learning rate at the end of decay.
            i.e., final_lr = base_lr * decay_factor.
        last_epoch (int, optional): The index of the last epoch. Default: -1.
            By PyTorch convention, set to -1 on first call.
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        warmup_learning_rate: float,
        decay_factor: float = 0.01,
        last_epoch: int = -1,
    ):

        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.warmup_learning_rate = warmup_learning_rate
        self.decay_factor = decay_factor

        # Validate
        if not 0 <= self.warmup_steps <= self.total_steps:
            raise ValueError(
                f"warmup_steps ({warmup_steps}) must be in [0, total_steps ({total_steps})]."
            )
        if not 0.0 <= self.decay_factor <= 1.0:
            raise ValueError(
                f"decay_factor ({decay_factor}) should be between [0.0, 1.0]."
            )

        # Store each param group’s base LR for reference
        # This is what we will decay from
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        Compute the learning rate for each param group at step = self.last_epoch + 1.
        self.last_epoch is incremented by PyTorch at each scheduler step call.
        """
        step = self.last_epoch  # 0-based step index

        # For each param group, compute the updated LR
        lrs = []
        for base_lr in self.base_lrs:
            if step < self.warmup_steps:
                # ---- Warmup phase ----
                # LR linearly increases from `warmup_learning_rate` to `base_lr`
                progress = step / max(1.0, self.warmup_steps)  # avoid div-by-zero
                lr = self.warmup_learning_rate + progress * (
                    base_lr - self.warmup_learning_rate
                )
            else:
                # ---- Cosine decay phase ----
                # We consider steps beyond warmup_steps
                # progress in [0.0, 1.0] for the decay phase
                progress = (step - self.warmup_steps) / max(
                    1, self.total_steps - self.warmup_steps
                )

                # Cosine decay from base_lr down to base_lr * decay_factor
                # Formula: lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * progress))
                max_lr = base_lr
                min_lr = base_lr * self.decay_factor
                lr = min_lr + 0.5 * (max_lr - min_lr) * (
                    1 + math.cos(math.pi * progress)
                )

            lrs.append(lr)
        return lrs

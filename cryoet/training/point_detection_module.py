from typing import Optional, Any

import lightning as L
import torch
from lightning.pytorch.utilities.types import LRSchedulerTypeUnion
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from pytorch_toolbelt.optimization.functional import build_optimizer_param_groups
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

from .args import MyTrainingArguments
from cryoet.schedulers import WarmupCosineScheduler


class PointDetectionModel(L.LightningModule):
    def __init__(
        self,
        model,
        train_args: MyTrainingArguments,
    ):
        super().__init__()
        self.model = model
        self.train_args = train_args

    def forward(self, volume, labels=None, **loss_kwargs):
        return self.model(
            volume=volume,
            labels=labels,
            **loss_kwargs,
        )

    def training_step(self, batch, batch_idx):
        num_items_in_batch = batch["labels"].eq(1).sum().item()
        outputs = self(**batch, num_items_in_batch=num_items_in_batch)
        loss = outputs.loss

        self.log(
            "train/loss",
            loss,
            batch_size=len(batch["volume"]),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "train/num_items_in_batch",
            num_items_in_batch,
            batch_size=len(batch["volume"]),
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=False,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        num_items_in_batch = batch["labels"].eq(1).sum().item()
        outputs = self(**batch, num_items_in_batch=num_items_in_batch)

        val_loss = outputs.loss
        self.log(
            "val/loss",
            val_loss,
            batch_size=len(batch["volume"]),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        self.log(
            "val/num_items_in_batch",
            num_items_in_batch,
            batch_size=len(batch["volume"]),
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=False,
        )
        return val_loss

    def _get_tb_logger(self, trainer) -> Optional[SummaryWriter]:
        tb_logger: Optional[TensorBoardLogger] = None
        for logger in trainer.loggers:
            if isinstance(logger, TensorBoardLogger):
                tb_logger = logger
                break

        if tb_logger is not None:
            return tb_logger.experiment
        return None

    def configure_optimizers(self):
        param_groups, optimizer_kwargs = build_optimizer_param_groups(
            model=self,
            learning_rate=self.train_args.learning_rate,
            weight_decay=self.train_args.weight_decay,
            apply_weight_decay_on_bias=False,
            apply_weight_decay_on_norm=False,
        )

        if self.train_args.optim == "adamw_8bit":
            import bitsandbytes as bnb

            optimizer = bnb.optim.AdamW8bit(param_groups, **optimizer_kwargs)
        elif self.train_args.optim == "paged_adamw_8bit":
            import bitsandbytes as bnb

            optimizer = bnb.optim.AdamW8bit(
                param_groups,
                betas=(self.train_args.adam_beta1, self.train_args.adam_beta2),
                **optimizer_kwargs,
                is_paged=True,
            )
        elif self.train_args.optim == "adamw_torch_fused":
            optimizer = torch.optim.AdamW(
                param_groups,
                betas=(self.train_args.adam_beta1, self.train_args.adam_beta2),
                **optimizer_kwargs,
                fused=True,
            )
        elif self.train_args.optim == "adamw_torch":
            optimizer = torch.optim.AdamW(
                param_groups,
                betas=(self.train_args.adam_beta1, self.train_args.adam_beta2),
                **optimizer_kwargs,
            )
        else:
            raise KeyError(f"Unknown optimizer {self.train_args.optim}")

        warmup_steps = 0
        if self.train_args.warmup_steps > 0:
            warmup_steps = self.train_args.warmup_steps
            self.trainer.print(f"Using warmup steps: {warmup_steps}")
        elif self.train_args.warmup_ratio > 0:
            warmup_steps = int(
                self.train_args.warmup_ratio * self.trainer.estimated_stepping_batches
            )
            self.trainer.print(f"Using warmup steps: {warmup_steps}")

        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            warmup_learning_rate=1e-7,
            total_steps=self.trainer.estimated_stepping_batches,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "name": "cosine_with_warmup",
            },
        }

    def lr_scheduler_step(
        self, scheduler: LRSchedulerTypeUnion, metric: Optional[Any]
    ) -> None:
        scheduler.step()

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        # Log gradient norm (mean and max)
        if self.global_step % self.train_args.logging_steps == 0:
            with torch.no_grad():
                all_grads = torch.stack(
                    [
                        torch.norm(p.grad)
                        for p in self.model.parameters()
                        if p.requires_grad and p.grad is not None
                    ]
                ).view(-1)
                max_grads = torch.max(all_grads).item()
                mean_grads = all_grads.mean()
                self.log(
                    "train/mean_grad",
                    mean_grads,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=True,
                    logger=True,
                    sync_dist=False,
                    rank_zero_only=True,
                )
                self.log(
                    "train/max_grad",
                    max_grads,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=True,
                    logger=True,
                    sync_dist=False,
                    rank_zero_only=True,
                )

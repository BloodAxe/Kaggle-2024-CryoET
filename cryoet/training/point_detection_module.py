import dataclasses
from collections import defaultdict
from typing import Optional, Any

import lightning as L
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.utilities.types import LRSchedulerTypeUnion
from pytorch_toolbelt.utils import all_gather
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from pytorch_toolbelt.optimization.functional import build_optimizer_param_groups
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

from .args import MyTrainingArguments
from cryoet.schedulers import WarmupCosineScheduler
from .visualization import render_heatmap
from ..data.parsers import CLASS_LABEL_TO_CLASS_NAME, ANGSTROMS_IN_PIXEL
from ..data.point_detection_dataset import decoder_centers_from_heatmap
from ..metric import score_submission

from pytorch_toolbelt.utils.distributed import is_dist_avail_and_initialized
import torch.distributed as dist


def maybe_all_reduce(x: Tensor, op=dist.ReduceOp.SUM):
    if not is_dist_avail_and_initialized():
        return x

    xc = x.clone()
    dist.all_reduce(xc, op=op)
    return xc


@dataclasses.dataclass
class AccumulatedPredictionContainer:
    probas: Tensor
    counter: Tensor

    @classmethod
    def from_shape(cls, shape, num_classes, device="cpu"):
        shape = tuple(shape)
        return cls(probas=torch.zeros((num_classes,) + shape, device=device), counter=torch.zeros(shape, device=device))

    def accumulate(self, tile_scores, tile_offsets_zyx):
        probas_view = self.probas[
            :,
            tile_offsets_zyx[0] : tile_offsets_zyx[0] + tile_scores.shape[1],
            tile_offsets_zyx[1] : tile_offsets_zyx[1] + tile_scores.shape[2],
            tile_offsets_zyx[2] : tile_offsets_zyx[2] + tile_scores.shape[3],
        ]
        counter_view = self.counter[
            tile_offsets_zyx[0] : tile_offsets_zyx[0] + tile_scores.shape[1],
            tile_offsets_zyx[1] : tile_offsets_zyx[1] + tile_scores.shape[2],
            tile_offsets_zyx[2] : tile_offsets_zyx[2] + tile_scores.shape[3],
        ]

        # Crop tile_scores to the view shape
        tile_scores = tile_scores[:, : probas_view.shape[1], : probas_view.shape[2], : probas_view.shape[3]]

        probas_view += tile_scores
        counter_view += 1


class PointDetectionModel(L.LightningModule):
    def __init__(
        self,
        model,
        train_args: MyTrainingArguments,
    ):
        super().__init__()
        self.model = model
        self.train_args = train_args
        self.validation_predictions = None
        self.gather_num_items_in_batch = train_args.average_tokens_across_devices

    def forward(self, volume, labels=None, **loss_kwargs):
        return self.model(
            volume=volume,
            labels=labels,
            **loss_kwargs,
        )

    def training_step(self, batch, batch_idx):
        num_items_in_batch = batch["labels"].eq(1).sum()
        if self.gather_num_items_in_batch:
            num_items_in_batch = maybe_all_reduce(num_items_in_batch)

        outputs = self(**batch, num_items_in_batch=num_items_in_batch)

        if self.gather_num_items_in_batch:
            outputs.loss.mul_(self.trainer.world_size)

        loss = outputs.loss

        self.log(
            "train/loss",
            loss,
            batch_size=len(batch["volume"]),
            sync_dist=True,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            name="train/num_items_in_batch",
            value=int(num_items_in_batch),
            batch_size=len(batch["volume"]),
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=False,
        )
        return loss

    def on_validation_start(self) -> None:
        self.validation_predictions = {}

    def accumulate_predictions(self, outputs, batch):
        probas = outputs.logits.sigmoid().cpu()
        num_classes = probas.shape[1]

        for study, tile_offsets_zyx, volume_shape, tile_scores in zip(
            batch["study"], batch["tile_offsets_zyx"], batch["volume_shape"], probas
        ):
            if study not in self.validation_predictions:
                self.validation_predictions[study] = AccumulatedPredictionContainer(
                    probas=torch.zeros((num_classes,) + tuple(volume_shape), dtype=tile_scores.dtype),
                    counter=torch.zeros(tuple(volume_shape), dtype=tile_scores.dtype),
                )

            self.validation_predictions[study].accumulate(tile_scores, tile_offsets_zyx)

    def on_validation_epoch_end(self) -> None:
        torch.cuda.empty_cache()
        all_validation_predictions = all_gather(self.validation_predictions)
        self.validation_predictions = None

        submission = defaultdict(list)

        averaged_predictions = {}
        for validation_predictions in all_validation_predictions:
            for (
                study_name,
                accumulated_predictions,
            ) in validation_predictions.items():
                if study_name not in averaged_predictions:
                    averaged_predictions[study_name] = AccumulatedPredictionContainer(
                        probas=torch.zeros_like(accumulated_predictions.probas),
                        counter=torch.zeros_like(accumulated_predictions.counter),
                    )

                averaged_predictions[study_name].probas += accumulated_predictions.probas
                averaged_predictions[study_name].counter += accumulated_predictions.counter

        for study_name, accumulated_predictions in averaged_predictions.items():
            accumulated_predictions.probas /= accumulated_predictions.counter
            accumulated_predictions.probas.masked_fill_(accumulated_predictions.counter == 0, 0.0)

            self.log_heatmaps(study_name, accumulated_predictions.probas)

            topk_scores, topk_clses, topk_coords_px = decoder_centers_from_heatmap(
                accumulated_predictions.probas.unsqueeze(0), top_k=512
            )
            topk_scores = topk_scores[0].float().cpu().numpy()
            top_coords = topk_coords_px[0].float().cpu().numpy() * ANGSTROMS_IN_PIXEL
            topk_clses = topk_clses[0].cpu().numpy()

            for cls, coord, score in zip(topk_clses, top_coords, topk_scores):
                submission["experiment"].append(study_name)
                submission["particle_type"].append(CLASS_LABEL_TO_CLASS_NAME[int(cls)])
                submission["score"].append(float(score))
                submission["x"].append(float(coord[0]))
                submission["y"].append(float(coord[1]))
                submission["z"].append(float(coord[2]))

        submission = pd.DataFrame.from_dict(submission)
        # print(submission.sort_values(by="score", ascending=False).head(20))

        score_thresholds = np.linspace(0.05, 1.0, 20) ** 2
        score_values = []
        score_details = []

        for score_threshold in score_thresholds:
            keep_mask = submission["score"] >= score_threshold
            submission_filtered = submission[keep_mask]
            s = score_submission(
                solution=self.trainer.datamodule.solution.copy(),
                submission=submission_filtered.copy(),
                row_id_column_name="id",
                distance_multiplier=0.5,
                beta=4,
            )
            score_values.append(s[0])
            score_details.append(s[1])

        best_score = np.argmax(score_values)

        extra_values = dict(("val/" + k, v) for k, v in score_details[best_score].items())

        self.log_dict(
            {
                "val/score": score_values[best_score],
                "val/threshold": score_thresholds[best_score],
                **extra_values,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=False,
            rank_zero_only=False,
        )

    def validation_step(self, batch, batch_idx):
        num_items_in_batch = batch["labels"].eq(1).sum().item()
        outputs = self(**batch, num_items_in_batch=num_items_in_batch)

        self.accumulate_predictions(outputs, batch)

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

    def log_heatmaps(self, study_name: str, heatmap: Tensor):
        if self.trainer.is_global_zero:
            tb_logger = self._get_tb_logger(self.trainer)
            heatmap = render_heatmap(heatmap)

            if tb_logger is not None:
                tb_logger.add_images(
                    tag=f"val/{study_name}",
                    img_tensor=heatmap,
                    global_step=self.global_step,
                    dataformats="HWC",
                )

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
        elif self.train_args.optim == "sgd":
            optimizer = torch.optim.SGD(param_groups, momentum=0.9, nesterov=True, **optimizer_kwargs)
        else:
            raise KeyError(f"Unknown optimizer {self.train_args.optim}")

        warmup_steps = 0
        if self.train_args.warmup_steps > 0:
            warmup_steps = self.train_args.warmup_steps
            self.trainer.print(f"Using warmup steps: {warmup_steps}")
        elif self.train_args.warmup_ratio > 0:
            warmup_steps = int(self.train_args.warmup_ratio * self.trainer.estimated_stepping_batches)
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

    def lr_scheduler_step(self, scheduler: LRSchedulerTypeUnion, metric: Optional[Any]) -> None:
        scheduler.step()

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        # Log gradient norm (mean and max)
        if self.global_step % self.train_args.logging_steps == 0:
            with torch.no_grad():
                all_grads = torch.stack(
                    [torch.norm(p.grad) for p in self.model.parameters() if p.requires_grad and p.grad is not None]
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

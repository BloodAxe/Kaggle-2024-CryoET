import os
from typing import Optional, Any, Dict, List, Tuple

import lightning as L
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities.types import LRSchedulerTypeUnion
from matplotlib import pyplot as plt
from pytorch_toolbelt.optimization.functional import build_optimizer_param_groups
from pytorch_toolbelt.utils import all_gather
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter

from cryoet.modelling.detection.detection_head import ObjectDetectionOutput
from cryoet.schedulers import WarmupCosineScheduler
from .args import MyTrainingArguments, ModelArguments, DataArguments
from .od_accumulator import AccumulatedObjectDetectionPredictionContainer
from .visualization import render_heatmap
from ..data.parsers import CLASS_LABEL_TO_CLASS_NAME, ANGSTROMS_IN_PIXEL, TARGET_SIGMAS
from ..metric import score_submission
from ..modelling.detection.functional import decode_detections_with_nms


class EnsembleObjectDetectionModel(L.LightningModule):
    def __init__(
        self,
        *,
        models: List[nn.Module],
        data_args: DataArguments,
        model_args: ModelArguments,
        train_args: MyTrainingArguments,
    ):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.data_args = data_args
        self.train_args = train_args
        self.model_args = model_args
        self.validation_predictions = None
        self.average_tokens_across_devices = train_args.average_tokens_across_devices

    def forward(self, volume, labels=None, **loss_kwargs):
        outputs = []
        for model in self.models:
            output = model(
                volume=volume,
                labels=labels,
                apply_loss_on_each_stride=self.model_args.apply_loss_on_each_stride,
                average_tokens_across_devices=self.average_tokens_across_devices,
                use_l1_loss=self.train_args.use_l1_loss,
                use_offset_head=self.model_args.use_offset_head,
                assigner_max_anchors_per_point=self.model_args.assigner_max_anchors_per_point,
                assigner_alpha=self.model_args.assigner_alpha,
                assigner_beta=self.model_args.assigner_beta,
                **loss_kwargs,
            )
            outputs.append(output)
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self(
            **batch,
        )

        loss = sum([o.loss for o in outputs])

        # Apply KL
        # Pick a teacher model among the outputs, rest are students
        teacher_index = batch_idx % len(outputs)
        teacher_output = outputs[teacher_index]
        student_outputs = outputs[:teacher_index] + outputs[teacher_index + 1 :]

        teacher_logits_loss = 0
        teacher_offset_loss = 0

        for student_output in student_outputs:
            l1 = [
                torch.nn.functional.binary_cross_entropy_with_logits(student_logits, teacher_logits.detach().sigmoid())
                for student_logits, teacher_logits in zip(student_output.logits, teacher_output.logits)
            ]
            l2 = [
                torch.nn.functional.l1_loss(student_offset, teacher_offset.detach())
                for student_offset, teacher_offset in zip(student_output.offsets, teacher_output.offsets)
            ]

            teacher_logits_loss += sum(l1)
            teacher_offset_loss += sum(l2)

        loss = loss + teacher_logits_loss + teacher_offset_loss

        self.log_dict(
            dict(
                {
                    "train/teacher_logits_loss": teacher_logits_loss,
                    "train/teacher_offset_loss": teacher_offset_loss,
                }
            ),
            batch_size=len(batch["volume"]),
            sync_dist=True,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        for model_index, output in enumerate(outputs):
            self.log_dict(
                dict((f"train/{model_index}_{k}", v) for k, v in output.loss_dict.items()),
                batch_size=len(batch["volume"]),
                sync_dist=True,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        return loss

    def on_train_epoch_start(self) -> None:
        torch.cuda.empty_cache()

    def on_validation_start(self) -> None:
        self.validation_predictions: Dict[str, AccumulatedObjectDetectionPredictionContainer] = {}

    def accumulate_predictions(self, outputs: List[ObjectDetectionOutput], batch):
        tile_offsets_zyx = batch["tile_offsets_zyx"]

        strides = outputs[0].strides

        scores = []
        offsets = []
        num_feature_maps = len(scores)

        for i in range(num_feature_maps):
            averaged_probs = torch.stack([torch.sigmoid(output.logits[i]) for output in outputs], dim=0).mean(dim=0)
            scores.append(averaged_probs.cpu())

            averaged_offsets = torch.stack([output.offsets[i] for output in outputs], dim=0).mean(dim=0)
            offsets.append(averaged_offsets.cpu())

        num_classes = scores[0].shape[1]

        batch_size = len(batch["study"])
        for i in range(batch_size):
            study = batch["study"][i]
            volume_shape = batch["volume_shape"][i]
            tile_coord = tile_offsets_zyx[i]

            if study not in self.validation_predictions:
                self.validation_predictions[study] = AccumulatedObjectDetectionPredictionContainer.from_shape(
                    volume_shape, num_classes=num_classes, strides=strides, device="cpu", dtype=torch.float16
                )

            self.validation_predictions[study].accumulate(
                scores_list=[s[i] for s in scores],
                offsets_list=[o[i] for o in offsets],
                tile_coords_zyx=tile_coord,
            )

    def on_validation_epoch_end(self) -> None:
        torch.cuda.empty_cache()

        submission = dict(
            experiment=[],
            particle_type=[],
            score=[],
            x=[],
            y=[],
            z=[],
        )
        score_thresholds = np.linspace(0.14, 1.0, 20, endpoint=False) ** 2

        weights = {
            "apo-ferritin": 1,
            "beta-amylase": 0,
            "beta-galactosidase": 2,
            "ribosome": 1,
            "thyroglobulin": 2,
            "virus-like-particle": 1,
        }

        for study_name in self.trainer.datamodule.valid_studies:
            preds = self.validation_predictions.get(study_name, None)
            preds = all_gather(preds)

            preds = [p for p in preds if p is not None]
            if len(preds) == 0:
                continue

            accumulated_predictions = preds[0]
            for p in preds[1:]:
                accumulated_predictions += p

            scores, offsets = accumulated_predictions.merge_()

            # Save averaged heatmap for further postprocessing hyperparam tuning
            # if self.trainer.is_global_zero:
            #     torch.save({"scores": scores, "offsets": offsets}, os.path.join(self.trainer.log_dir, f"{study_name}.pth"))
            #
            #     self.trainer.datamodule.solution.to_csv(
            #         os.path.join(self.trainer.log_dir, f"{study_name}.csv"),
            #     )

            self.log_heatmaps(study_name, scores)

            topk_coords_px, topk_clases, topk_scores = decode_detections_with_nms(
                scores,
                offsets,
                strides=accumulated_predictions.strides,
                class_sigmas=TARGET_SIGMAS,
                min_score=score_thresholds.min(),
                iou_threshold=0.6,
                use_centernet_nms=self.model_args.use_centernet_nms,
                use_single_label_per_anchor=self.model_args.use_single_label_per_anchor,
                pre_nms_top_k=16536,
            )
            topk_scores = topk_scores.float().cpu().numpy()
            topk_coords = topk_coords_px.float().cpu().numpy() * ANGSTROMS_IN_PIXEL
            topk_clases = topk_clases.cpu().numpy()

            for cls, coord, score in zip(topk_clases, topk_coords, topk_scores):
                submission["experiment"].append(study_name)
                submission["particle_type"].append(CLASS_LABEL_TO_CLASS_NAME[int(cls)])
                submission["score"].append(float(score))
                submission["x"].append(float(coord[0]))
                submission["y"].append(float(coord[1]))
                submission["z"].append(float(coord[2]))

            # print("Added predictions for", study_name, "to dataframe")

        submission = pd.DataFrame.from_dict(submission)

        # self.trainer.print(submission.sort_values(by="score", ascending=False).head(20))

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

        keys = list(score_details[0].keys())
        per_class_scores = []
        for scores_dict in score_details:
            per_class_scores.append([scores_dict[k] for k in keys])
        per_class_scores = np.array(per_class_scores)  # [threshold, class]

        best_index_per_class = np.argmax(per_class_scores, axis=0)  # [class]
        best_threshold_per_class = np.array([score_thresholds[i] for i in best_index_per_class])  # [class]
        best_score_per_class = np.array([per_class_scores[i, j] for j, i in enumerate(best_index_per_class)])  # [class]
        averaged_score = np.sum([weights[k] * best_score_per_class[i] for i, k in enumerate(keys)]) / sum(weights.values())

        if self.trainer.is_global_zero:
            print("Scores", list(zip(score_values, score_thresholds)))

        self.log_plots(
            dict((key, (score_thresholds, per_class_scores[:, i])) for i, key in enumerate(keys)), "Threshold", "Score"
        )

        self.log_dict(
            {
                "val/score": averaged_score,
                **{f"val/{k}": best_score_per_class[i] for i, k in enumerate(keys)},
                **{f"val/{k}_threshold": best_threshold_per_class[i] for i, k in enumerate(keys)},
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=False,
            rank_zero_only=False,
        )

    def log_plots(self, plots: Dict[str, Tuple[np.ndarray, np.ndarray]], x_title, y_title):
        if self.trainer.is_global_zero:
            f = plt.figure()

            for key, (x, y) in plots.items():
                plt.plot(x, y, label=key)

            plt.xlabel(x_title)
            plt.ylabel(y_title)
            plt.legend()
            plt.tight_layout()

            tb_logger = self._get_tb_logger(self.trainer)
            if tb_logger is not None:
                tb_logger.add_figure("val/score_plot", f, global_step=self.global_step)

            plt.close(f)

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)

        self.accumulate_predictions(outputs, batch)

        # self.log_dict(
        #     dict(("val/" + k, v) for k, v in outputs.loss_dict.items()),
        #     batch_size=len(batch["volume"]),
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True,
        #     logger=True,
        #     sync_dist=True,
        # )
        return outputs

    def _get_tb_logger(self, trainer) -> Optional[SummaryWriter]:
        tb_logger: Optional[TensorBoardLogger] = None
        for logger in trainer.loggers:
            if isinstance(logger, TensorBoardLogger):
                tb_logger = logger
                break

        if tb_logger is not None:
            return tb_logger.experiment
        return None

    def log_heatmaps(self, study_name: str, heatmaps: List[Tensor]):
        if self.trainer.is_global_zero:
            tb_logger = self._get_tb_logger(self.trainer)
            for i, heatmap in enumerate(heatmaps):
                heatmap = render_heatmap(heatmap)

                if tb_logger is not None:
                    tb_logger.add_images(
                        tag=f"val/{study_name}_{i}",
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
        elif self.train_args.optim == "radam":
            optimizer = torch.optim.RAdam(
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

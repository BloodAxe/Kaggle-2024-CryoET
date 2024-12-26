import typing

import lightning as L
import torch
from dotenv import load_dotenv
from lightning.fabric import Fabric
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from pytorch_toolbelt.utils import count_parameters
from transformers import (
    HfArgumentParser,
)

from cryoet.data.point_detection_data_module import PointDetectionDataModule
from cryoet.modelling.configuration import SwinUNETRForPointDetectionConfig
from cryoet.modelling.unetr_point_detection import SwinUNETRForPointDetection
from cryoet.training.args import MyTrainingArguments, ModelArguments, DataArguments
from cryoet.training.point_detection_module import PointDetectionModel


def main():
    fabric = Fabric()
    parser = HfArgumentParser((MyTrainingArguments, ModelArguments, DataArguments))
    training_args, model_args, data_args = parser.parse_args_into_dataclasses()
    training_args = typing.cast(MyTrainingArguments, training_args)

    if training_args.output_dir is None:
        training_args.output_dir = (
            f"runs/swin_unetr_point_detection_fold_{data_args.fold}"
        )

    training_args.master_print(f"Training arguments: {training_args}")

    extra_model_args = {}
    if training_args.bf16:
        extra_model_args["torch_dtype"] = torch.bfloat16
    training_args.master_print(f"Extra model args: {extra_model_args}")

    # model_kwargs = build_model_args_from_commandline(model_args)
    model_kwargs = {}
    training_args.master_print(f"Model kwargs: {model_kwargs}")

    config = SwinUNETRForPointDetectionConfig()
    model = SwinUNETRForPointDetection(config)
    backbone_sd = torch.load(
        "pretrained/swin_unetr_btcv_segmentation/models/model.pt", weights_only=True
    )
    model.backbone.load_state_dict(backbone_sd, strict=True)

    training_args.master_print(
        f"Model parameters: {count_parameters(model, human_friendly=True)}"
    )

    # if training_args.gradient_checkpointing:
    #     model.gradient_checkpointing_enable()

    with fabric.rank_zero_first():
        data_module = PointDetectionDataModule(
            root=data_args.data_root,
            mode="denoised",
            window_size=96,
            stride=64,
            fold=data_args.fold,
            train_batch_size=training_args.per_device_train_batch_size,
            valid_batch_size=training_args.per_device_eval_batch_size,
            dataloader_num_workers=training_args.dataloader_num_workers,
            dataloader_persistent_workers=training_args.dataloader_persistent_workers,
            dataloader_pin_memory=training_args.dataloader_pin_memory,
        )

    model_module = PointDetectionModel(model, training_args)

    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        save_last=True,
        auto_insert_metric_name=False,
        save_top_k=3,
        filename="{step:03d}-loss-{val/loss:0.4f}",
    )

    loggers = []

    report_to = training_args.report_to
    if isinstance(report_to, list) and len(report_to) == 1:
        report_to = report_to[0]

    if "tensorboard" in report_to:
        logger = TensorBoardLogger(save_dir=training_args.output_dir)
        loggers.append(logger)
    if "wandb" in report_to:
        logger = WandbLogger()
        loggers.append(logger)
    if len(loggers) == 1:
        loggers = loggers[0]

    if fabric.world_size > 1:
        if training_args.ddp_find_unused_parameters:
            strategy = "ddp_find_unused_parameters_true"
        else:
            strategy = "ddp"
    else:
        strategy = "auto"

    callbacks = [checkpoint_callback]
    lr_monitor = LearningRateMonitor(logging_interval="step")
    if "wandb" not in training_args.report_to:
        callbacks.append(lr_monitor)

    precision = infer_training_precision(training_args)
    fabric.print(f"Training precision: {precision}")

    trainer = L.Trainer(
        strategy=strategy,
        max_epochs=int(training_args.num_train_epochs),
        max_steps=training_args.max_steps,
        precision=precision,
        log_every_n_steps=training_args.logging_steps,
        default_root_dir=training_args.output_dir,
        callbacks=callbacks,
        accumulate_grad_batches=training_args.gradient_accumulation_steps,
        gradient_clip_val=training_args.max_grad_norm,
        logger=loggers,
    )

    trainer.fit(model_module, datamodule=data_module)


def infer_training_precision(training_args):
    return "bf16-mixed" if training_args.bf16 else (16 if training_args.fp16 else 32)


if __name__ == "__main__":
    load_dotenv()
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.set_float32_matmul_precision("high")
    main()

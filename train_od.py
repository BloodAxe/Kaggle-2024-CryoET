import typing

import lightning as L
import torch
from dotenv import load_dotenv
from lightning.fabric import Fabric
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from pytorch_toolbelt.utils import count_parameters
from transformers import (
    HfArgumentParser,
)

from cryoet.data.detection.detection_data_module import ObjectDetectionDataModule
from cryoet.modelling.detection.segresnet_object_detection import SegResNetForObjectDetection, SegResNetForObjectDetectionConfig
from cryoet.training.args import MyTrainingArguments, ModelArguments, DataArguments
from cryoet.training.ema import BetaDecay, EMACallback
from cryoet.training.object_detection_module import ObjectDetectionModel


def main():
    fabric = Fabric()
    parser = HfArgumentParser((MyTrainingArguments, ModelArguments, DataArguments))
    training_args, model_args, data_args = parser.parse_args_into_dataclasses()
    training_args = typing.cast(MyTrainingArguments, training_args)

    data_args = typing.cast(DataArguments, data_args)

    if training_args.output_dir is None:
        output_dir_name = f"runs/od_{model_args.model_name}_fold_{data_args.fold}"
        if data_args.use_sliding_crops:
            output_dir_name += "_sc"
        if data_args.use_random_crops:
            output_dir_name += "_rc"
        if data_args.use_instance_crops:
            output_dir_name += "_ic"

        output_dir_name += "_" + data_args.train_modes.replace(",", "_")

        training_args.output_dir = output_dir_name

    training_args.master_print(f"Training arguments: {training_args}")

    extra_model_args = {}
    if training_args.bf16:
        extra_model_args["torch_dtype"] = torch.bfloat16
    training_args.master_print(f"Extra model args: {extra_model_args}")

    # model_kwargs = build_model_args_from_commandline(model_args)
    model_kwargs = {}
    training_args.master_print(f"Model kwargs: {model_kwargs}")

    if model_args.model_name == "segresnet":
        config = SegResNetForObjectDetectionConfig(window_size=model_args.window_size)
        model = SegResNetForObjectDetection(config)
    else:
        raise ValueError(f"Unknown model name: {model_args.model_name}")

    if model_args.pretrained_backbone_path is not None:
        backbone_sd = torch.load(model_args.pretrained_backbone_path, weights_only=True)
        model.backbone.load_state_dict(backbone_sd, strict=True)
        training_args.master_print(f"Loaded pretrained backbone from {model_args.pretrained_backbone_path}")

    training_args.master_print(f"Model parameters: {count_parameters(model, human_friendly=True)}")

    # if training_args.gradient_checkpointing:
    #     model.gradient_checkpointing_enable()

    with fabric.rank_zero_first():
        data_module = ObjectDetectionDataModule(
            data_args=data_args,
            train_args=training_args,
            window_size=model_args.window_size,
            stride=32,
        )

    model_module = ObjectDetectionModel(model, training_args)

    checkpoint_callback = ModelCheckpoint(
        monitor="val/score",
        mode="max",
        save_last=True,
        auto_insert_metric_name=False,
        save_top_k=5,
        filename=f"{model_args.model_name}_{model_args.window_size}_fold_{data_args.fold}"
        + "_{step:03d}-score-{val/score:0.4f}-threshold-{val/threshold:0.4f}",
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

    strategy = infer_strategy(training_args, fabric)

    callbacks = [checkpoint_callback]

    if training_args.early_stopping > 0:
        callbacks.append(EarlyStopping(monitor="val/score", min_delta=0.001, patience=training_args.early_stopping, mode="max"))

    lr_monitor = LearningRateMonitor(logging_interval="step")
    if "wandb" not in training_args.report_to:
        callbacks.append(lr_monitor)

    precision = infer_training_precision(training_args)
    fabric.print(f"Training precision: {precision}")

    if training_args.ema:
        callbacks.append(EMACallback(decay=BetaDecay(max_decay=training_args.ema_decay, beta=training_args.ema_beta)))
        fabric.print(f"Using EMA with decay={training_args.ema_decay} and beta={training_args.ema_beta}")

    fabric.print("Batch Size:", training_args.per_device_train_batch_size, training_args.per_device_eval_batch_size)

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


def infer_strategy(training_args, fabric):
    if fabric.world_size > 1:
        if training_args.ddp_find_unused_parameters:
            strategy = "ddp_find_unused_parameters_true"
        else:
            strategy = "ddp"
    else:
        strategy = "auto"
    return strategy


def infer_training_precision(training_args):
    return "bf16-mixed" if training_args.bf16 else (16 if training_args.fp16 else 32)


if __name__ == "__main__":
    load_dotenv()
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.set_float32_matmul_precision("high")
    torch.autograd.set_detect_anomaly(True)
    main()

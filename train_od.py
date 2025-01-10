import json
import os
import typing

import lightning as L
import torch
from dotenv import load_dotenv
from lightning.fabric import Fabric
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from pytorch_toolbelt.utils import count_parameters, transfer_weights
from transformers import (
    HfArgumentParser,
)

from cryoet.data.detection.data_module import ObjectDetectionDataModule
from cryoet.modelling.detection.dynunet import DynUNetForObjectDetectionConfig, DynUNetForObjectDetection
from cryoet.modelling.detection.maxvit_unet25d import MaxVitUnet25d, MaxVitUnet25dConfig
from cryoet.modelling.detection.segresnet_object_detection import SegResNetForObjectDetection, SegResNetForObjectDetectionConfig
from cryoet.modelling.detection.segresnet_object_detection_v2 import (
    SegResNetForObjectDetectionV2,
    SegResNetForObjectDetectionV2Config,
)
from cryoet.modelling.detection.unet3d_detection import UNet3DForObjectDetection, UNet3DForObjectDetectionConfig
from cryoet.modelling.detection.unetr import SwinUNETRForObjectDetection, SwinUNETRForObjectDetectionConfig

from cryoet.training.args import MyTrainingArguments, ModelArguments, DataArguments
from cryoet.training.ema import BetaDecay, EMACallback
from cryoet.training.object_detection_module import ObjectDetectionModel


def main():
    fabric = Fabric()
    parser = HfArgumentParser((MyTrainingArguments, ModelArguments, DataArguments))
    training_args, model_args, data_args = parser.parse_args_into_dataclasses()
    training_args = typing.cast(MyTrainingArguments, training_args)
    model_args = typing.cast(ModelArguments, model_args)

    data_args = typing.cast(DataArguments, data_args)

    if training_args.output_dir is None:
        output_dir_name = f"runs/od_{model_args.model_name}_fold_{data_args.fold}"
        if data_args.use_sliding_crops:
            output_dir_name += "_sc"
        if data_args.use_random_crops:
            output_dir_name += "_rc"
        if data_args.use_instance_crops:
            output_dir_name += "_ic"
        if model_args.use_stride2 and not model_args.use_stride4:
            output_dir_name += "_s2"
        if model_args.use_stride4 and not model_args.use_stride2:
            output_dir_name += "_s4"
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
    elif model_args.model_name == "segresnetv2":
        config = SegResNetForObjectDetectionV2Config(use_stride2=model_args.use_stride2, use_stride4=model_args.use_stride4)
        model = SegResNetForObjectDetectionV2(config)
    elif model_args.model_name == "unet3d":
        config = UNet3DForObjectDetectionConfig(window_size=model_args.window_size)
        model = UNet3DForObjectDetection(config)
    elif model_args.model_name == "maxvit_nano_unet25d":
        config = MaxVitUnet25dConfig(img_size=model_args.window_size)
        model = MaxVitUnet25d(config)
    elif model_args.model_name == "dynunet":
        config = DynUNetForObjectDetectionConfig(use_stride2=model_args.use_stride2, use_stride4=model_args.use_stride4)
        model = DynUNetForObjectDetection(config)
    elif model_args.model_name == "dynunet_v2":
        config = DynUNetForObjectDetectionConfig(
            # act_name="gelu",
            # dropout=0.1,
            # res_block=True,
            use_stride2=model_args.use_stride2,
            use_stride4=model_args.use_stride4,
        )
        model = DynUNetForObjectDetection(config)
    elif model_args.model_name == "unet3d-fat":
        config = UNet3DForObjectDetectionConfig(
            encoder_channels=[32, 64, 128, 256],
            num_blocks_per_stage=(2, 3, 4, 6),
            num_blocks_per_decoder_stage=(2, 2, 2),
            intermediate_channels=64,
            offset_intermediate_channels=16,
            window_size=model_args.window_size,
        )
        model = UNet3DForObjectDetection(config)
    elif model_args.model_name == "unetr":
        config = SwinUNETRForObjectDetectionConfig()
        model = SwinUNETRForObjectDetection(config)
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
            model_args=model_args,
        )

    model_module = ObjectDetectionModel(model=model, data_args=data_args, model_args=model_args, train_args=training_args)

    if training_args.transfer_weights is not None:
        checkpoint = torch.load(training_args.transfer_weights, map_location="cpu")
        transfer_weights(model_module, checkpoint["state_dict"])
        training_args.master_print(f"Loaded weights from {training_args.transfer_weights}")

    checkpoint_callback = ModelCheckpoint(
        monitor="val/score",
        mode="max",
        save_last=True,
        auto_insert_metric_name=False,
        save_top_k=5,
        filename=f"{model_args.model_name}_{model_args.depth_window_size}x{model_args.spatial_window_size}x{model_args.spatial_window_size}_fold_{data_args.fold}"
        + "_{step:03d}-score-{val/score:0.4f}-at-{val/apo-ferritin_threshold:0.3f}-{val/beta-galactosidase_threshold:0.3f}-{val/ribosome_threshold:0.3f}-{val/thyroglobulin_threshold:0.3f}-{val/virus-like-particle_threshold:0.3f}",
        # + "_{step:03d}-score-{val/score:0.4f}",
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
        num_sanity_val_steps=0,
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

    # Save hyperparams
    config = {**training_args.to_dict(), **model_args.to_dict(), **data_args.to_dict()}
    with open(os.path.join(training_args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4, sort_keys=True)


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
    torch.backends.cudnn.benchmark = True
    main()

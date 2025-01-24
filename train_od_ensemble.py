import json
import os
import time
import traceback
import typing
from datetime import datetime
from pathlib import Path

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
from pytorch_toolbelt.utils import count_parameters, transfer_weights, is_dist_avail_and_initialized

from cryoet.data.detection.data_module import ObjectDetectionDataModule
from cryoet.ensembling import average_checkpoints
from cryoet.modelling.detection.dynunet import DynUNetForObjectDetectionConfig, DynUNetForObjectDetection
from cryoet.modelling.detection.litehrnet import HRNetv2ForObjectDetection, HRNetv2ForObjectDetectionConfig
from cryoet.modelling.detection.segresnet_object_detection_v2 import (
    SegResNetForObjectDetectionV2,
    SegResNetForObjectDetectionV2Config,
)
from cryoet.training.args import MyTrainingArguments, ModelArguments, DataArguments
from cryoet.training.ema import BetaDecay, EMACallback
from cryoet.training.ensemble_object_detection_module_v2 import EnsembleObjectDetectionModelV2
from cryoet.training.ensemble_object_detection_module import EnsembleObjectDetectionModel
from train_od import build_model_name_slug


def main():
    fabric = Fabric()
    parser = HfArgumentParser((MyTrainingArguments, ModelArguments, DataArguments))
    training_args, model_args, data_args = parser.parse_args_into_dataclasses()
    training_args = typing.cast(MyTrainingArguments, training_args)
    model_args = typing.cast(ModelArguments, model_args)
    data_args = typing.cast(DataArguments, data_args)

    L.seed_everything(training_args.seed)
    model_name_slug = build_model_name_slug(data_args, model_args, training_args)

    # Make timestamp to differentiate runs in YYMMDD_HHMM format
    timestamp = datetime.now().strftime("%y%m%d_%H%M")

    if training_args.output_dir is None:
        output_dir_name = f"runs/{model_name_slug}"

        training_args_str = f"{training_args.optim.value}_lr_{training_args.learning_rate:.0e}_wd_{training_args.weight_decay}_b1_{training_args.adam_beta1}_b2_{training_args.adam_beta2}"
        if training_args.ema:
            training_args_str += f"_ema_{training_args.ema_decay}_{training_args.ema_beta}"

        training_args.output_dir = os.path.join(output_dir_name, f"{timestamp}_{training_args_str}")

    try:
        num_classes = 6 if model_args.use_6_classes else 5

        model1 = SegResNetForObjectDetectionV2(
            config=SegResNetForObjectDetectionV2Config(
                use_stride2=model_args.use_stride2,
                use_stride4=model_args.use_stride4,
                use_offset_head=model_args.use_offset_head,
                num_classes=num_classes,
            )
        )
        if model_args.pretrained_backbone_path is not None:
            backbone_sd = torch.load(model_args.pretrained_backbone_path, weights_only=True)
            model1.backbone.load_state_dict(backbone_sd, strict=True)
            training_args.master_print(f"Loaded pretrained backbone from {model_args.pretrained_backbone_path}")

        model2 = DynUNetForObjectDetection(
            config=DynUNetForObjectDetectionConfig(
                use_stride2=model_args.use_stride2,
                use_stride4=model_args.use_stride4,
                num_classes=num_classes,
            )
        )

        model3 = HRNetv2ForObjectDetection(
            config=HRNetv2ForObjectDetectionConfig(
                # use_stride2=model_args.use_stride2,
                # use_stride4=model_args.use_stride4,
                num_classes=num_classes,
            )
        )

        training_args.master_print(f"Model parameters: {count_parameters(model1, human_friendly=True)}")
        training_args.master_print(f"Model parameters: {count_parameters(model2, human_friendly=True)}")
        training_args.master_print(f"Model parameters: {count_parameters(model3, human_friendly=True)}")

        with fabric.rank_zero_first():
            data_module = ObjectDetectionDataModule(
                data_args=data_args,
                train_args=training_args,
                model_args=model_args,
            )

        if model_args.model_name == "ensemblev1":
            model_module = EnsembleObjectDetectionModel(
                models=[model1, model2, model3], data_args=data_args, model_args=model_args, train_args=training_args
            )
        elif model_args.model_name == "ensemblev2":
            model_module = EnsembleObjectDetectionModelV2(
                models=[model1, model2, model3], data_args=data_args, model_args=model_args, train_args=training_args
            )
        else:
            raise ValueError(f"Unknown model name: {model_args.model_name}")

        if training_args.transfer_weights is not None:
            checkpoint = torch.load(training_args.transfer_weights, map_location="cpu")
            transfer_weights(model_module, checkpoint["state_dict"])
            training_args.master_print(f"Loaded weights from {training_args.transfer_weights}")

        checkpoint_callback = ModelCheckpoint(
            dirpath=training_args.output_dir,
            enable_version_counter=False,
            monitor="val/score",
            mode="max",
            save_last=True,
            auto_insert_metric_name=False,
            save_top_k=5,
            filename=f"{timestamp}_{model_name_slug}"
            + "_{step:03d}-score-{val/score:0.4f}-at-{val/apo-ferritin_threshold:0.3f}-{val/beta-galactosidase_threshold:0.3f}-{val/ribosome_threshold:0.3f}-{val/thyroglobulin_threshold:0.3f}-{val/virus-like-particle_threshold:0.3f}",
            # + "_{step:03d}-score-{val/score:0.4f}",
        )

        loggers = []

        report_to = training_args.report_to
        if isinstance(report_to, list) and len(report_to) == 1:
            report_to = report_to[0]

        if "tensorboard" in report_to:
            logger = TensorBoardLogger(save_dir=training_args.output_dir, name=None, version="")
            loggers.append(logger)
        if "wandb" in report_to:
            logger = WandbLogger()
            loggers.append(logger)
        if len(loggers) == 1:
            loggers = loggers[0]

        strategy = infer_strategy(training_args, fabric)

        callbacks = [checkpoint_callback]

        if training_args.early_stopping > 0:
            callbacks.append(
                EarlyStopping(monitor="val/score", min_delta=0.001, patience=training_args.early_stopping, mode="max")
            )

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
        if trainer.is_global_zero:
            config = {**training_args.to_dict(), **model_args.to_dict(), **data_args.to_dict()}
            with open(os.path.join(training_args.output_dir, "config.json"), "w") as f:
                json.dump(config, f, indent=4, sort_keys=True)
            print("Saved config")

        # Trace & Save
        models_output_dir = Path(checkpoint_callback.best_model_path).parent

        # Average checkpoint
        best_k_models = list(checkpoint_callback.best_k_models.keys())
        averaged_filename = f"{timestamp}_{model_name_slug}"
        tmp_averaged_checkpoint = models_output_dir / f"{averaged_filename}.pt"

        if trainer.is_global_zero:
            average_checkpoints(*best_k_models, output_path=tmp_averaged_checkpoint)
            print("Averaged checkpoints")

        if is_dist_avail_and_initialized():
            torch.distributed.barrier()

        model_module.load_state_dict(
            torch.load(tmp_averaged_checkpoint, map_location=model_module.device, weights_only=True)["state_dict"]
        )
        metrics = trainer.validate(model=model_module, datamodule=data_module)
        print(metrics)

    except Exception as e:
        with open(os.path.join(training_args.output_dir, f"error_rank_{training_args.local_rank}.log"), "w") as f:
            f.write(str(e))
            f.write("\n")
            f.write(traceback.format_exc())


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

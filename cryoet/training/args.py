import os
import typing
from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments


@dataclass
class ModelArguments:

    pretrained_model_name_or_path: str = field(
        default=None,
    )

    pretrained_backbone_path: str = field(
        default=None,
    )


def data_root_default_factory():
    return os.environ.get("CRYOET_DATA_ROOT", None)


@dataclass
class DataArguments:
    data_root: str = field(default_factory=data_root_default_factory)
    fold: int = field(default=0)

    use_random_crops: bool = field(default=False)
    use_instance_crops: bool = field(default=False)


@dataclass
class MyTrainingArguments(TrainingArguments):
    bf16: bool = field(default=True, metadata={"help": "Use bfloat16 precision"})
    tf32: bool = field(default=True, metadata={"help": "Use tf32 precision"})

    save_total_limit: int = field(default=3, metadata={"help": "Save total limit"})
    max_grad_norm: float = field(default=3.0, metadata={"help": "Max grad norm"})

    save_strategy: str = field(default="epoch", metadata={"help": "Save strategy"})
    eval_strategy: str = field(default="epoch", metadata={"help": "Evaluation strategy"})

    logging_steps: int = field(default=1, metadata={"help": "Logging steps"})

    gradient_checkpointing: bool = field(default=True, metadata={"help": "Gradient checkpointing"})

    load_best_model_at_end: bool = field(default=True, metadata={"help": "Load best model at the end"})

    ddp_find_unused_parameters: bool = field(default=False, metadata={"help": "DDP find unused parameters"})

    metric_for_best_model: str = field(default="eval_loss", metadata={"help": "Metric for best model"})

    greater_is_better: bool = field(default=False, metadata={"help": "Greater is better"})

    learning_rate: float = field(default=3e-4, metadata={"help": "Learning rate"})
    weight_decay: float = field(default=0.0001, metadata={"help": "Weight decay"})

    report_to: typing.Union[str, typing.List[str]] = field(default="tensorboard", metadata={"help": "Report to"})

    optim: str = field(default="adamw_torch")

    output_dir: str = None

    adam_beta1: float = field(default=0.9, metadata={"help": "Adam beta1"})
    adam_beta2: float = field(default=0.95, metadata={"help": "Adam beta2"})

    sanity_checking: bool = field(default=False, metadata={"help": "Sanity checking"})

    def master_print(self, *args):
        if self.process_index == 0:
            print(*args)


def build_model_args_from_commandline(model_args: ModelArguments):
    model_kwargs = {}
    if model_args.num_groups is not None:
        model_kwargs["num_groups"] = model_args.num_groups
    if model_args.num_hidden_layers is not None:
        model_kwargs["num_hidden_layers"] = model_args.num_hidden_layers
    if model_args.num_attention_heads is not None:
        model_kwargs["num_attention_heads"] = model_args.num_attention_heads
    if model_args.norm_type is not None:
        model_kwargs["norm_type"] = model_args.norm_type
    if model_args.activation is not None:
        model_kwargs["activation"] = model_args.activation
    if model_args.loss_type is not None:
        model_kwargs["loss_type"] = model_args.loss_type
    if model_args.use_deep_supervision_loss:
        model_kwargs["use_deep_supervision_loss"] = model_args.use_deep_supervision_loss
    if model_args.lm_head_bias_init is not None:
        model_kwargs["lm_head_bias_init"] = model_args.lm_head_bias_init
    if model_args.hidden_size is not None:
        model_kwargs["hidden_size"] = model_args.hidden_size
    if model_args.use_twin_mlp:
        model_kwargs["use_twin_mlp"] = model_args.use_twin_mlp
    if model_args.learnable_residuals:
        model_kwargs["learnable_residuals"] = model_args.learnable_residuals
    if model_args.depthwise_linear_groups is not None:
        model_kwargs["depthwise_linear_groups"] = model_args.depthwise_linear_groups
    return model_kwargs

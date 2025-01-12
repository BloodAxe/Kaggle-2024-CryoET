import os
import typing
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Optional, Any, Dict

from transformers import TrainingArguments


def data_root_default_factory():
    return os.environ.get("CRYOET_DATA_ROOT", None)


@dataclass
class ModelArguments:
    model_name: str = field(
        default="segresnet",
    )
    pretrained_model_name_or_path: str = field(
        default=None,
    )

    pretrained_backbone_path: str = field(
        default=None,
    )

    use_qfl_loss: bool = field(default=False)

    spatial_window_size: int = field(default=128)
    depth_window_size: int = field(default=96)

    spatial_window_step: int = field(default=90)
    depth_window_step: int = field(default=62)

    use_centernet_nms: bool = field(
        default=True, metadata={"help": "Enable CenterNet NMS when decoding. Relevant only for OD models."}
    )

    use_single_label_per_anchor: bool = field(
        default=True, metadata={"help": "If true, only one label per anchor is used in post-processing"}
    )

    use_offset_head: bool = field(default=True)

    use_stride4: bool = field(default=True)
    use_stride2: bool = field(default=True)

    assigner_max_anchors_per_point: int = field(default=13, metadata={"help": "Maximum number of anchors per point"})
    assigner_alpha: float = field(default=1.0, metadata={"help": "Alpha"})
    assigner_beta: float = field(default=6.0, metadata={"help": "Beta"})

    apply_loss_on_each_stride: bool = field(default=False)

    def _dict_torch_dtype_to_str(self, d: Dict[str, Any]) -> None:
        """
        Checks whether the passed dictionary and its nested dicts have a *torch_dtype* key and if it's not None,
        converts torch.dtype to a string of just the type. For example, `torch.float32` get converted into *"float32"*
        string, which can then be stored in the json format.
        """
        if d.get("torch_dtype", None) is not None and not isinstance(d["torch_dtype"], str):
            d["torch_dtype"] = str(d["torch_dtype"]).split(".")[1]
        for value in d.values():
            if isinstance(value, dict):
                self._dict_torch_dtype_to_str(value)

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        # filter out fields that are defined as field(init=False)
        d = {field.name: getattr(self, field.name) for field in fields(self) if field.init}

        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
        self._dict_torch_dtype_to_str(d)

        return d


@dataclass
class DataArguments:
    data_root: str = field(default_factory=data_root_default_factory)
    fold: int = field(default=0)

    use_sliding_crops: bool = field(default=False)
    use_random_crops: bool = field(default=False)
    use_instance_crops: bool = field(default=False)

    use_random_flips: bool = field(default=True)

    train_modes: str = field(default="denoised")
    valid_modes: str = field(default="denoised")

    num_crops_per_study: int = field(default=256)

    scale_limit: float = field(default=0.05)

    z_rotation_limit: float = field(default=360.0)
    y_rotation_limit: float = field(default=0.0)
    x_rotation_limit: float = field(default=0.0)

    random_erase_prob: float = field(default=0.0)

    copy_paste_prob: float = field(default=0.0)
    copy_paste_limit: int = field(default=1)

    gaussian_noise_sigma: float = field(default=0.0)

    def _dict_torch_dtype_to_str(self, d: Dict[str, Any]) -> None:
        """
        Checks whether the passed dictionary and its nested dicts have a *torch_dtype* key and if it's not None,
        converts torch.dtype to a string of just the type. For example, `torch.float32` get converted into *"float32"*
        string, which can then be stored in the json format.
        """
        if d.get("torch_dtype", None) is not None and not isinstance(d["torch_dtype"], str):
            d["torch_dtype"] = str(d["torch_dtype"]).split(".")[1]
        for value in d.values():
            if isinstance(value, dict):
                self._dict_torch_dtype_to_str(value)

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        # filter out fields that are defined as field(init=False)
        d = {field.name: getattr(self, field.name) for field in fields(self) if field.init}

        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"

        self._dict_torch_dtype_to_str(d)

        return d


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

    output_dir: str = None

    adam_beta1: float = field(default=0.9, metadata={"help": "Adam beta1"})
    adam_beta2: float = field(default=0.95, metadata={"help": "Adam beta2"})

    sanity_checking: bool = field(default=False, metadata={"help": "Sanity checking"})

    ema: bool = field(default=False, metadata={"help": "Exponential moving average"})
    ema_decay: float = field(default=0.995, metadata={"help": "Exponential moving average decay"})
    ema_beta: float = field(default=10, metadata={"help": "Exponential moving average beta"})

    early_stopping: int = field(default=0, metadata={"help": "Early stopping"})

    use_l1_loss: bool = field(default=False, metadata={"help": "If true, adds L1 loss on offsets prediction"})

    transfer_weights: Optional[str] = field(default=None, metadata={"help": "Path to the weights to transfer"})

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

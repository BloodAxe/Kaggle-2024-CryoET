from pathlib import Path
from typing import List, Union, Tuple, Optional

import torch
import os

from torch import nn
from pytorch_toolbelt.utils import get_non_wrapped_model

from cryoet.modelling.ch_models.adaptors import MdlCh20dCe2c2_resnet34, MdlCh20dCe2_resnet34, MdlCh20dCe2_effnetb3
from cryoet.modelling.detection.dynunet import DynUNetForObjectDetectionConfig, DynUNetForObjectDetection
from cryoet.modelling.detection.litehrnet import HRNetv2ForObjectDetectionConfig, HRNetv2ForObjectDetection
from cryoet.modelling.detection.segresnet_object_detection_v2 import (
    SegResNetForObjectDetectionV2Config,
    SegResNetForObjectDetectionV2,
)


def infer_model_device(model: nn.Module) -> Optional[torch.device]:
    """
    Get the device where the model's parameters are stored.
    This function returns device of the first parameter of the model, assuming there is no
    cross-device parameter movement inside the model.
    :param model: Model to get the device from.
    :return: Device where the model's parameters are stored.
             The function may return None if the model has no parameters or buffers.
    """
    try:
        first_parameter = next(iter(model.parameters()))
        return first_parameter.device
    except StopIteration:
        try:
            first_buffer = next(iter(model.buffers()))
            return first_buffer.device
        except StopIteration:
            return None


def average_checkpoints(*ckpt_paths: str, output_path: Union[str, Path] = "averaged_model.pt"):
    """
    Averages the 'state_dict' weights of multiple PyTorch checkpoints.

    Args:
        ckpt_paths (list of str): List of paths to checkpoint files.
        output_path (str): Path to save the averaged checkpoint.
    """

    if not ckpt_paths:
        raise ValueError("No checkpoint paths were provided.")

    # Load state dicts from each checkpoint
    state_dicts = []
    for path in ckpt_paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint file '{path}' does not exist.")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if "state_dict" not in checkpoint:
            raise KeyError(f"'state_dict' not found in checkpoint '{path}'.")
        state_dicts.append(checkpoint["state_dict"])

    # Get all keys from the first checkpoint
    keys = state_dicts[0].keys()
    final_state_dict = {}

    for key in keys:
        # Collect the values (tensors) for this key from all checkpoints
        values = [sd[key] for sd in state_dicts]

        # Check the dtype of the first value (assuming all dtypes match)
        first_val = values[0]

        if not all(v.shape == first_val.shape for v in values):
            raise ValueError(f"Tensor shapes for key '{key}' are not consistent across checkpoints.")

        if first_val.dtype == torch.bool:
            # For bool, ensure all are identical
            for val in values[1:]:
                if not torch.equal(val, first_val):
                    raise ValueError(f"Boolean values for key '{key}' differ between checkpoints.")
            final_state_dict[key] = first_val  # Use the first if all identical

        elif torch.is_floating_point(first_val):
            # Average float values
            stacked = torch.stack(values, dim=0)
            averaged = stacked.mean(dim=0)
            final_state_dict[key] = averaged

        elif first_val.dtype in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
            torch.uint16,
            torch.uint32,
            torch.uint64,
        }:
            # Average integer values (using integer division)
            stacked = torch.stack(values, dim=0)
            summed = stacked.sum(dim=0, dtype=torch.int64)
            averaged = summed // len(values)
            final_state_dict[key] = averaged.to(first_val.dtype)

        else:
            # If you have other special dtypes to handle, add logic here
            # or simply copy the first value if that is your intended behavior.
            raise TypeError(f"Unsupported dtype '{first_val.dtype}' encountered for key '{key}'.")

    # Save the final averaged state_dict
    torch.save({"state_dict": final_state_dict}, str(output_path))
    print(f"Averaged checkpoint saved to: {output_path}")


def trace_model_and_save(window_size: Tuple[int, int, int], model: nn.Module, traced_checkpoint_path: Path):
    model = get_non_wrapped_model(model)
    device = infer_model_device(model)

    with torch.no_grad():
        example_input = torch.randn(
            1,
            1,
            *window_size,
        ).to(device)
        traced_model = torch.jit.trace(model.eval(), example_input)
        torch.jit.save(traced_model, str(traced_checkpoint_path))


def jit_model_from_checkpoint(
    checkpoint_path: Path,
    torch_device="cpu",
    torch_dtype=torch.float16,
    num_classes=6,
    use_stride2=True,
    use_stride4=False,
    window_size=(192, 128, 128),
    **kwargs,
):
    checkpoint_path = str(checkpoint_path)
    if checkpoint_path.endswith(".jit"):
        traced_model = torch.jit.load(checkpoint_path, map_location=torch_device).to(torch_dtype)
    elif checkpoint_path.endswith(".pt") or checkpoint_path.endswith(".ckpt") or checkpoint_path.endswith(".pth"):
        model = model_from_checkpoint(
            checkpoint_path, num_classes=num_classes, use_stride2=use_stride2, use_stride4=use_stride4, **kwargs
        )
        model = model.to(device=torch_device, dtype=torch_dtype).eval()
        example_input = torch.randn(1, 1, *window_size).to(device=torch_device, dtype=torch_dtype)
        traced_model = torch.jit.trace(model, example_input)
    else:
        raise ValueError(f"Unknown checkpoint extension: {checkpoint_path}")

    return traced_model


def model_from_checkpoint(checkpoint_path: Path | str, **kwargs):
    checkpoint_path = Path(checkpoint_path)
    model_prefix = None

    if "hrnet" in checkpoint_path.stem:
        config = HRNetv2ForObjectDetectionConfig(**kwargs)
        model = HRNetv2ForObjectDetection(config)
        state_dict_key = "state_dict"
        model_prefix = "model."
    elif "dynunet" in checkpoint_path.stem:
        config = DynUNetForObjectDetectionConfig(**kwargs)
        model = DynUNetForObjectDetection(config)
        state_dict_key = "state_dict"
        model_prefix = "model."
    elif "segresnetv2" in checkpoint_path.stem:
        config = SegResNetForObjectDetectionV2Config(**kwargs)
        model = SegResNetForObjectDetectionV2(config)
        state_dict_key = "state_dict"
        model_prefix = "model."
    elif "weights-cryo-cfg-ch-48h-ce2c2" in str(checkpoint_path) or "cfg_ch_48h_ce2c2_bf2_ek" in str(checkpoint_path):
        model = MdlCh20dCe2c2_resnet34()
        state_dict_key = "model"
    elif "weights-cryo-cfg-ch-48j2" in str(checkpoint_path) or "cfg_ch_48j2_bf162_ek" in str(checkpoint_path):
        model = MdlCh20dCe2_resnet34()
        state_dict_key = "model"
    elif "weights-cryo-cfg-ch-48k" in str(checkpoint_path) or "cfg_ch_48k_bf162_ek" in str(checkpoint_path):
        model = MdlCh20dCe2_effnetb3()
        state_dict_key = "model"
    else:
        raise ValueError(f"Unknown model type: {checkpoint_path.stem}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_state_dict = checkpoint[state_dict_key] if state_dict_key is not None else checkpoint
    if model_prefix is not None:
        model_state_dict = {k.replace(model_prefix, ""): v for k, v in model_state_dict.items() if k.startswith(model_prefix)}

    model.load_state_dict(model_state_dict, strict=True)
    return model


def infer_fold(checkpoint_name):
    for i in range(7):
        if f"fold_{i}" in checkpoint_name or f"fold-{i}" in checkpoint_name or f"fold{i}" in checkpoint_name:
            return i

    raise ValueError(f"Could not infer fold from checkpoint name: {checkpoint_name}")

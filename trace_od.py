from pathlib import Path

import torch

from cryoet.modelling.detection.segresnet_object_detection_v2 import (
    SegResNetForObjectDetectionV2,
    SegResNetForObjectDetectionConfig,
)


def trace_and_save(checkpoint_path, traced_checkpoint_path, window_size=96):
    checkpoint = torch.load(str(checkpoint_path), weights_only=True)

    config = SegResNetForObjectDetectionConfig(window_size=window_size)
    model = SegResNetForObjectDetectionV2(config).cuda().eval()
    model_state_dict = checkpoint["state_dict"]
    model_state_dict = {k.replace("model.", ""): v for k, v in model_state_dict.items()}

    model.load_state_dict(model_state_dict, strict=True)

    example_input = torch.randn(1, 1, window_size, window_size, window_size).cuda()
    traced_model = torch.jit.trace(model, example_input)
    torch.jit.save(traced_model, str(traced_checkpoint_path))


def main(*checkpoints, window_size=96):
    for checkpoint in checkpoints:
        checkpoint_path = Path(checkpoint)
        traced_checkpoint_path = checkpoint_path.with_suffix(".jit")

        trace_and_save(checkpoint_path, traced_checkpoint_path, window_size=window_size)


if __name__ == "__main__":
    import fire

    fire.Fire(main)

from pathlib import Path

import torch

from cryoet.modelling.detection.segresnet_object_detection_v2 import (
    SegResNetForObjectDetectionV2,
    SegResNetForObjectDetectionV2Config,
)


def trace_and_save(checkpoint_path, traced_checkpoint_path, window_size=96, **kwargs):
    print("kwargs", kwargs)
    checkpoint = torch.load(str(checkpoint_path), weights_only=True)

    config = SegResNetForObjectDetectionV2Config(use_stride4=False, **kwargs)
    model = SegResNetForObjectDetectionV2(config).cuda().eval()
    model_state_dict = checkpoint["state_dict"]
    model_state_dict = {k.replace("model.", ""): v for k, v in model_state_dict.items() if k.startswith("model.")}

    model.load_state_dict(model_state_dict, strict=True)

    example_input = torch.randn(1, 1, window_size, window_size, window_size).cuda()
    traced_model = torch.jit.trace(model, example_input)
    torch.jit.save(traced_model, str(traced_checkpoint_path))


def main(*checkpoints, window_size=96, **kwargs):
    for checkpoint in checkpoints:
        checkpoint_path = Path(checkpoint)
        traced_checkpoint_path = checkpoint_path.with_suffix(".jit")

        trace_and_save(checkpoint_path, traced_checkpoint_path, window_size=window_size, **kwargs)


if __name__ == "__main__":
    import fire

    fire.Fire(main)

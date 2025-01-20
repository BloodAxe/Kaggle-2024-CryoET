from pathlib import Path

import torch

from cryoet.modelling.detection.dynunet import (
    DynUNetForObjectDetection,
    DynUNetForObjectDetectionConfig,
)


def trace_and_save(checkpoint_path, traced_checkpoint_path, window_size=96, use_stride2=True, use_stride4=False, num_classes=5):
    checkpoint = torch.load(str(checkpoint_path), weights_only=True)

    config = DynUNetForObjectDetectionConfig(
        res_block=True,
        use_stride2=use_stride2,
        use_stride4=use_stride4,
        num_classes=num_classes,
        object_size=32,
        intermediate_channels=64,
        offset_intermediate_channels=8,
    )
    model = DynUNetForObjectDetection(config).cuda().eval()
    model_state_dict = checkpoint["state_dict"]
    model_state_dict = {k.replace("model.", ""): v for k, v in model_state_dict.items() if k.startswith("model.")}

    model.load_state_dict(model_state_dict, strict=True)

    example_input = torch.randn(1, 1, window_size, window_size, window_size).cuda()
    traced_model = torch.jit.trace(model, example_input)
    torch.jit.save(traced_model, str(traced_checkpoint_path))


def main(*checkpoints, **kwargs):
    for checkpoint in checkpoints:
        checkpoint_path = Path(checkpoint)
        traced_checkpoint_path = checkpoint_path.with_suffix(".jit")

        trace_and_save(checkpoint_path, traced_checkpoint_path, **kwargs)


if __name__ == "__main__":
    import fire

    fire.Fire(main)

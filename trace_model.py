from pathlib import Path

import torch

from cryoet.ensembling import model_from_checkpoint


@torch.no_grad()
def trace_and_save(checkpoint_path, traced_checkpoint_path, **kwargs):
    model = model_from_checkpoint(checkpoint_path, **kwargs)
    model = model.cuda().eval()

    example_input = torch.randn(1, 1, 192, 128, 128).cuda()
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

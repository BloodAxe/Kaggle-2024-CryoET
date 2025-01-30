from pathlib import Path

import torch

from cryoet.ensembling import model_from_checkpoint


def check_model_can_be_cast_to_fp16(model: torch.nn.Module):
    finfo = torch.finfo(torch.float16)

    for name, p in model.named_parameters():
        if torch.is_floating_point(p):
            # Check if a is in the range of fp16
            p_numel = p.numel()

            # Compute fraction of values that are below the minimum
            p_lt_min = (p < finfo.min).sum().item() / p_numel
            p_gt_max = (p > finfo.max).sum().item() / p_numel

            if p_lt_min > 0 or p_gt_max > 0:
                print(f"Parameter {name} has {p_lt_min:.2%} values below {finfo.min} and {p_gt_max:.2%} values above {finfo.max}")

            p_fp16 = p.half().float()
            p_diff = (p - p_fp16).abs()
            p_diff_max = p_diff.max().item()
            p_diff_mean = p_diff.mean().item()
            p_diff_rel = p_diff / p.abs().clamp(min=1e-6)
            p_diff_rel_mean = p_diff_rel.mean().item()
            p_diff_rel_max = p_diff_rel.max().item()
            print(f"Parameter {name} has relative max rounding error {100 * p_diff_rel_max:.2f}%")


@torch.no_grad()
def trace_and_save(checkpoint_path, traced_checkpoint_path, **kwargs):
    model = model_from_checkpoint(checkpoint_path, **kwargs)
    model = model.cuda().eval()

    check_model_can_be_cast_to_fp16(model)

    example_input = torch.randn(1, 1, 192, 128, 128).cuda()
    traced_model = torch.jit.trace(model, example_input)
    torch.jit.save(traced_model, str(traced_checkpoint_path))


def main(*checkpoints, num_classes=6, use_stride2=True, use_stride4=False, **kwargs):
    for checkpoint in checkpoints:
        checkpoint_path = Path(checkpoint)
        traced_checkpoint_path = checkpoint_path.with_suffix(".jit")

        trace_and_save(
            checkpoint_path,
            traced_checkpoint_path,
            num_classes=num_classes,
            use_stride2=use_stride2,
            use_stride4=use_stride4,
            **kwargs,
        )


if __name__ == "__main__":
    import fire

    fire.Fire(main)

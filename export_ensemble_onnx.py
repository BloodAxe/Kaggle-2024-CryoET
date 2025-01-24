from pathlib import Path

import onnx
import torch
from torch import nn

from cryoet.modelling.detection.dynunet import DynUNetForObjectDetection, DynUNetForObjectDetectionConfig
from cryoet.modelling.detection.litehrnet import HRNetv2ForObjectDetection, HRNetv2ForObjectDetectionConfig
from cryoet.modelling.detection.segresnet_object_detection_v2 import (
    SegResNetForObjectDetectionV2Config,
    SegResNetForObjectDetectionV2,
)


class Ensemble(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        all_scores = []
        all_offsets = []

        for model in self.models:
            (logits,), (offsets,) = model(x)
            all_scores.append(logits.sigmoid())
            all_offsets.append(offsets)

        scores = torch.mean(torch.stack(all_scores, dim=0), dim=0)
        offsets = torch.mean(torch.stack(all_offsets, dim=0), dim=0)
        return scores, offsets


def model_from_checkpoint(checkpoint_path: Path, **kwargs):
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    model_state_dict = checkpoint["state_dict"]
    model_state_dict = {k.replace("model.", ""): v for k, v in model_state_dict.items() if k.startswith("model.")}

    if "hrnet" in checkpoint_path.stem:
        config = HRNetv2ForObjectDetectionConfig(**kwargs)
        model = HRNetv2ForObjectDetection(config)
    elif "dynunet" in checkpoint_path.stem:
        config = DynUNetForObjectDetectionConfig(**kwargs)
        model = DynUNetForObjectDetection(config)
    elif "segresnetv2" in checkpoint_path.stem:
        config = SegResNetForObjectDetectionV2Config(**kwargs)
        model = SegResNetForObjectDetectionV2(config)
    else:
        raise ValueError(f"Unknown model type: {checkpoint_path.stem}")

    model.load_state_dict(model_state_dict, strict=True)
    return model


def main(*checkpoints, output_onnx: str, opset=15, **kwargs):
    models = [model_from_checkpoint(checkpoint, **kwargs) for checkpoint in checkpoints]
    ensemble = Ensemble(models).cuda().eval()

    output_onnx = Path(output_onnx)
    dummy_input = torch.randn(1, 1, 192, 128, 128, device="cuda")

    torch.onnx.export(
        model=ensemble,
        args=(dummy_input,),
        f=output_onnx,
        verbose=False,
        verify=True,
        opset_version=opset,
        input_names=["volume"],
        output_names=["scores", "offsets"],
    )

    torch.jit.save(
        torch.jit.trace(
            func=ensemble,
            example_inputs=(dummy_input,),
        ),
        f=str(output_onnx.with_suffix(".jit")),
    )

    print(f"Exported ensemble to {output_onnx}")
    print("Models in ensemble:")
    for checkpoint in checkpoints:
        print(checkpoint)

    try:
        import onnxsim

        simplified_model, success = onnxsim.simplify(model=onnx.load(output_onnx))
        if not success:
            print("Failed to simplify model")
        else:
            simplified_model_path = output_onnx.with_suffix(".simplified.onnx")
            onnx.save(simplified_model, simplified_model_path)
            print(f"Simplified model saved to {simplified_model_path}")

    except ImportError:
        print("onnxsim not found, skipping optimization")


# python export_ensemble_onnx.py --num_classes=6 --use_stride4=False --use_stride2=True --output_onnx=v4_segresnet_dynunet_ensemble.onnx runs/dynunet_fold_0_6x96x128x128_rc_ic_s2_re_0.05/250118_0912_adamw_torch_lr_3e-04_wd_0.0001_b1_0.95_b2_0.99_ema_0.995_10/dynunet_fold_0_6x96x128x128_rc_ic_s2_re_0.05/lightning_logs/version_0/checkpoints/250118_0912_dynunet_fold_0_6x96x128x128_rc_ic_s2_re_0.05_1484-score-0.8272-at-0.145-0.330-0.215-0.235-0.405.ckpt  runs/dynunet_fold_1_6x96x128x128_rc_ic_s2_re_0.05/250118_0953_adamw_torch_lr_3e-04_wd_0.0001_b1_0.95_b2_0.99_ema_0.995_10/dynunet_fold_1_6x96x128x128_rc_ic_s2_re_0.05/lightning_logs/version_0/checkpoints/250118_0953_dynunet_fold_1_6x96x128x128_rc_ic_s2_re_0.05_4982-score-0.8337-at-0.390-0.185-0.215-0.210-0.235.ckpt  runs/dynunet_fold_2_6x96x128x128_rc_ic_s2_re_0.05/250118_1112_adamw_torch_lr_3e-04_wd_0.0001_b1_0.95_b2_0.99_ema_0.995_10/dynunet_fold_2_6x96x128x128_rc_ic_s2_re_0.05/lightning_logs/version_0/checkpoints/250118_1112_dynunet_fold_2_6x96x128x128_rc_ic_s2_re_0.05_2650-score-0.7971-at-0.425-0.160-0.550-0.275-0.650.ckpt  runs/dynunet_fold_3_6x96x128x128_rc_ic_s2_re_0.05/250118_1208_adamw_torch_lr_3e-04_wd_0.0001_b1_0.95_b2_0.99_ema_0.995_10/dynunet_fold_3_6x96x128x128_rc_ic_s2_re_0.05/lightning_logs/version_0/checkpoints/250118_1208_dynunet_fold_3_6x96x128x128_rc_ic_s2_re_0.05_1590-score-0.8418-at-0.385-0.345-0.345-0.305-0.210.ckpt  runs/dynunet_fold_4_6x96x128x128_rc_ic_s2_re_0.05/250118_1253_adamw_torch_lr_3e-04_wd_0.0001_b1_0.95_b2_0.99_ema_0.995_10/dynunet_fold_4_6x96x128x128_rc_ic_s2_re_0.05/lightning_logs/version_0/checkpoints/250118_1253_dynunet_fold_4_6x96x128x128_rc_ic_s2_re_0.05_1060-score-0.8420-at-0.230-0.340-0.255-0.375-0.140.ckpt runs/segresnetv2_fold_0_6x96x128x128_rc_ic_s2/250117_2136_adamw_torch_lr_1e-04_wd_0.01_b1_0.95_b2_0.99/segresnetv2_fold_0_6x96x128x128_rc_ic_s2/lightning_logs/version_0/checkpoints/250117_2136_segresnetv2_fold_0_6x96x128x128_rc_ic_s2_2560-score-0.8457-at-0.265-0.290-0.195-0.150-0.550.ckpt  runs/segresnetv2_fold_1_6x96x128x128_rc_ic_s2/250117_2225_adamw_torch_lr_1e-04_wd_0.01_b1_0.95_b2_0.99/segresnetv2_fold_1_6x96x128x128_rc_ic_s2/lightning_logs/version_0/checkpoints/250117_2225_segresnetv2_fold_1_6x96x128x128_rc_ic_s2_2880-score-0.8366-at-0.345-0.110-0.185-0.135-0.305.ckpt  runs/segresnetv2_fold_2_6x96x128x128_rc_ic_s2/250117_1623_adamw_torch_lr_1e-04_wd_0.01_b1_0.95_b2_0.99/segresnetv2_fold_2_6x96x128x128_rc_ic_s2/lightning_logs/version_0/checkpoints/250117_1623_segresnetv2_fold_2_6x96x128x128_rc_ic_s2_2240-score-0.8046-at-0.355-0.280-0.395-0.340-0.185.ckpt  runs/segresnetv2_fold_3_6x96x128x128_rc_ic_s2/250118_0038_adamw_torch_lr_1e-04_wd_0.01_b1_0.95_b2_0.99/segresnetv2_fold_3_6x96x128x128_rc_ic_s2/lightning_logs/version_0/checkpoints/250118_0038_segresnetv2_fold_3_6x96x128x128_rc_ic_s2_2240-score-0.8398-at-0.230-0.345-0.405-0.360-0.255.ckpt  runs/segresnetv2_fold_4_6x96x128x128_rc_ic_s2/250118_0108_adamw_torch_lr_1e-04_wd_0.01_b1_0.95_b2_0.99/segresnetv2_fold_4_6x96x128x128_rc_ic_s2/lightning_logs/version_0/checkpoints/250118_0108_segresnetv2_fold_4_6x96x128x128_rc_ic_s2_2880-score-0.8437-at-0.165-0.350-0.245-0.235-0.270.ckpt
if __name__ == "__main__":
    import fire

    fire.Fire(main)

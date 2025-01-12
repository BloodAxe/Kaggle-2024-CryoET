import timm
import torch
import torch.nn as nn
from pytorch_toolbelt.utils import count_parameters
from pytorch_toolbelt.modules import get_activation_block
from torch.nn import Conv3d, BatchNorm3d, Upsample, ConvTranspose2d
from transformers import PretrainedConfig

from cryoet.modelling.detection.detection_head import ObjectDetectionHead, ObjectDetectionOutput
from cryoet.modelling.detection.functional import object_detection_loss
import einops

from timm.models.hrnet import hrnet_w18_small_v2, hrnet_w32


class PixelShuffle3d(nn.Module):
    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        r = self.upscale_factor
        b, c, d, h, w = x.shape

        out = einops.rearrange(x, "b (c r1 r2 r3) d h w -> b c (d r1) (h r2) (w r3)", r1=r, r2=r, r3=r)
        return out


def convert_2d_to_3d(model: nn.Module) -> nn.Module:
    """
    Recursively convert all Conv2d layers in `model` to Conv3d,
    and all BatchNorm2d layers to BatchNorm3d.
    Replicates the existing Conv2d weights along the 3rd dimension (depth=1 by default).
    """
    for name, module in model.named_children():

        # If we find a Conv2d, replace it with a Conv3d.
        if isinstance(module, nn.Conv2d):
            # --------------------------------------------
            # 1) Check that the 2D kernel is square
            # --------------------------------------------
            if module.kernel_size[0] != module.kernel_size[1]:
                raise ValueError(
                    f"Non-square kernel detected: {module.kernel_size}. " "This example only handles square kernels (k, k)."
                )
            k = module.kernel_size[0]

            # --------------------------------------------
            # 2) Build a new Conv3d with kernel_size = (k, k, k)
            #    using the same hyperparameters as best we can
            # --------------------------------------------
            new_conv = nn.Conv3d(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel_size=(k, k, k),
                # For stride, padding, and dilation, we replicate
                # the 2D values in each dimension:
                stride=(module.stride[0], module.stride[0], module.stride[1]),
                padding=(module.padding[0], module.padding[0], module.padding[1]),
                dilation=(module.dilation[0], module.dilation[0], module.dilation[1]),
                groups=module.groups,
                bias=(module.bias is not None),
            )

            # --------------------------------------------
            # 3) Copy and replicate the 2D weights -> 3D
            # old_weight shape: (out_c, in_c, k, k)
            # new_weight shape: (out_c, in_c, k, k, k)
            # --------------------------------------------
            with torch.no_grad():
                old_weight = module.weight  # shape: (out_c, in_c, k, k)
                # Expand along a new depth dimension
                old_weight_3d = old_weight.unsqueeze(2)  # (out_c, in_c, 1, k, k)
                old_weight_3d = old_weight_3d.repeat(1, 1, k, 1, 1)  # (out_c, in_c, k, k, k)
                new_conv.weight.copy_(old_weight_3d)

                if module.bias is not None:
                    new_conv.bias.copy_(module.bias)

            # Replace the old Conv2d with our new Conv3d
            setattr(model, name, new_conv)

        # If we find a BatchNorm2d, replace it with a BatchNorm3d.
        elif isinstance(module, nn.BatchNorm2d):
            new_bn = nn.InstanceNorm3d(
                num_features=module.num_features,
                eps=module.eps,
                momentum=module.momentum,
                affine=module.affine,
                # track_running_stats=module.track_running_stats,
            )

            # Copy running statistics and affine parameters
            # with torch.no_grad():
            #     if module.affine:
            #         new_bn.weight.copy_(module.weight)
            #         new_bn.bias.copy_(module.bias)
            #     new_bn.running_mean.copy_(module.running_mean)
            #     new_bn.running_var.copy_(module.running_var)
            #
            # Replace the BatchNorm2d with BatchNorm3d
            setattr(model, name, new_bn)

        elif isinstance(module, nn.ReLU):
            # Replace with SILU
            setattr(model, name, nn.SiLU(inplace=True))
        else:
            # Recursively convert children
            convert_2d_to_3d(module)

    return model


class HRNetv2ForObjectDetection(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = timm.create_model(
            "hrnet_w48_ssld.paddle_in1k",
            in_chans=1,
            pretrained=True,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),
        )
        self.backbone = convert_2d_to_3d(backbone)

        self.up_32 = nn.Sequential(
            nn.InstanceNorm3d(1024),
            nn.Conv3d(1024, 512, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="trilinear"),
        )
        self.up_16 = nn.Sequential(
            nn.InstanceNorm3d(512),
            nn.Conv3d(512, 256, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="trilinear"),
        )
        self.up_8 = nn.Sequential(
            nn.InstanceNorm3d(256),
            nn.Conv3d(256, 128, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="trilinear"),
        )
        self.up_4 = nn.Sequential(
            nn.InstanceNorm3d(128),
            nn.Conv3d(128, 64, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="trilinear"),
        )
        self.head = ObjectDetectionHead(
            in_channels=64, num_classes=5, intermediate_channels=64, offset_intermediate_channels=8, stride=2
        )

    def forward(self, volume, labels=None, **loss_kwargs):
        fm2, fm4, fm8, fm16, fm32 = self.backbone(volume)

        x = fm2 + self.up_4(fm4 + self.up_8(fm8 + self.up_16(fm16 + self.up_32(fm32))))
        logits, offsets = self.head(x)

        logits = [logits]
        offsets = [offsets]
        strides = [self.head.stride]

        if torch.jit.is_tracing():
            return logits, offsets

        loss = None
        loss_dict = None
        if labels is not None:
            loss, loss_dict = object_detection_loss(logits, offsets, strides, labels, **loss_kwargs)

        return ObjectDetectionOutput(logits=logits, offsets=offsets, strides=strides, loss=loss, loss_dict=loss_dict)


if __name__ == "__main__":
    model = HRNetv2ForObjectDetection()
    print(count_parameters(model, human_friendly=True))
    x = torch.randn(1, 1, 96, 256, 256)
    with torch.no_grad():
        y = model(x)
        print(y)

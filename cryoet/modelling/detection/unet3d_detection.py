from typing import Tuple, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .detection_head import ObjectDetectionHead

from transformers import PretrainedConfig


class RepVGGBlock3D(nn.Module):
    """
    A 3D RepVGG-like block with:
        - One 3x3x3 conv
        - One 1x1x1 conv
        - Optional residual path
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, use_residual: bool = True, bias=False):
        super().__init__()
        self.use_residual = use_residual

        if self.use_residual:
            self.alpha = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        else:
            self.alpha = 1

        # 3x3x3 convolution
        self.conv3x3 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias)

        # 1x1x1 convolution
        self.conv1x1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=bias)

        self.bn = nn.InstanceNorm3d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_3x3 = self.conv3x3(x)
        out_1x1 = self.conv1x1(x)

        # Sum of 3x3 and 1x1 branches
        out = out_3x3 + out_1x1

        if self.use_residual:
            out = out + x * self.alpha

        out = self.bn(out)
        out = self.act(out)
        return out


class Downsample3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Conv3d with stride=2.
        self.down = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.down(x)


class Upsample3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Transposed convolution with stride=2.
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x, output_size=None):
        return self.up(x, output_size)


class Unet3DEncoderStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks):
        super().__init__()
        self.down = Downsample3D(in_channels, out_channels)
        self.blocks = nn.Sequential(
            *[RepVGGBlock3D(in_channels=out_channels, out_channels=out_channels, use_residual=True) for _ in range(num_blocks)]
        )

    def forward(self, x):
        x = self.down(x)
        x = self.blocks(x)
        return x


class Unet3DDecoderStage(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels: int, num_blocks):
        super().__init__()
        self.up = Upsample3D(in_channels, out_channels)

        blocks = []
        in_channels = out_channels + skip_channels
        for i in range(num_blocks):
            block = RepVGGBlock3D(in_channels=in_channels, out_channels=out_channels, use_residual=i > 0)
            in_channels = out_channels
            blocks.append(block)

        self.blocks = nn.Sequential(*blocks)
        self.out_channels = out_channels

    def forward(self, x, skip):
        x_up = self.up(x, output_size=skip.shape[2:])
        y = torch.cat([x_up, skip], dim=1)
        y = self.blocks(y)
        return y


class UNet3DBackbone(nn.Module):
    """
    3D U-Net with configurable RepVGG blocks and the ability to decode
    only up to a certain stride rather than full resolution.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output channels (segmentation classes, etc.).
        encoder_channels (List[int]): Number of channels at each encoder stage.
        num_blocks_per_stage (List[int]): How many RepVGG blocks to use at each stage.
        use_residual (bool): Enable/disable residual path in RepVGGBlock3D.
        decode_final_stride (int): Output stride relative to input. Must be a power of 2.
                                   e.g. decode_final_stride=2 => output is 1/2 the original resolution.
    """

    def __init__(
        self,
        in_channels: int,
        encoder_channels: Iterable[int],
        num_blocks_per_stage: Iterable[int],
        num_blocks_per_decoder_stage: Iterable[int],
    ):
        super().__init__()

        self.in_channels = in_channels
        self.encoder_channels = tuple(encoder_channels)
        self.num_down_stages = len(self.encoder_channels)  # total encoder stages
        self.num_blocks_per_stage = tuple(num_blocks_per_stage)
        self.num_blocks_per_decoder_stage = tuple(num_blocks_per_decoder_stage)

        # ------------------------------------------------------------------
        # Build the Encoder
        # ------------------------------------------------------------------
        self.encoders = nn.ModuleList()

        prev_ch = in_channels
        for stage_idx, out_ch in enumerate(encoder_channels):
            stage = Unet3DEncoderStage(prev_ch, out_ch, self.num_blocks_per_stage[stage_idx])
            prev_ch = out_ch
            self.encoders.append(stage)

        # ------------------------------------------------------------------
        # Build the Decoder
        # ------------------------------------------------------------------
        self.num_decoder_stages = len(self.num_blocks_per_decoder_stage)

        # We'll build that many decoder modules, reversing the encoder_channels:
        decode_channels = self.encoder_channels[::-1]  # e.g. [256, 128, 64, 32]

        self.decoders = nn.ModuleList()

        # For each decoder stage i, we upsample from decode_channels[i] -> decode_channels[i+1],
        # then do a RepVGG block. We'll handle skip connections in forward().
        # We only build "num_decoder_stages" of them.

        decoder_output_channels = []

        for i in range(self.num_decoder_stages):
            up_in_ch = decode_channels[i]
            up_out_ch = decode_channels[i + 1]

            block = Unet3DDecoderStage(
                up_in_ch, up_out_ch, skip_channels=up_out_ch, num_blocks=self.num_blocks_per_decoder_stage[i]
            )
            decoder_output_channels.append(up_out_ch)
            self.decoders.append(block)

        self.decoder_output_channels = decoder_output_channels
        self.out_channels = self.decoder_output_channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        out = x

        # ----------------
        # Encoder forward
        # ----------------
        for stage in self.encoders:
            out = stage(out)
            skips.append(out)

        # -----------
        # Decoder forward
        # -----------
        skips = skips[::-1]
        out = skips[0]

        for i, dec_stage in enumerate(self.decoders):
            # Concatenate skip from shallower level
            skip = skips[i + 1]  # i+1 is "one level shallower" in reversed list
            out = dec_stage(out, skip)

        return out


class UNet3DForObjectDetectionConfig(PretrainedConfig):
    def __init__(
        self,
        in_channels=1,
        num_classes=5,
        encoder_channels=(16, 24, 32, 64, 96),
        num_blocks_per_stage=(1, 1, 2, 2, 2),
        num_blocks_per_decoder_stage=(1, 2, 2, 2),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.encoder_channels = encoder_channels
        self.num_blocks_per_stage = num_blocks_per_stage
        self.num_blocks_per_decoder_stage = num_blocks_per_decoder_stage


class UNet3DForObjectDetection(nn.Module):
    def __init__(self, config: UNet3DForObjectDetectionConfig):
        super().__init__()

        self.backbone = UNet3DBackbone(
            in_channels=config.in_channels,
            encoder_channels=config.encoder_channels,
            num_blocks_per_stage=config.num_blocks_per_stage,
            num_blocks_per_decoder_stage=config.num_blocks_per_decoder_stage,
        )

        self.head = ObjectDetectionHead(
            in_channels=self.backbone.out_channels, num_classes=config.num_classes, intermediate_channels=32, stride=1
        )

    def forward(self, volume, labels=None, **loss_kwargs):
        features = self.backbone(volume)
        return self.head(features, labels, **loss_kwargs)

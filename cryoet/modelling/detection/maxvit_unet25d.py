import einops
import torch
from torch import nn
from transformers import PretrainedConfig
from pytorch_toolbelt.modules.encoders.timm.maxvit import MaxVitEncoder
from torch.utils import checkpoint as checkpoint_utils

from cryoet.modelling.detection.detection_head import ObjectDetectionHead, ObjectDetectionOutput
from cryoet.modelling.detection.functional import object_detection_loss


class MaxVitUnet25dConfig(PretrainedConfig):
    def __init__(
        self,
        model_name="maxxvitv2_nano_rw_256.sw_in1k",
        num_classes: int = 5,
        use_checkpointing: bool = True,
        img_size=256,
        drop_rate=0.05,
        drop_path_rate=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.use_checkpointing = use_checkpointing
        self.model_name = model_name
        self.img_size = img_size
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate


class Upsample25D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Transposed convolution with stride=2.
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(1, 2, 2), stride=(1, 2, 2))

    def forward(self, x, output_size=None):
        return self.up(x, output_size)


class BilinearUpsample25D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Transposed convolution with stride=2.
        self.project = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x, output_size=None):
        return torch.nn.functional.interpolate(self.project(x), size=output_size, mode="trilinear", align_corners=False)


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


class Unet25DDecoderStage(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels: int, num_blocks):
        super().__init__()
        # self.up = Upsample25D(in_channels, out_channels)
        self.up = BilinearUpsample25D(in_channels, out_channels)

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


class MaxVitUnet25d(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_checkpointing = config.use_checkpointing
        self.encoder = MaxVitEncoder(
            model_name=config.model_name,
            pretrained=config.model_name is not None,
            img_size=config.img_size,
            drop_rate=config.drop_rate,
            drop_path_rate=config.drop_path_rate,
        ).change_input_channels(1)

        self.output_spec = self.encoder.get_output_spec()

        self.decoder16 = Unet25DDecoderStage(
            in_channels=self.output_spec.channels[-1], out_channels=128, skip_channels=self.output_spec.channels[-2], num_blocks=2
        )

        self.decoder8 = Unet25DDecoderStage(
            in_channels=self.decoder16.out_channels, out_channels=128, skip_channels=self.output_spec.channels[-3], num_blocks=2
        )

        self.decoder4 = Unet25DDecoderStage(
            in_channels=self.decoder8.out_channels, out_channels=128, skip_channels=self.output_spec.channels[-4], num_blocks=2
        )

        self.decoder2 = Unet25DDecoderStage(
            in_channels=self.decoder4.out_channels, out_channels=64, skip_channels=self.output_spec.channels[-5], num_blocks=2
        )

        self.head4 = ObjectDetectionHead(
            in_channels=self.decoder4.out_channels,
            num_classes=config.num_classes,
            stride=4,
            intermediate_channels=64,
            offset_intermediate_channels=32,
        )

        self.head2 = ObjectDetectionHead(
            in_channels=self.decoder2.out_channels,
            num_classes=config.num_classes,
            stride=2,
            intermediate_channels=48,
            offset_intermediate_channels=16,
        )

    def forward(self, volume, labels=None, **loss_kwargs):
        B, C, D, H, W = volume.size()
        # print(volume.size())
        volume_reshaped = einops.rearrange(volume, "B C D H W -> (B D) C H W")

        if self.use_checkpointing and not torch.jit.is_tracing():
            feature_maps = checkpoint_utils.checkpoint(
                self.encoder, volume_reshaped, preserve_rng_state=True, use_reentrant=False
            )
        else:
            feature_maps = self.encoder(volume_reshaped)

        # Reshape back to B D C H W
        feature_maps = [einops.rearrange(fm, "(B D) C H W -> B C D H W", B=B, D=D) for fm in feature_maps]

        # Run decoder
        fm2, fm4, fm8, fm16, fm32 = feature_maps
        out16 = self.decoder16(fm32, fm16)
        out8 = self.decoder8(out16, fm8)
        out4 = self.decoder4(out8, fm4)
        out2 = self.decoder2(out4, fm2)

        output4 = self.head4(out4)
        output2 = self.head2(out2)

        if torch.jit.is_tracing():
            logits4, offsets4 = output4
            logits2, offsets2 = output2
            return (logits4, logits2), (offsets4, offsets2)

        logits = [
            output4.logits,
            output2.logits,
        ]
        offsets = [
            output4.offsets,
            output2.offsets,
        ]
        strides = [
            self.head4.stride,
            self.head2.stride,
        ]

        loss = None
        loss_dict = None
        if labels is not None:
            loss, loss_dict = object_detection_loss(logits, offsets, strides, labels, **loss_kwargs)

        return ObjectDetectionOutput(logits=logits, offsets=offsets, strides=strides, loss=loss, loss_dict=loss_dict)

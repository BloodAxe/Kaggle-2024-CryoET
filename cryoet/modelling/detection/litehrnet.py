import timm
import torch
import torch.nn as nn
from pytorch_toolbelt.utils import count_parameters
from transformers import PretrainedConfig

from cryoet.modelling.detection.detection_head import ObjectDetectionHead, ObjectDetectionOutput
from cryoet.modelling.detection.functional import object_detection_loss, convert_2d_to_3d


class PixelShuffle3d(nn.Module):
    def __init__(self, in_channels, upscale_factor):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels // (upscale_factor**3)
        self.upscale_factor = upscale_factor

    def forward(self, x):
        B, CRRR, D, H, W = x.shape
        R = self.upscale_factor  # from your code, or compute it if known
        C = self.out_channels

        # 1) Split the (C*R*R*R) dimension into (C, R, R, R)
        x_reshaped = x.reshape(B, C, R, R, R, D, H, W)

        # 2) Permute to move the dimensions into the order (B, C, D, R, H, R, W, R)
        x_permuted = x_reshaped.permute(0, 1, 5, 2, 6, 3, 7, 4)
        # The indexing here is:
        #   0 -> B
        #   1 -> C
        #   5 -> D
        #   2 -> R (paired with D to form D*R)
        #   6 -> H
        #   3 -> R (paired with H to form H*R)
        #   7 -> W
        #   4 -> R (paired with W to form W*R)

        # 3) Final reshape to [B, C, D*R, H*R, W*R]
        out = x_permuted.reshape(B, C, D * R, H * R, W * R)
        return out


class HRNetv2ForObjectDetectionConfig(PretrainedConfig):
    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes


class HRNetv2ForObjectDetection(nn.Module):
    def __init__(self, config: HRNetv2ForObjectDetectionConfig):
        super().__init__()
        self.config = config
        backbone = timm.create_model(
            "hrnet_w18_small_v2.ms_in1k",
            in_chans=1,
            pretrained=True,
            features_only=True,
            out_indices=(1, 2, 3, 4),
            use_incre_features=False,
        )
        self.backbone = convert_2d_to_3d(backbone)

        self.up_32 = nn.Sequential(
            nn.BatchNorm3d(1024),
            nn.Conv3d(1024, 512, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="trilinear"),
        )
        self.up_16 = nn.Sequential(
            nn.Conv3d(512 + 512, 256, kernel_size=1),
            nn.BatchNorm3d(256),
            nn.SiLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.SiLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="trilinear"),
        )
        self.up_8 = nn.Sequential(
            nn.BatchNorm3d(512),
            nn.Conv3d(512, 128, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.SiLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="trilinear"),
        )
        self.up_4 = nn.Sequential(
            nn.BatchNorm3d(256),
            nn.Conv3d(256, 128, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.SiLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="trilinear"),
        )
        self.head = ObjectDetectionHead(
            in_channels=64, num_classes=config.num_classes, intermediate_channels=64, offset_intermediate_channels=8, stride=2
        )

    def decoder_forward(self, feature_maps):
        fm4, fm8, fm16, fm32 = feature_maps

        x = torch.cat([fm16, self.up_32(fm32)], dim=1)
        x = torch.cat([fm8, self.up_16(x)], dim=1)
        x = torch.cat([fm4, self.up_8(x)], dim=1)
        x = self.up_4(x)
        return x

    def forward(self, volume, labels=None, **loss_kwargs):
        feature_maps = self.backbone(volume)
        x = self.decoder_forward(feature_maps)

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
    config = HRNetv2ForObjectDetectionConfig()
    model = HRNetv2ForObjectDetection(config)
    print(count_parameters(model, human_friendly=True))
    x = torch.randn(1, 1, 96, 256, 256)
    with torch.no_grad():
        y = model(x)
        print(y)

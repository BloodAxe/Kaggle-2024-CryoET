import timm
import torch
import torch.nn as nn
from pytorch_toolbelt.utils import count_parameters
from transformers import PretrainedConfig

from cryoet.modelling.detection.detection_head import ObjectDetectionHead, ObjectDetectionOutput
from cryoet.modelling.detection.functional import object_detection_loss, convert_2d_to_3d


class ConvNextForObjectDetectionConfig(PretrainedConfig):
    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes


class ConvNextForObjectDetection(nn.Module):
    def __init__(self, config: ConvNextForObjectDetectionConfig):
        super().__init__()
        self.config = config
        backbone = timm.create_model(
            "convnextv2_nano.fcmae_ft_in22k_in1k",
            in_chans=1,
            pretrained=True,
            features_only=True,
            # out_indices=(1, 2, 3, 4),
        )
        self.backbone = convert_2d_to_3d(backbone)

        self.up_32 = nn.Sequential(
            nn.BatchNorm3d(640),
            nn.Conv3d(640, 320, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="trilinear"),
        )
        self.up_16 = nn.Sequential(
            nn.BatchNorm3d(320),
            nn.Conv3d(320, 160, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="trilinear"),
        )
        self.up_8 = nn.Sequential(
            nn.BatchNorm3d(160),
            nn.Conv3d(160, 80, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="trilinear"),
        )
        self.up_4 = nn.Sequential(
            nn.BatchNorm3d(80),
            nn.Conv3d(80, 80, kernel_size=1),
            nn.SiLU(inplace=True),
            # PixelShuffle3d(64 * 8, 2),
            nn.Upsample(scale_factor=2, mode="trilinear"),
        )
        self.head = ObjectDetectionHead(
            in_channels=80, num_classes=config.num_classes, intermediate_channels=64, offset_intermediate_channels=8, stride=2
        )

    def forward(self, volume, labels=None, **loss_kwargs):
        feature_maps = self.backbone(volume)
        # for f in feature_maps:
        #     print(f.size())
        fm4, fm8, fm16, fm32 = feature_maps

        x = self.up_4(fm4 + self.up_8(fm8 + self.up_16(fm16 + self.up_32(fm32))))
        logits, offsets = self.head(x)

        logits = [logits]
        offsets = [offsets]
        strides = [self.head.stride]

        if torch.jit.is_tracing() or torch.jit.is_scripting():
            return logits, offsets

        loss = None
        loss_dict = None
        if labels is not None:
            loss, loss_dict = object_detection_loss(logits, offsets, strides, labels, **loss_kwargs)

        return ObjectDetectionOutput(logits=logits, offsets=offsets, strides=strides, loss=loss, loss_dict=loss_dict)


if __name__ == "__main__":
    config = ConvNextForObjectDetectionConfig()
    model = ConvNextForObjectDetection(config)

    print(count_parameters(model, human_friendly=True))
    x = torch.randn(1, 1, 96, 256, 256)
    with torch.no_grad():
        y = model(x)
        print(y)

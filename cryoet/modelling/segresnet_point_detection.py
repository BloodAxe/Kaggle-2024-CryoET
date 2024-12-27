from monai.networks.nets import SegResNet
from torch import nn

from .point_detection_head import PointDetectionHead

from transformers import PretrainedConfig


class SegResNetForPointDetectionConfig(PretrainedConfig):
    def __init__(
        self,
        spatial_dims=3,
        img_size=96,
        in_channels=1,
        out_channels=14,
        feature_size=48,
        num_classes=5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.spatial_dims = spatial_dims
        self.img_size = img_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_size = feature_size
        self.num_classes = num_classes


class SegResNetForPointDetection(nn.Module):
    def __init__(self, config: SegResNetForPointDetectionConfig):
        super().__init__()

        self.backbone = SegResNet(
            spatial_dims=config.spatial_dims,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
        )

        self.head = PointDetectionHead(in_channels=config.out_channels, num_classes=config.num_classes)

    def forward(self, volume, labels=None, **loss_kwargs):
        features = self.backbone(volume)
        return self.head(features, labels, **loss_kwargs)

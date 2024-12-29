from monai.networks.nets import SegResNet
from torch import nn

from .point_detection_head import PointDetectionHead

from transformers import PretrainedConfig


class SegResNetForPointDetectionConfig(PretrainedConfig):
    def __init__(
        self,
        spatial_dims=3,
        in_channels=1,
        out_channels=105,
        init_filters=32,
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        dropout_prob=0.2,
        num_classes=5,
        use_qfl_loss: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_filters = init_filters
        self.blocks_down = blocks_down
        self.blocks_up = blocks_up
        self.dropout_prob = dropout_prob
        self.num_classes = num_classes
        self.use_qfl_loss = use_qfl_loss


class SegResNetForPointDetection(nn.Module):
    def __init__(self, config: SegResNetForPointDetectionConfig):
        super().__init__()

        self.backbone = SegResNet(
            spatial_dims=config.spatial_dims,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            init_filters=config.init_filters,
            blocks_down=config.blocks_down,
            blocks_up=config.blocks_up,
            dropout_prob=config.dropout_prob,
        )

        self.head = PointDetectionHead(
            in_channels=config.out_channels, num_classes=config.num_classes, use_qfl_loss=config.use_qfl_loss
        )

    def forward(self, volume, labels=None, **loss_kwargs):
        features = self.backbone(volume)
        return self.head(features, labels, **loss_kwargs)

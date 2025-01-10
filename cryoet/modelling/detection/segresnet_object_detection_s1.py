import torch
from torch import nn
from transformers import PretrainedConfig

from .detection_head import ObjectDetectionHead, ObjectDetectionOutput
from .functional import object_detection_loss
from .segresnet_object_detection_v2 import SegResNetBackbone


class SegResNetForObjectDetectionS1Config(PretrainedConfig):
    def __init__(
        self,
        spatial_dims=3,
        in_channels=1,
        out_channels=105,
        init_filters=32,
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        dropout_prob=0.1,
        num_classes=5,
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


class SegResNetForObjectDetectionS1(nn.Module):
    def __init__(self, config: SegResNetForObjectDetectionS1Config):
        super().__init__()
        self.config = config
        self.backbone = SegResNetBackbone(
            spatial_dims=config.spatial_dims,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            init_filters=config.init_filters,
            blocks_down=config.blocks_down,
            blocks_up=config.blocks_up,
        )
        self.drop2d = nn.Dropout3d(p=config.dropout_prob)
        self.head = ObjectDetectionHead(
            in_channels=32, num_classes=config.num_classes, intermediate_channels=32, offset_intermediate_channels=16, stride=1
        )

    def forward(self, volume, labels=None, **loss_kwargs):
        _, feature_maps = self.backbone(volume)
        fm1 = feature_maps[-1]

        logits, offsets = self.head(self.drop2d(fm1))
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

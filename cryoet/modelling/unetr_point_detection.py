import dataclasses
from typing import Optional

import torch
from monai.networks.nets import SwinUNETR
from torch import nn, Tensor
import torch.nn.functional as F

from cryoet.modelling.configuration import SwinUNETRForPointDetectionConfig


@dataclasses.dataclass
class PointDetectionOutput:
    logits: Tensor
    loss: Optional[Tensor]


def point_detection_loss(logits, labels, num_items_in_batch=None, alpha=2, beta=4, eps=1e-6, **kwargs):
    """
    Compute centernet loss for object detection

    :param logits: Predicted headmap logits BxCxDxHxW
    :param labels: Target heatmap BxCxDxHxW
    :return: Single scalar loss
    """
    pos_mask = labels.eq(1)
    neg_mask = ~pos_mask

    pt = logits.sigmoid().clamp(eps, 1 - eps)

    pos_loss: Tensor = -torch.pow(1 - pt, alpha) * F.logsigmoid(logits) * pos_mask
    neg_loss: Tensor = -torch.pow(1.0 - labels, beta) * torch.pow(pt, alpha) * F.logsigmoid(-logits) * neg_mask

    loss = (neg_loss + pos_loss).sum()
    if num_items_in_batch is None:
        num_items_in_batch = pos_mask.sum().item()

    num_items_in_batch = max(num_items_in_batch, 1)
    return loss.div_(num_items_in_batch)


class PointDetectionHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, num_classes, kernel_size=3, padding=1)

        torch.nn.init.zeros_(self.conv.weight)
        torch.nn.init.constant_(self.conv.bias, -3)

    def forward(self, features, labels=None, **loss_kwargs):
        logits = self.conv(features)

        loss = None
        if labels is not None:
            loss = point_detection_loss(logits.float(), labels.float(), **loss_kwargs)

        return PointDetectionOutput(logits=logits, loss=loss)


class SwinUNETRForPointDetection(nn.Module):
    def __init__(self, config: SwinUNETRForPointDetectionConfig):
        super().__init__()

        self.backbone = SwinUNETR(
            spatial_dims=config.spatial_dims,
            img_size=config.img_size,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            feature_size=config.feature_size,
            use_checkpoint=True,
        )

        self.head = PointDetectionHead(in_channels=config.out_channels, num_classes=config.num_classes)

    def forward(self, volume, labels=None, **loss_kwargs):
        features = self.backbone(volume)
        return self.head(features, labels, **loss_kwargs)

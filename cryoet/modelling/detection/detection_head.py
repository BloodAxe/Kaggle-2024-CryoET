import torch
from torch import nn, Tensor
import dataclasses
from typing import Optional, Dict, List

from .functional import anchors_for_offsets_feature_map, object_detection_loss


@dataclasses.dataclass
class ObjectDetectionOutput:
    logits: List[Tensor]
    offsets: List[Tensor]
    strides: List[int]

    loss: Optional[Tensor]
    loss_dict: Optional[Dict]


class ObjectDetectionHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, stride: int, intermediate_channels: int = 64):
        super().__init__()
        self.cls_stem = nn.Sequential(
            nn.Conv3d(in_channels, intermediate_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.InstanceNorm3d(intermediate_channels),
            nn.Conv3d(intermediate_channels, intermediate_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.InstanceNorm3d(intermediate_channels),
        )

        self.offset_stem = nn.Sequential(
            nn.Conv3d(in_channels, intermediate_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.InstanceNorm3d(intermediate_channels),
            nn.Conv3d(intermediate_channels, intermediate_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.InstanceNorm3d(intermediate_channels),
        )

        self.cls_head = nn.Conv3d(intermediate_channels, num_classes, kernel_size=1, padding=0)
        self.offset_head = nn.Conv3d(intermediate_channels, 3, kernel_size=1, padding=0)

        self.stride = stride

        torch.nn.init.zeros_(self.offset_head.weight)
        torch.nn.init.constant_(self.offset_head.bias, 0)

        torch.nn.init.zeros_(self.cls_head.weight)
        torch.nn.init.constant_(self.cls_head.bias, -4)

    def forward(self, features, labels=None, **loss_kwargs):
        logits = self.cls_head(self.cls_stem(features))
        offsets = self.offset_head(self.offset_stem(features)).tanh() * self.stride

        if torch.jit.is_tracing():
            return logits, offsets

        anchors = anchors_for_offsets_feature_map(offsets, self.stride)

        loss = None
        loss_dict = None
        if labels is not None:
            loss, loss_dict = object_detection_loss(logits.float(), offsets.float(), anchors.float(), labels, **loss_kwargs)

        return ObjectDetectionOutput(logits=[logits], offsets=[offsets], strides=[self.stride], loss=loss, loss_dict=loss_dict)

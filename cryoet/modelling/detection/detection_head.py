import dataclasses
from typing import Optional, Dict, List

import torch
from torch import nn, Tensor


@dataclasses.dataclass
class ObjectDetectionOutput:
    logits: List[Tensor]
    offsets: List[Tensor]
    strides: List[int]

    loss: Optional[Tensor]
    loss_dict: Optional[Dict]


class ObjectDetectionHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        stride: int,
        intermediate_channels: int = 64,
        offset_intermediate_channels: int = 32,
        use_offset_head: bool = True,
    ):
        super().__init__()
        self.use_offset_head = use_offset_head
        self.cls_stem = nn.Sequential(
            nn.Conv3d(in_channels, intermediate_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.InstanceNorm3d(intermediate_channels),
            nn.Conv3d(intermediate_channels, intermediate_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.InstanceNorm3d(intermediate_channels),
        )

        self.cls_head = nn.Conv3d(intermediate_channels, num_classes, kernel_size=1, padding=0)

        if use_offset_head:
            self.offset_stem = nn.Sequential(
                nn.Conv3d(in_channels, offset_intermediate_channels, kernel_size=3, padding=1),
                nn.SiLU(inplace=True),
                nn.InstanceNorm3d(offset_intermediate_channels),
                nn.Conv3d(offset_intermediate_channels, offset_intermediate_channels, kernel_size=3, padding=1),
                nn.SiLU(inplace=True),
                nn.InstanceNorm3d(offset_intermediate_channels),
            )

            self.offset_head = nn.Conv3d(offset_intermediate_channels, 3, kernel_size=1, padding=0)
            torch.nn.init.zeros_(self.offset_head.weight)
            torch.nn.init.constant_(self.offset_head.bias, 0)

        self.stride = stride

        torch.nn.init.zeros_(self.cls_head.weight)
        torch.nn.init.constant_(self.cls_head.bias, -4)

    def forward(self, features):
        logits = self.cls_head(self.cls_stem(features))
        if self.use_offset_head:
            offsets = self.offset_head(self.offset_stem(features)).tanh() * self.stride
        else:
            # Dummy offsets
            offsets = torch.zeros(
                logits.size(0), 3, logits.size(2), logits.size(3), logits.size(4), device=logits.device, dtype=logits.dtype
            )

        return logits, offsets

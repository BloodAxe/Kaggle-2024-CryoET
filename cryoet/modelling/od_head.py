import dataclasses
from typing import Optional

import einops
import torch
from torch import nn, Tensor

from .task_aligned_assigner import TaskAlignedAssigner


@dataclasses.dataclass
class ObjectDetectionOutput:
    logits: Tensor
    offsets: Tensor
    anchors: Tensor

    loss: Optional[Tensor]


def decode_detections(logits, offsets, anchors):
    """
    Decode detections from logits and offsets
    :param logits: Predicted logits B C D H W
    :param offsets: Predicted offsets B 3 D H W
    :param anchors: Stride of the network

    :return: Tuple of probas and centers:
             probas - B N C
             centers - B N 3

    """
    centers = anchors + offsets

    logits = einops.rearrange(logits, "B C D H W -> B (D H W) C")
    centers = einops.rearrange(centers, "B C D H W -> B (D H W) C")
    anchors = einops.rearrange(anchors, "B C D H W -> B (D H W) C")
    return logits, centers, anchors


def varifocal_loss(pred_logits: Tensor, gt_score: Tensor, label: Tensor, alpha=0.75, gamma=2.0) -> Tensor:
    pred_score = pred_logits.sigmoid()
    weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label
    loss = weight * torch.nn.functional.binary_cross_entropy_with_logits(pred_logits, gt_score, reduction="none")
    return loss.sum()


def iou_loss(pred_centers, assigned_centers, assigned_scores, assigned_sigmas, mask_positive):
    d = ((pred_centers - assigned_centers) ** 2).sum(dim=-1, keepdim=False)  # [B, L]

    weight = assigned_scores.sum(-1)

    e: Tensor = d / (2 * assigned_sigmas**2)
    iou = torch.exp(-e)  # [B, M1, M2]
    loss = (1 - iou) * weight

    return torch.masked_fill(loss, ~mask_positive, 0).sum()


def object_detection_loss(logits, offsets, anchors, labels, num_items_in_batch=None, alpha=2, eps=1e-6, **kwargs):
    """
    Compute the detection loss adapted for 3D data
    It uses keypoint-like IOU loss (negative exponent of mse) to assign the objectness score to the center of the object

    :param logits:  Predicted headmap logits BxCxDxHxW
    :param offsets: Predicted offsets Bx3xDxHxW
    :param anchors: Anchor points Bx3xDxHxW
    :param labels:  Target labels encoded as BxNx5 where N is the number of objects and 5 is x,y,z,class,sigma
    :return:        Single scalar loss
    """

    # 1) Decode predictions
    pred_logits, pred_centers, anchor_points = decode_detections(logits, offsets, anchors)
    # shapes:
    # pred_logits:  [B, L, C]
    # pred_centers: [B, L, 3]
    batch_size, num_anchors, num_classes = pred_logits.size()

    # 2) Extract GT: [B, 5, N] => [B, 3, N], [B, N], [B, N]
    #    labels = (x, y, z, class, sigma)
    true_centers = labels[:, :, :3]  # [B, n, 3]
    true_labels = labels[:, :, 3:4]  # [B, n, 1]
    true_sigmas = labels[:, :, 4:5]  # [B, n, 1]

    # 4) Perform dynamic anchor assignment
    assigner = TaskAlignedAssigner()
    assigned_labels, assigned_centers, assigned_scores, assigned_sigmas = assigner.forward(
        pred_scores=pred_logits,  # [B, C, L]
        pred_centers=pred_centers,  # [B, L, 3]
        anchor_points=anchor_points,
        true_labels=torch.masked_fill(true_labels, true_labels.eq(-100), 0),
        true_centers=true_centers,
        true_sigmas=true_sigmas,
        pad_gt_mask=true_labels.eq(-100),
        bg_index=num_classes,
    )

    # 5) Classification loss: focal
    #    Use assigned_labels for each anchor
    #    Typically alpha=0.25, gamma=2.0 in standard focal;
    #    The user gave alpha=2.0 => that might be "gamma". So let's do:
    one_hot_label = torch.nn.functional.one_hot(assigned_labels, num_classes + 1)[..., :-1]
    cls_loss = varifocal_loss(
        pred_logits,
        assigned_scores,
        one_hot_label,
    )

    reg_loss = iou_loss(
        pred_centers=pred_centers,
        assigned_centers=assigned_centers,
        assigned_scores=assigned_scores,
        assigned_sigmas=assigned_sigmas,
        mask_positive=assigned_labels != num_classes,
    )

    total_loss = cls_loss + reg_loss
    divisor = assigned_scores.sum()

    return total_loss / divisor


class ObjectDetectionHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, stride: int):
        super().__init__()
        self.cls_head = nn.Conv3d(in_channels, num_classes, kernel_size=3, padding=1)
        self.offset_head = nn.Conv3d(in_channels, 3, kernel_size=3, padding=1)
        self.stride = stride

        torch.nn.init.zeros_(self.offset_head.weight)
        torch.nn.init.constant_(self.offset_head.bias, 0)

        # torch.nn.init.zeros_(self.conv.weight)
        # torch.nn.init.constant_(self.conv.bias, -3)

    def forward(self, features, labels=None, **loss_kwargs):
        logits = self.cls_head(features)
        offsets = self.offset_head(features)

        if torch.jit.is_tracing():
            return logits, offsets

        anchors = anchors_for_offsets_feature_map(offsets, self.stride)

        loss = None
        if labels is not None:
            loss = object_detection_loss(logits, offsets, anchors, labels, **loss_kwargs)

        return ObjectDetectionOutput(logits=logits, offsets=offsets, anchors=anchors, loss=loss)


def anchors_for_offsets_feature_map(offsets, stride):
    anchors = (
        torch.stack(
            torch.meshgrid(
                torch.arange(offsets.size(-3), device=offsets.device),
                torch.arange(offsets.size(-2), device=offsets.device),
                torch.arange(offsets.size(-1), device=offsets.device),
                indexing="ij",
            )
        )
        .float()
        .add_(0.5)
        .mul_(stride)
    )
    anchors = anchors[None, ...].repeat(offsets.size(0), 1, 1, 1, 1)
    return anchors

import dataclasses
from typing import Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F


@dataclasses.dataclass
class ObjectDetectionOutput:
    logits: Tensor
    offsets: Tensor

    loss: Optional[Tensor]


def decode_detections(logits, offsets, stride):
    """
    Decode detections from logits and offsets
    :param logits: Predicted logits B C D H W
    :param offsets: Predicted offsets B 3 D H W
    :param stride: Stride of the network

    :return: Tuple of probas and centers:
             probas - B C N
             centers - B 3 N

    """
    anchors = (
        torch.stack(
            torch.meshgrid(
                torch.arange(logits.size(-3), device=logits.device),
                torch.arange(logits.size(-2), device=logits.device),
                torch.arange(logits.size(-1), device=logits.device),
                indexing="ij",
            )
        )
        .add_(0.5)
        .float()
        .mul_(stride)
    )

    centers = anchors + offsets

    logits = torch.flatten(logits, start_dim=2)
    centers = torch.flatten(centers, start_dim=2)

    return logits, centers


def batch_pairwise_keypoints_iou(
    pred_keypoints: torch.Tensor,
    true_keypoints: torch.Tensor,
    true_sigmas: torch.Tensor,
) -> Tensor:
    """
    Calculate batched OKS (Object Keypoint Similarity) between two sets of keypoints.

    :param pred_keypoints: Centers with the shape [B, M1, 3]
    :param true_keypoints: Centers with the shape [B, M2, 3]
    :param true_sigmas:    Sigmas with the shape [B, M2]
    :param eps (float):    Small constant for numerical stability
    :return iou:           OKS between gt_keypoints and pred_keypoints with the shape [B, M1, M2]
    """

    centers1 = pred_keypoints[:, :, None, :]  # [B, 1, M2, 3]
    centers2 = true_keypoints[:, None, :, :]  # [B, M1, 1, 3]

    d = ((centers1 - centers2) ** 2).sum(dim=-1, keepdim=False)  # [B, M1, M2]

    sigmas = true_sigmas[:, None, :]  # [B, M1, M2]

    e: Tensor = d / (2 * sigmas**2)
    iou = torch.exp(-e)  # [B, M1, M2]
    return iou


def object_detection_loss(logits, offsets, labels, num_items_in_batch=None, alpha=2, beta=4, eps=1e-6, **kwargs):
    """
    Compute the Yolo-NAS Detection loss adapted for 3D data
    It uses keypoint-like IOU loss (negative exponent of mse) to assign the objectness score to the center of the object

    :param logits: Predicted headmap logits BxCxDxHxW
    :param offsets: Predicted offsets Bx3xDxHxW
    :param labels: Target labels encoded as Bx5xN where N is the number of objects and 5 is x,y,z,class,sigma
    :return: Single scalar loss
    """
    # 1) Decode predictions
    pred_logits, pred_centers = decode_detections(logits, offsets, stride=32)
    # shapes:
    # pred_logits:  [B, C, M]
    # pred_centers: [B, 3, M]

    # 2) Extract GT: [B, 5, N] => [B, 3, N], [B, N], [B, N]
    #    labels = (x, y, z, class, sigma)
    true_centers = labels[:, :3, :]  # [B, 3, N]
    true_labels = labels[:, 3, :]  # [B, N]
    true_sigmas = labels[:, 4, :]  # [B, N]

    # 3) Compute IoU (OKS) = batch_pairwise_keypoints_iou
    #    We want pred_centers => shape [B, M, 3], so permute
    pred_centers_perm = pred_centers.permute(0, 2, 1).contiguous()  # [B, M, 3]
    true_centers_perm = true_centers.permute(0, 2, 1).contiguous()  # [B, N, 3]
    iou_scores = batch_pairwise_keypoints_iou(
        pred_centers_perm,  # [B, M, 3]
        true_centers_perm,  # [B, N, 3]
        true_sigmas,  # [B, N]
    )  # -> [B, M, N]

    # 4) Perform dynamic anchor assignment
    ignore_mask = true_labels.eq(-100)  # [B, N]
    assigned_labels, matched_gt_idx = dynamic_anchor_assignment(
        iou_scores,  # [B, M, N]
        true_labels,  # [B, N]
        ignore_mask,
        pred_logits,  # [B, C, M]
    )
    # assigned_labels: [B, M] => in [-1..C-1]
    # matched_gt_idx:  [B, M] => index of GT matched or -1

    # 5) Classification loss: focal
    #    Use assigned_labels for each anchor
    #    Typically alpha=0.25, gamma=2.0 in standard focal;
    #    The user gave alpha=2.0 => that might be "gamma". So let's do:
    cls_loss = focal_loss(
        pred_logits, assigned_labels, alpha=0.25, gamma=alpha, eps=eps  # or your choice  # pass alpha=2.0 as focal gamma
    )

    # 6) Regression (OKS) loss for matched predictions
    #    matched_gt_idx[b, m] is the GT index or -1 if unmatched
    #    Let’s define the regression loss as:  reg_loss = sum(1 - iou_scores[b,m, idx]) for matched
    # shape of iou_scores: [B, M, N], so iou_scores[b,m, matched_gt_idx[b,m]]
    reg_mask = matched_gt_idx >= 0  # [B, M] => True if matched
    b_idx = torch.arange(reg_mask.size(0), device=reg_mask.device).unsqueeze(-1)
    m_idx = torch.arange(reg_mask.size(1), device=reg_mask.device).unsqueeze(0)

    # Gather the iou_score for matched entries
    matched_iou = iou_scores[b_idx, m_idx, matched_gt_idx.clamp_min(0)]
    # Zero out where not matched
    matched_iou = matched_iou * reg_mask

    # define regression loss
    reg_loss = (1.0 - matched_iou).sum() / (reg_mask.sum().clamp_min(1.0))

    # 7) Combine losses
    total_loss = cls_loss + reg_loss
    return total_loss


class ObjectDetectionHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, stride: int):
        super().__init__()
        self.cls_head = nn.Conv3d(in_channels, num_classes, kernel_size=3, padding=1)
        self.offset_head = nn.Conv3d(in_channels, 3, kernel_size=3, padding=1)
        self.stride = stride

        torch.nn.init.zeros_(self.conv.weight)
        torch.nn.init.constant_(self.conv.bias, -3)

    def forward(self, features, labels=None, **loss_kwargs):
        logits = self.conv(features)
        offsets = self.offset_head(features)

        centers = torch.meshgrid(
            torch.arange(logits.size(-3), device=logits.device),
            torch.arange(logits.size(-2), device=logits.device),
            torch.arange(logits.size(-1), device=logits.device),
        )

        if torch.jit.is_tracing():
            return logits, offsets

        loss = None
        if labels is not None:
            # loss = point_detection_loss(logits.float(), labels.float(), **loss_kwargs)
            loss = quality_focal_loss(logits.float(), labels.float(), **loss_kwargs)

        return PointDetectionOutput(logits=logits, loss=loss)

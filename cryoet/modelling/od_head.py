import dataclasses
from typing import Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from pytorch_toolbelt.losses.functional import focal_loss_with_logits


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


def dynamic_anchor_assignment(
    iou_scores: Tensor,
    true_labels: Tensor,
    ignore_mask: Tensor,
    pred_logits: Tensor,
):
    """
    Simple example of dynamic anchor assignment:
      - For each GT in the batch (that is not -100), pick the single best anchor
        with highest iou_score.
      - Ties are broken by simply taking the max idx.
      - If no iou is > 0, you might choose to ignore that GT as well (depends on threshold).

    :param iou_scores:  [B, M_pred, M_gt]
    :param true_labels: [B, M_gt]   (class label)
    :param ignore_mask: [B, M_gt]   (bool) True => ignore
    :param pred_logits: [B, C, M_pred]
    :return assigned_labels:  [B, M_pred] in [-1..C-1], -1 for background/unmatched
            matched_gt_idx:   [B, M_pred] which GT index is matched, or -1
    """
    B, M_pred, M_gt = iou_scores.shape
    C = pred_logits.size(1)

    # Initialize everything to -1 => background
    assigned_labels = iou_scores.new_full((B, M_pred), -1, dtype=torch.long)
    matched_gt_idx = iou_scores.new_full((B, M_pred), -1, dtype=torch.long)

    # We will do it "per batch" to avoid for loops over M_gt
    # 1) Invalidate iou_scores for GTs that are ignored
    #    iou_scores[b, :, j] = -1 if ignore_mask[b, j] == True
    iou_scores = iou_scores.clone()
    # broadcast ignore_mask => [B, 1, M_gt]
    iou_scores[ignore_mask[:, None, :].expand_as(iou_scores)] = -1.0

    # 2) For each GT, find the best anchor
    #    best_pred = argmax over dimension=1 => shape [B, M_gt]
    iou_scores_t = iou_scores.transpose(1, 2)  # shape [B, M_gt, M_pred]
    best_pred_idx = iou_scores_t.argmax(dim=-1)  # [B, M_gt]
    best_pred_vals = iou_scores_t.max(dim=-1).values  # [B, M_gt]

    # (optional) Could filter out GTs with best iou < some threshold if desired
    # threshold = 0.1
    # valid_gt_mask = best_pred_vals > threshold
    # but let's keep them as-is for simplicity

    # 3) Assign each GT to that best anchor
    #    Because multiple GTs can select the same anchor, the last GT might overwrite the first.
    #    For true "dynamic matching", you'd do a Hungarian or top-k approach.
    b_idx = torch.arange(B, device=iou_scores.device).unsqueeze(-1)  # [B,1]
    gt_idx = torch.arange(M_gt, device=iou_scores.device).unsqueeze(0)  # [1,M_gt]

    anchor_idx_for_gt = best_pred_idx  # [B, M_gt]
    # We also get the class from true_labels
    class_for_gt = true_labels.clone()  # [B, M_gt]
    # Some might be -100, we ignore them => we won't assign

    # flatten out (B, M_gt) => (B*M_gt,) for indexing
    # or we do a scatter_ approach:
    # assigned_labels[b, anchor_idx_for_gt[b, j]] = class_for_gt[b, j], for each j
    # matched_gt_idx[b, anchor_idx_for_gt[b, j]] = j
    for b in range(B):
        for j in range(M_gt):
            cl = class_for_gt[b, j].item()
            if cl == -100:
                continue  # ignore
            a = anchor_idx_for_gt[b, j].item()
            assigned_labels[b, a] = cl
            matched_gt_idx[b, a] = j

    return assigned_labels, matched_gt_idx


def object_detection_loss(logits, offsets, labels, num_items_in_batch=None, alpha=2, stride: int = 32, eps=1e-6, **kwargs):
    """
    Compute the detection loss adapted for 3D data
    It uses keypoint-like IOU loss (negative exponent of mse) to assign the objectness score to the center of the object

    :param logits: Predicted headmap logits BxCxDxHxW
    :param offsets: Predicted offsets Bx3xDxHxW
    :param labels: Target labels encoded as Bx5xN where N is the number of objects and 5 is x,y,z,class,sigma
    :return: Single scalar loss
    """
    # 1) Decode predictions
    pred_logits, pred_centers = decode_detections(logits, offsets, stride=stride)
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
    cls_loss = focal_loss_with_logits(pred_logits, assigned_labels, gamma=alpha, eps=eps, reduction="none")

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
    reg_loss = torch.masked_fill(1.0 - matched_iou, reg_mask, 0).sum()

    # 7) Combine losses
    total_loss = cls_loss + reg_loss

    return total_loss / num_items_in_batch


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

        if torch.jit.is_tracing():
            return logits, offsets

        loss = None
        if labels is not None:
            loss = object_detection_loss(logits, offsets, labels, **loss_kwargs)

        return ObjectDetectionOutput(logits=logits, offsets=offsets, loss=loss)

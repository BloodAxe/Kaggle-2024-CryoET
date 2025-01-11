from typing import Optional

import torch
from torch import Tensor, nn


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
    :return iou:           OKS between gt_keypoints and pred_keypoints with the shape [B, M2, M1]
    """

    centers1 = pred_keypoints[:, None, :, :]  # [B, M2, 1, 3]
    centers2 = true_keypoints[:, :, None, :]  # [B, 1, M1, 3]

    d = ((centers1 - centers2) ** 2).sum(dim=-1, keepdim=False)  # [B, M1, M2]

    sigmas = true_sigmas.reshape(true_keypoints.size(0), true_keypoints.size(1), 1)  # [B, M2, 1]

    e: Tensor = d / (2 * sigmas**2)
    iou = torch.exp(-e)  # [B, M2, M1]
    return iou


def compute_max_iou_anchor(ious: Tensor) -> Tensor:
    """
    For each anchor, find the GT with the largest IOU.

    :param ious: Tensor (float32) of shape[B, n, L], n: num_gts, L: num_anchors
    :return: is_max_iou is Tensor (float32) of shape[B, n, L], value=1. means selected
    """
    num_max_boxes = ious.shape[-2]
    max_iou_index = ious.argmax(dim=-2)
    is_max_iou: Tensor = torch.nn.functional.one_hot(max_iou_index, num_max_boxes).permute([0, 2, 1])
    return is_max_iou.type_as(ious)


def gather_topk_anchors(
    metrics: Tensor, topk: int, largest: bool = True, topk_mask: Optional[Tensor] = None, eps: float = 1e-9
) -> Tensor:
    """

    :param metrics:     Tensor(float32) of shape[B, n, L], n: num_gts, L: num_anchors
    :param topk:        The number of top elements to look for along the axis.
    :param largest:     If set to true, algorithm will sort by descending order, otherwise sort by ascending order.
    :param topk_mask:   Tensor(float32) of shape[B, n, 1], ignore bbox mask,
    :param eps:         Default: 1e-9

    :return: is_in_topk, Tensor (float32) of shape[B, n, L], value=1. means selected
    """
    num_anchors = metrics.shape[-1]
    topk_metrics, topk_idxs = torch.topk(metrics, topk, dim=-1, largest=largest)
    if topk_mask is None:
        topk_mask = (topk_metrics.max(dim=-1, keepdim=True).values > eps).type_as(metrics)
    is_in_topk = torch.nn.functional.one_hot(topk_idxs, num_anchors).sum(dim=-2).type_as(metrics)
    return is_in_topk * topk_mask


def check_points_inside_bboxes(anchor_points: Tensor, gt_centers: Tensor, gt_radius: Tensor, eps: float = 0.05) -> Tensor:
    """

    :param anchor_points: Tensor (float32) of shape[B, L, 2], "xy" format, L: num_anchors
    :param gt_centers:    Tensor (float32) of shape[B, n, 2], "xy" format, n: num gt points
    :param gt_radius:     Tensor (float32) of shape [B, n, 1]. Default: None.
    :param eps:           Default: 1e-9

    :return is_in_bboxes: Tensor (float32) of shape[B, n, L], value=1. means selected
    """
    iou = batch_pairwise_keypoints_iou(anchor_points, gt_centers, gt_radius)
    is_in_bboxes = iou > eps

    return is_in_bboxes.type_as(gt_centers)


class TaskAlignedAssigner(nn.Module):

    def __init__(self, max_anchors_per_point=13, alpha=1.0, beta=6.0, eps=1e-9):
        """

        :param max_anchors_per_point: Maximum number of achors that is selected for each gt box
        :param alpha: Power factor for class probabilities of predicted boxes (Used compute alignment metric)
        :param beta: Power factor for IoU score of predicted boxes (Used compute alignment metric)
        :param eps: Small constant for numerical stability
        """
        super(TaskAlignedAssigner, self).__init__()
        self.topk = max_anchors_per_point
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(
        self,
        pred_scores: Tensor,
        pred_centers: Tensor,
        anchor_points: Tensor,
        true_labels: Tensor,
        true_centers: Tensor,
        true_sigmas: Tensor,
        pad_gt_mask: Tensor,
        bg_index: int,
    ):
        """
        This code is based on https://github.com/fcjian/TOOD/blob/master/mmdet/core/bbox/assigners/task_aligned_assigner.py

        The assignment is done in following steps
        1. compute alignment metric between all bbox (bbox of all pyramid levels) and gt
        2. select top-k bbox as candidates for each gt
        3. limit the positive sample's center in gt (because the anchor-free detector
           only can predict positive distance)
        4. if an anchor box is assigned to multiple gts, the one with the
           highest iou will be selected.

        :param pred_scores: Tensor (float32): predicted class probability, shape(B, L, C)
        :param pred_centers: Tensor (float32): predicted bounding boxes, shape(B, L, 3)
        :param anchor_points: Tensor (float32): pre-defined anchors, shape(L, 2), "cxcy" format
        :param true_labels: Tensor (int64|int32): Label of gt_bboxes, shape(B, n, 1)
        :param true_centers: Tensor (float32): Ground truth bboxes, shape(B, n, 3)
        :param pad_gt_mask: Tensor (float32): 1 means bbox, 0 means no bbox, shape(B, n, 1).
                            Can be None, which means all gt_bboxes are valid.
        :param bg_index: int ( background index
        :return:
            - assigned_labels, Tensor of shape (B, L)
            - assigned_bboxes, Tensor of shape (B, L, 4)
            - assigned_scores, Tensor of shape (B, L, C)
        """
        assert pred_scores.ndim == pred_centers.ndim
        assert true_labels.ndim == true_centers.ndim and true_centers.ndim == 3

        batch_size, num_anchors, num_classes = pred_scores.shape
        _, num_max_boxes, _ = true_centers.shape

        # negative batch
        if num_max_boxes == 0:
            assigned_labels = torch.full([batch_size, num_anchors], bg_index, dtype=torch.long, device=true_labels.device)
            assigned_points = torch.zeros([batch_size, num_anchors, 3], device=true_labels.device)
            assigned_scores = torch.zeros([batch_size, num_anchors, num_classes], device=true_labels.device)
            assigned_sigmas = torch.zeros([batch_size, num_anchors], device=true_labels.device)
            return assigned_labels, assigned_points, assigned_scores, assigned_sigmas

        # compute iou between gt and pred bbox, [B, n, L]
        # ious = batch_iou_similarity(gt_bboxes, pred_bboxes)
        ious = batch_pairwise_keypoints_iou(
            pred_centers,  # [B, M, 3]
            true_centers,  # [B, N, 3]
            true_sigmas,  # [B, N]
        )  # -> [B, M, N]

        # gather pred bboxes class score
        pred_scores = torch.permute(pred_scores, [0, 2, 1])
        batch_ind = torch.arange(end=batch_size, dtype=true_labels.dtype, device=true_labels.device).unsqueeze(-1)
        gt_labels_ind = torch.stack([batch_ind.tile([1, num_max_boxes]), true_labels.squeeze(-1)], dim=-1)

        bbox_cls_scores = pred_scores[gt_labels_ind[..., 0], gt_labels_ind[..., 1]]

        # compute alignment metrics, [B, n, L]
        alignment_metrics = bbox_cls_scores.pow(self.alpha) * ious.pow(self.beta)

        # check the positive sample's center in gt, [B, n, L]
        is_in_gts = check_points_inside_bboxes(anchor_points, true_centers, true_sigmas)

        # select topk largest alignment metrics pred bbox as candidates
        # for each gt, [B, n, L]
        is_in_topk = gather_topk_anchors(alignment_metrics * is_in_gts, self.topk, topk_mask=pad_gt_mask)

        # select positive sample, [B, n, L]
        mask_positive = is_in_topk * is_in_gts * pad_gt_mask

        # if an anchor box is assigned to multiple gts,
        # the one with the highest iou will be selected, [B, n, L]
        mask_positive_sum = mask_positive.sum(dim=-2)
        if mask_positive_sum.max() > 1:
            mask_multiple_gts = (mask_positive_sum.unsqueeze(1) > 1).tile([1, num_max_boxes, 1])
            is_max_iou = compute_max_iou_anchor(ious)
            mask_positive = torch.where(mask_multiple_gts, is_max_iou, mask_positive)
            mask_positive_sum = mask_positive.sum(dim=-2)
        assigned_gt_index = mask_positive.argmax(dim=-2)

        # assigned target
        assigned_gt_index = assigned_gt_index + batch_ind * num_max_boxes
        assigned_labels = torch.gather(true_labels.flatten(), index=assigned_gt_index.flatten(), dim=0)
        assigned_labels = assigned_labels.reshape([batch_size, num_anchors])
        assigned_labels = torch.where(mask_positive_sum > 0, assigned_labels, torch.full_like(assigned_labels, bg_index))

        assigned_points = true_centers.reshape([-1, 3])[assigned_gt_index.flatten(), :]
        assigned_points = assigned_points.reshape([batch_size, num_anchors, 3])

        assigned_sigmas = true_sigmas.reshape([-1])[assigned_gt_index.flatten()]
        assigned_sigmas = assigned_sigmas.reshape([batch_size, num_anchors])

        assigned_scores = torch.nn.functional.one_hot(assigned_labels, num_classes + 1)
        ind = list(range(num_classes + 1))
        ind.remove(bg_index)
        assigned_scores = torch.index_select(
            assigned_scores, index=torch.tensor(ind, device=assigned_scores.device, dtype=torch.long), dim=-1
        )
        # rescale alignment metrics
        alignment_metrics *= mask_positive
        max_metrics_per_instance = alignment_metrics.max(dim=-1, keepdim=True).values
        max_ious_per_instance = (ious * mask_positive).max(dim=-1, keepdim=True).values
        alignment_metrics = alignment_metrics / (max_metrics_per_instance + self.eps) * max_ious_per_instance
        alignment_metrics = alignment_metrics.max(dim=-2).values.unsqueeze(-1)
        assigned_scores = assigned_scores * alignment_metrics

        return assigned_labels, assigned_points, assigned_scores, assigned_sigmas

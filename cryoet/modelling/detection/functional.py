import einops
import torch
from torch import Tensor

from .task_aligned_assigner import TaskAlignedAssigner
from pytorch_toolbelt.utils.distributed import is_dist_avail_and_initialized, get_world_size
import torch.distributed as dist


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


def keypoint_similarity(pts1, pts2, sigmas):
    """
    Compute similarity between two sets of keypoints
    :param pts1: ...x3
    :param pts2: ...x3
    """
    d = ((pts1 - pts2) ** 2).sum(dim=-1, keepdim=False)  # []
    e: Tensor = d / (2 * sigmas**2)
    iou = torch.exp(-e)
    return iou


def iou_loss(pred_centers, assigned_centers, assigned_scores, assigned_sigmas, mask_positive):
    weight = assigned_scores.sum(-1)

    iou = keypoint_similarity(pred_centers, assigned_centers, assigned_sigmas)
    loss2 = (1 - iou) * weight

    loss_reduced_iou = torch.masked_fill(loss2, ~mask_positive, 0).sum()
    return loss_reduced_iou

    # loss = (
    #     torch.nn.functional.smooth_l1_loss(pred_centers, assigned_centers, reduction="none")
    #     .sum(dim=-1, keepdim=False)
    #     .mul(weight)
    # )

    # loss_reduced = torch.masked_fill(loss, ~mask_positive, 0).sum()
    # return loss_reduced


def maybe_all_reduce(x: Tensor, op=dist.ReduceOp.SUM):
    if not is_dist_avail_and_initialized():
        return x

    xc = x.clone()
    dist.all_reduce(xc, op=op)
    return xc


def object_detection_loss(logits, offsets, anchors, labels, average_tokens_across_devices: bool = False, **kwargs):
    """
    Compute the detection loss adapted for 3D data
    It uses keypoint-like IOU loss (negative exponent of mse) to assign the objectness score to the center of the object

    :param logits:  Predicted headmap logits BxCxDxHxW
    :param offsets: Predicted offsets Bx3xDxHxW
    :param anchors: Anchor points Bx3xDxHxW
    :param labels:  Target labels encoded as BxNx5 where N is the number of objects and 5 is x,y,z,class,sigma
    :return:        Single scalar loss
    """

    # if not torch.isfinite(logits).all():
    #     print("Logits are not finite")
    # if not torch.isfinite(offsets).all():
    #     print("Offsets are not finite")

    # 1) Decode predictions
    pred_logits, pred_centers, anchor_points = decode_detections(logits, offsets, anchors)
    # shapes:
    # pred_logits:  [B, L, C]
    # pred_centers: [B, L, 3]
    batch_size, num_anchors, num_classes = pred_logits.size()

    # 2) Extract GT: [B, 5, N] => [B, 3, N], [B, N], [B, N]
    #    labels = (x, y, z, class, sigma)
    true_centers = labels[:, :, :3]  # [B, n, 3]
    true_labels = labels[:, :, 3:4].long()  # [B, n, 1]
    true_sigmas = labels[:, :, 4:5]  # [B, n, 1]

    # 4) Perform dynamic anchor assignment
    assigner = TaskAlignedAssigner()
    assigned_labels, assigned_centers, assigned_scores, assigned_sigmas = assigner(
        pred_scores=pred_logits.detach().sigmoid(),
        pred_centers=pred_centers,
        anchor_points=anchor_points,
        true_labels=torch.masked_fill(true_labels, true_labels.eq(-100), 0),
        true_centers=true_centers,
        true_sigmas=true_sigmas,
        pad_gt_mask=true_labels.ne(-100),
        bg_index=num_classes,
    )
    # if not torch.isfinite(assigned_scores).all():
    #     print("Assigned scores are not finite")

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
    # if not torch.isfinite(cls_loss).all():
    #     print("Classification loss is not finite")

    reg_loss = iou_loss(
        pred_centers=pred_centers,
        assigned_centers=assigned_centers,
        assigned_scores=assigned_scores,
        assigned_sigmas=assigned_sigmas,
        mask_positive=assigned_labels != num_classes,
    )

    total_loss = cls_loss  # + reg_loss
    divisor = assigned_scores.sum()

    # present_labels = true_labels.numel() - true_labels.eq(-100).sum()
    # print("Total loss", total_loss, "Divisor", divisor, "present_labels", present_labels, flush=True)
    if average_tokens_across_devices and is_dist_avail_and_initialized():
        total_loss = maybe_all_reduce(total_loss)
        divisor = maybe_all_reduce(divisor.detach())

    divisor = divisor.clamp_min(1)

    return total_loss / divisor, divisor


def anchors_for_offsets_feature_map(offsets, stride):
    z, y, x = torch.meshgrid(
        torch.arange(offsets.size(-3), device=offsets.device),
        torch.arange(offsets.size(-2), device=offsets.device),
        torch.arange(offsets.size(-1), device=offsets.device),
        indexing="ij",
    )
    anchors = torch.stack([x, y, z], dim=0)
    anchors = anchors.float().add_(0.5).mul_(stride)

    anchors = anchors[None, ...].repeat(offsets.size(0), 1, 1, 1, 1)
    return anchors

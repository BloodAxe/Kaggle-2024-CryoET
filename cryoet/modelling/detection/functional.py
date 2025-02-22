from typing import List, Tuple, Union, Optional

import einops
import numpy as np
import timm.layers
import torch
from timm.layers.grn import GlobalResponseNorm
from torch import Tensor, nn

from .global_response_norm import GlobalResponseNorm3d
from .layer_norm_3d import LayerNorm3d
from .task_aligned_assigner import TaskAlignedAssigner
from pytorch_toolbelt.utils.distributed import is_dist_avail_and_initialized, get_world_size
import torch.distributed as dist

from ...data.functional import as_tuple_of_3


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


def decode_detections(logits: Tensor | List[Tensor], offsets: Tensor | List[Tensor], strides: int | List[int]):
    """
    Decode detections from logits and offsets
    :param logits: Predicted logits B C D H W
    :param offsets: Predicted offsets B 3 D H W
    :param anchors: Stride of the network

    :return: Tuple of probas and centers:
             probas - B N C
             centers - B N 3

    """
    if torch.is_tensor(logits):
        logits = [logits]
    if torch.is_tensor(offsets):
        offsets = [offsets]
    if isinstance(strides, int):
        strides = [strides]

    anchors = [anchors_for_offsets_feature_map(offset, s) for offset, s in zip(offsets, strides)]

    logits_flat = []
    centers_flat = []
    anchors_flat = []

    for logit, offset, anchor in zip(logits, offsets, anchors):
        centers = anchor + offset

        logits_flat.append(einops.rearrange(logit, "B C D H W -> B (D H W) C"))
        centers_flat.append(einops.rearrange(centers, "B C D H W -> B (D H W) C"))
        anchors_flat.append(einops.rearrange(anchor, "B C D H W -> B (D H W) C"))

    logits_flat = torch.cat(logits_flat, dim=1)
    centers_flat = torch.cat(centers_flat, dim=1)
    anchors_flat = torch.cat(anchors_flat, dim=1)

    return logits_flat, centers_flat, anchors_flat


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


def iou_loss(pred_centers, assigned_centers, assigned_scores, assigned_sigmas, mask_positive, use_l1_loss=False):
    num_pos = mask_positive.sum()
    if num_pos > 0:
        weight = assigned_scores.sum(-1)

        iou = keypoint_similarity(pred_centers, assigned_centers, assigned_sigmas)
        iou_loss = 1 - iou

        loss = iou_loss

        if use_l1_loss:
            l1_loss = torch.nn.functional.smooth_l1_loss(pred_centers, assigned_centers, reduction="none").sum(-1)
            loss = loss + l1_loss

        loss_reduced_iou = torch.masked_fill(loss * weight, ~mask_positive, 0).sum()
        return loss_reduced_iou
    else:
        loss_reduced_iou = torch.zeros([], device=pred_centers.device)

    return loss_reduced_iou


def maybe_all_reduce(x: Tensor, op=dist.ReduceOp.SUM):
    if not is_dist_avail_and_initialized():
        return x

    xc = x.clone()
    dist.all_reduce(xc, op=op)
    return xc


def object_detection_loss(
    logits: Tensor | List[Tensor],
    offsets: Tensor | List[Tensor],
    strides: int | List[int],
    labels: Tensor,
    average_tokens_across_devices: bool = False,
    use_l1_loss: bool = False,
    use_offset_head: bool = True,
    assigned_min_iou_for_anchor=0.05,
    assigner_max_anchors_per_point: int = 13,
    assigner_alpha=1.0,
    assigner_beta=6.0,
    use_varifocal_loss: bool = True,
    use_cross_entropy_loss: bool = False,
    **kwargs,
):
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
    pred_logits, pred_centers, anchor_points = decode_detections(logits, offsets, strides)
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
    assigner = TaskAlignedAssigner(
        max_anchors_per_point=assigner_max_anchors_per_point,
        assigned_min_iou_for_anchor=assigned_min_iou_for_anchor,
        alpha=assigner_alpha,
        beta=assigner_beta,
    )

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
    if use_varifocal_loss:
        one_hot_label = torch.nn.functional.one_hot(assigned_labels, num_classes + 1)[..., :-1]
        cls_loss = varifocal_loss(
            pred_logits,
            assigned_scores,
            one_hot_label,
        )
    else:
        cls_loss = focal_loss(pred_logits, assigned_scores, alpha=-1)

    if use_cross_entropy_loss or True:
        bg_label_mask = assigned_labels.eq(num_classes)
        ce_loss = torch.nn.functional.cross_entropy(
            input=pred_logits.permute(0, 2, 1), target=torch.masked_fill(assigned_labels, bg_label_mask, -100), reduction="none"
        )
        ce_loss_sum = torch.sum(ce_loss * assigned_scores.sum(-1))
        cls_loss += ce_loss_sum

    # if not torch.isfinite(cls_loss).all():
    #     print("Classification loss is not finite")

    if use_offset_head:
        reg_loss = iou_loss(
            pred_centers=pred_centers,
            assigned_centers=assigned_centers,
            assigned_scores=assigned_scores,
            assigned_sigmas=assigned_sigmas,
            mask_positive=assigned_labels != num_classes,
            use_l1_loss=use_l1_loss,
        )
    else:
        reg_loss = torch.tensor(0.0, device=pred_centers.device)

    divisor = assigned_scores.sum()

    # present_labels = true_labels.numel() - true_labels.eq(-100).sum()
    # print("Total loss", total_loss, "Divisor", divisor, "present_labels", present_labels, flush=True)
    if average_tokens_across_devices and is_dist_avail_and_initialized():
        divisor = maybe_all_reduce(divisor.detach())
        cls_loss.mul_(get_world_size())
        reg_loss.mul_(get_world_size())

    divisor = divisor.clamp_min(1)
    cls_loss.div_(divisor)
    reg_loss.div_(divisor)
    loss = cls_loss + reg_loss

    loss_dict = {
        "loss": float(loss),
        "cls_loss": float(cls_loss),
        "reg_loss": float(reg_loss),
        "num_items_in_batch": float(divisor),
    }
    return loss, loss_dict


@torch.no_grad()
def decode_detections_with_nms(
    scores: List[Tensor],
    offsets: List[Tensor],
    strides: List[int],
    min_score: Union[float, List[float]],
    class_sigmas: List[float],
    iou_threshold: float = 0.25,
    use_single_label_per_anchor: bool = True,
    use_centernet_nms: bool = False,
    pre_nms_top_k: Optional[int] = None,
    class_map_gaussian_smoothing_kernel=0,
    centernet_nms_kernel: Union[int, Tuple[int, int, int]] = 3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Decode detections from scores and centers with NMS

    :param scores: Predicted scores of shape (C, D, H, W)
    :param offsets: Predicted offsets of shape (3, D, H, W)
    :param min_score: Minimum score to consider a detection
    :param class_sigmas: Class sigmas (class radius for NMS), length = number of classes
    :param iou_threshold: Threshold above which detections are suppressed

    :return:
        - final_centers [N, 3] (x, y, z)
        - final_labels [N]
        - final_scores [N]
    """

    # Number of classes is the second dimension of `scores`
    # e.g. scores shape = (C, D, H, W)
    num_classes = scores[0].shape[0]  # the 'C' dimension

    # Allow min_score to be a single value or a list of values
    min_score = np.asarray(min_score, dtype=np.float32).reshape(-1)
    if len(min_score) == 1:
        min_score = np.full(num_classes, min_score[0], dtype=np.float32)

    if class_map_gaussian_smoothing_kernel > 0:
        scores = [
            gaussian_blur_3d(s.unsqueeze(0), kernel_size=class_map_gaussian_smoothing_kernel, sigma=1.0).squeeze(0)
            for s in scores
        ]

    if use_centernet_nms:
        scores = [centernet_heatmap_nms(s.unsqueeze(0), kernel=centernet_nms_kernel).squeeze(0) for s in scores]

    scores, centers, _ = decode_detections([s.unsqueeze(0) for s in scores], [o.unsqueeze(0) for o in offsets], strides)
    scores = scores.squeeze(0)
    centers = centers.squeeze(0)

    labels_of_max_score = scores.argmax(dim=1)

    # Prepare final outputs
    final_labels_list = []
    final_scores_list = []
    final_centers_list = []

    # NMS per class
    for class_index in range(num_classes):
        sigma_value = float(class_sigmas[class_index])  # Get the sigma for this class
        score_threshold = float(min_score[class_index])
        score_mask = scores[:, class_index] >= score_threshold  # Filter out low-scoring detections

        if use_single_label_per_anchor:
            class_mask = labels_of_max_score.eq(class_index)  # Pick out only detections of this class
            mask = class_mask & score_mask
        else:
            mask = score_mask

        if not mask.any():
            continue

        class_scores = scores[mask, class_index]  # shape: [Nc]
        class_centers = centers[mask]  # shape: [Nc, 3]

        if pre_nms_top_k is not None and len(class_scores) > pre_nms_top_k:
            class_scores, sort_idx = torch.topk(class_scores, pre_nms_top_k, largest=True, sorted=True)
            class_centers = class_centers[sort_idx]
        else:
            class_scores, sort_idx = class_scores.sort(descending=True)
            class_centers = class_centers[sort_idx]

        # Run a simple “greedy” NMS
        keep_indices = []
        suppressed = torch.zeros_like(class_scores, dtype=torch.bool)  # track suppressed

        # print(f"Predictions for class {class_index}: ", torch.count_nonzero(class_mask).item())

        for i in range(class_scores.size(0)):
            if suppressed[i]:
                continue
            # Keep this detection
            keep_indices.append(i)

            # Suppress detections whose IoU with i is above threshold
            iou = keypoint_similarity(class_centers[i : i + 1, :], class_centers, sigma_value)

            high_iou_mask = iou > iou_threshold
            suppressed |= high_iou_mask.to(suppressed.device)

        print(f"Predictions for class {class_index} after NMS", len(keep_indices))

        # Gather kept detections for this class
        keep_indices = torch.as_tensor(keep_indices, dtype=torch.long, device=class_scores.device)
        final_labels_list.append(torch.full((keep_indices.numel(),), class_index, dtype=torch.long))
        final_scores_list.append(class_scores[keep_indices])
        final_centers_list.append(class_centers[keep_indices])

    # Concatenate from all classes
    final_labels = torch.cat(final_labels_list, dim=0) if final_labels_list else torch.empty((0,), dtype=torch.long)
    final_scores = torch.cat(final_scores_list, dim=0) if final_scores_list else torch.empty((0,))
    final_centers = torch.cat(final_centers_list, dim=0) if final_centers_list else torch.empty((0, 3))

    print(f"Final predictions after NMS: {final_centers.size(0)}")
    return final_centers, final_labels, final_scores


def gaussian_blur_3d(x: Tensor, kernel_size: int, sigma: float):
    # build gaussian kernel
    kd, kh, kw = as_tuple_of_3(kernel_size)
    z = torch.linspace(-(kd // 2), kd // 2, steps=kd)
    y = torch.linspace(-(kh // 2), kh // 2, steps=kh)
    x_ = torch.linspace(-(kw // 2), kw // 2, steps=kw)

    zz, yy, xx = torch.meshgrid(z, y, x_, indexing="ij")
    kernel_3d = torch.exp(-(xx**2 + yy**2 + zz**2) / (2 * sigma**2))
    kernel_3d /= kernel_3d.sum()
    #  normalize
    kernel_3d = kernel_3d.to(x.device).to(x.dtype)

    C = x.shape[1]
    kernel_3d = kernel_3d.view(1, 1, *kernel_3d.shape)
    kernel_3d = kernel_3d.repeat(C, 1, 1, 1, 1)

    # apply gaussian kernel
    x = torch.nn.functional.conv3d(x, weight=kernel_3d, bias=None, padding=kernel_size // 2, stride=1, groups=C)
    return x


def centernet_heatmap_nms(scores, kernel: Union[int, Tuple[int, int, int]] = 3):
    kernel = as_tuple_of_3(kernel)
    pad = (kernel[0] - 1) // 2, (kernel[1] - 1) // 2, (kernel[2] - 1) // 2

    maxpool = torch.nn.functional.max_pool3d(scores, kernel_size=kernel, padding=pad, stride=1)

    mask = scores == maxpool
    peaks = scores * mask
    return peaks


def focal_loss(pred_logits: Tensor, label: Tensor, alpha=0.25, gamma=2.0, reduction="sum") -> Tensor:
    pred_score = pred_logits.sigmoid()
    weight = torch.abs(pred_score - label).pow(gamma)
    if alpha > 0:
        alpha_t = alpha * label + (1 - alpha) * (1 - label)
        weight *= alpha_t
    loss = weight * torch.nn.functional.binary_cross_entropy_with_logits(pred_logits, label, reduction="none")

    if reduction == "sum":
        loss = loss.sum()
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "none":
        pass
    else:
        raise ValueError(f"Unsupported reduction type {reduction}")
    return loss


def convert_2d_to_3d(model: nn.Module, replace_relu_with_silu=False) -> nn.Module:
    """
    Recursively convert all Conv2d layers in `model` to Conv3d,
    and all BatchNorm2d layers to BatchNorm3d.
    Replicates the existing Conv2d weights along the 3rd dimension (depth=1 by default).
    """
    for name, module in model.named_children():

        # If we find a Conv2d, replace it with a Conv3d.
        if isinstance(module, nn.Conv2d):
            # --------------------------------------------
            # 1) Check that the 2D kernel is square
            # --------------------------------------------
            if module.kernel_size[0] != module.kernel_size[1]:
                raise ValueError(
                    f"Non-square kernel detected: {module.kernel_size}. " "This example only handles square kernels (k, k)."
                )
            k = module.kernel_size[0]

            # --------------------------------------------
            # 2) Build a new Conv3d with kernel_size = (k, k, k)
            #    using the same hyperparameters as best we can
            # --------------------------------------------
            new_conv = nn.Conv3d(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel_size=(k, k, k),
                # For stride, padding, and dilation, we replicate
                # the 2D values in each dimension:
                stride=(module.stride[0], module.stride[0], module.stride[1]),
                padding=(module.padding[0], module.padding[0], module.padding[1]),
                dilation=(module.dilation[0], module.dilation[0], module.dilation[1]),
                groups=module.groups,
                bias=(module.bias is not None),
            )

            # --------------------------------------------
            # 3) Copy and replicate the 2D weights -> 3D
            # old_weight shape: (out_c, in_c, k, k)
            # new_weight shape: (out_c, in_c, k, k, k)
            # --------------------------------------------
            with torch.no_grad():
                old_weight = module.weight  # shape: (out_c, in_c, k, k)
                # Expand along a new depth dimension
                old_weight_3d = old_weight.unsqueeze(2)  # (out_c, in_c, 1, k, k)
                old_weight_3d = old_weight_3d.repeat(1, 1, k, 1, 1).div(k)  # (out_c, in_c, k, k, k)
                new_conv.weight.copy_(old_weight_3d)

                if module.bias is not None:
                    new_conv.bias.copy_(module.bias)

            # Replace the old Conv2d with our new Conv3d
            setattr(model, name, new_conv)

        # If we find a BatchNorm2d, replace it with a BatchNorm3d.
        elif isinstance(module, nn.BatchNorm2d):
            new_bn = nn.BatchNorm3d(
                num_features=module.num_features,
                eps=module.eps,
                momentum=module.momentum,
                affine=module.affine,
                track_running_stats=module.track_running_stats,
            )

            # Copy running statistics and affine parameters
            with torch.no_grad():
                if module.affine:
                    new_bn.weight.copy_(module.weight)
                    new_bn.bias.copy_(module.bias)
                new_bn.running_mean.copy_(module.running_mean)
                new_bn.running_var.copy_(module.running_var)

            # Replace the BatchNorm2d with BatchNorm3d
            setattr(model, name, new_bn)
        elif isinstance(module, GlobalResponseNorm):
            channels_last = module.spatial_dim == (1, 2)
            new_norm = GlobalResponseNorm3d(dim=len(module.weight), eps=module.eps, channels_last=channels_last)
            with torch.no_grad():
                new_norm.weight.copy_(module.weight)
                new_norm.bias.copy_(module.bias)

            setattr(model, name, new_norm)
        elif isinstance(module, timm.layers.norm.LayerNorm2d):
            new_norm = LayerNorm3d(module.normalized_shape, eps=module.eps, affine=module.elementwise_affine)
            with torch.no_grad():
                if module.elementwise_affine:
                    new_norm.weight.copy_(module.weight)
                    new_norm.bias.copy_(module.bias)

            setattr(model, name, new_norm)

        elif isinstance(module, nn.ReLU) and replace_relu_with_silu:
            # Replace with SILU
            setattr(model, name, nn.SiLU(inplace=True))
        elif isinstance(module, nn.Dropout2d):
            # Replace with Dropout3d
            setattr(model, name, nn.Dropout3d(p=module.p, inplace=module.inplace))
        else:
            # Recursively convert children
            convert_2d_to_3d(module)

    return model

import torch
from typing import List, Tuple


def decode_detections_with_nms(
    scores: torch.Tensor, centers: torch.Tensor, min_score: float, class_sigmas: List[float], iou_threshold: float = 0.25
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Decode detections from scores and centers with NMS

    :param scores: Predicted scores of shape (B, C, D, H, W) or (C, D, H, W).
                   For simplicity, below we assume shape (C, D, H, W) and we flatten
                   in the first dimension: (C, -1).
    :param centers: Predicted centers of shape (B, D, H, W, 3) or (D, H, W, 3).
                    For simplicity, below we assume (D, H, W, 3) and flatten: (-1, 3).
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
    num_classes = scores.shape[0]  # the 'C' dimension

    # Flatten spatial dimensions so that scores.shape becomes (C, -1)
    scores = torch.flatten(scores, start_dim=1)  # shape: (C, D*H*W)
    scores = torch.transpose(scores, 0, 1)  # shape: (D*H*W, C)

    centers = torch.flatten(centers, start_dim=1)  # shape: (3, D*H*W)
    centers = torch.transpose(centers, 0, 1)  # shape: (D*H*W, 3)

    # max_scores: shape [D*H*W], class_labels: shape [D*H*W]
    max_scores = scores.max(dim=1)
    class_labels = max_scores.indices
    class_scores = max_scores.values

    # Filter out low-scoring detections
    mask = class_scores >= min_score
    labels = class_labels[mask]  # shape: [M]
    scores = class_scores[mask]  # shape: [M]
    centers = centers[mask]  # shape: [M, 3]

    # Sort remaining detections by descending score
    scores, sort_idx = scores.sort(descending=True)
    labels = labels[sort_idx]
    centers = centers[sort_idx]

    # Prepare final outputs
    final_labels_list = []
    final_scores_list = []
    final_centers_list = []

    # NMS per class
    for class_index in range(num_classes):
        # Pick out only detections of this class
        class_mask = labels == class_index
        if not class_mask.any():
            continue

        class_centers = centers[class_mask]  # shape: [Nc, 3]
        class_scores = scores[class_mask]  # shape: [Nc]

        # Get the sigma for this class
        sigma_value = float(class_sigmas[class_index])

        # Run a simple “greedy” NMS
        keep_indices = []
        suppressed = torch.zeros_like(class_scores, dtype=torch.bool)  # track suppressed

        for i in range(class_scores.size(0)):
            if suppressed[i]:
                continue
            # Keep this detection
            keep_indices.append(i)

            # Suppress detections whose IoU with i is above threshold

            # Precompute pairwise IoU = exp(-(||x_i - x_j||^2) / (2 * sigma^2))
            # shape of d: [Nc, Nc]
            d = ((class_centers[i : i + 1, :] - class_centers) ** 2).sum(dim=-1)
            e = d / (2 * sigma_value**2)
            iou = torch.exp(-e)  # shape: [Nc, Nc]

            high_iou_mask = iou[i] > iou_threshold
            suppressed |= high_iou_mask

        # Gather kept detections for this class
        keep_indices = torch.as_tensor(keep_indices, dtype=torch.long)
        final_labels_list.append(torch.full((keep_indices.numel(),), class_index, dtype=torch.long))
        final_scores_list.append(class_scores[keep_indices])
        final_centers_list.append(class_centers[keep_indices])

    # Concatenate from all classes
    final_labels = torch.cat(final_labels_list, dim=0) if final_labels_list else torch.empty((0,), dtype=torch.long)
    final_scores = torch.cat(final_scores_list, dim=0) if final_scores_list else torch.empty((0,))
    final_centers = torch.cat(final_centers_list, dim=0) if final_centers_list else torch.empty((0, 3))

    return final_centers, final_labels, final_scores

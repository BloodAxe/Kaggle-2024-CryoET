import math

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch

from cryoet.modelling.detection.detection_head import ObjectDetectionHead
from cryoet.modelling.detection.functional import (
    anchors_for_offsets_feature_map,
    iou_loss,
    keypoint_similarity,
    decode_detections,
    decode_detections_with_nms,
)
from cryoet.modelling.detection.task_aligned_assigner import batch_pairwise_keypoints_iou, TaskAlignedAssigner
from cryoet.training.od_accumulator import AccumulatedObjectDetectionPredictionContainer


def test_nms():
    offsets = torch.zeros(1, 3, 7, 8, 9).float()
    anchors = anchors_for_offsets_feature_map(offsets, stride=1)

    scores = torch.zeros(1, 5, 7, 8, 9).float()
    centers = offsets + anchors
    TARGET_SIGMAS = [5, 5, 6]

    scores[0, 0, 3, 4, 5] = 1.0
    scores[0, 0, 5, 6, 0] = 1.0

    scores[0, 1, 6, 6, 6] = 1.0
    scores[0, 1, 4, 4, 4] = 0.990

    scores[0, 2, 2, 2, 2] = 0.999

    topk_coords_px, topk_clses, topk_scores = decode_detections_with_nms(
        scores[0],
        centers[0],
        class_sigmas=TARGET_SIGMAS,
        min_score=0.01,
        iou_threshold=0.75,
    )

    print(topk_coords_px)
    print(topk_clses)
    print(topk_scores)


def test_assigner():
    offsets = torch.zeros(1, 3, 7, 8, 9)
    anchors = anchors_for_offsets_feature_map(offsets, stride=1)
    print(anchors[0, :, -1, 0, 0])
    print(anchors[0, :, 0, -1, 0])
    print(anchors[0, :, 0, 0, -1])

    labels = torch.tensor(
        [
            [
                [3, 4, 5, 0, 5],
                [6, 7, 8, 1, 5],
            ]
        ]
    )

    num_classes = 2
    logits = torch.randn((1, num_classes, 7, 8, 9))

    pred_logits, pred_centers, anchor_points = decode_detections(logits, offsets, anchors)

    true_centers = labels[:, :, :3]  # [B, n, 3]
    true_labels = labels[:, :, 3:4].long()  # [B, n, 1]
    true_sigmas = labels[:, :, 4:5]  # [B, n, 1]

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

    print(assigned_labels)
    print(assigned_centers)
    print(assigned_scores)
    print(assigned_sigmas)


def test_anchors_for_offsets_feature_map():
    offsets = torch.zeros(1, 3, 12, 34, 56)

    anchors = anchors_for_offsets_feature_map(offsets, stride=1)
    print(anchors[0, :, -1, 0, 0])
    print(anchors[0, :, 0, -1, 0])
    print(anchors[0, :, 0, 0, -1])


def test_loss():
    offsets = torch.zeros(1, 3, 96, 96, 96)

    anchors = anchors_for_offsets_feature_map(offsets, stride=1)

    gt = torch.tensor([[[40, 77, 11]]])

    anchors = einops.rearrange(anchors, "B C D H W -> B (D H W) C")
    iou = keypoint_similarity(anchors, gt, torch.tensor([5]))
    print(iou.max(), iou.min())
    # amax = iou.argmax()
    # print(np.unravel_index(amax, iou.shape))
    # print(anchors[0, amax])


def test_od_head():
    head = ObjectDetectionHead(
        in_channels=128,
        num_classes=5,
        stride=4,
    )

    gt = torch.tensor(
        [
            [
                [10, 20, 30, 0, 5],
                [50, 5, 25, 1, 6],
                [-100, -100, -100, -100, -100],
            ],
            [
                [30, 40, 50, 2, 6],
                [60, 70, 80, 3, 6],
                [69, 69, 69, 4, 6],
            ],
        ]
    )

    fm = torch.randn(2, 128, 32, 32, 32)
    output = head(fm, labels=gt)
    print(output.loss)


def test_batch_pairwise_keypoints_iou():
    pred_pt = torch.tensor(
        [
            [[49, 49, 149], [48, 48, 148], [47, 47, 147], [46, 46, 146], [110, 96, 95], [15, 5, 5], [168, 190, 201]],
        ]
    )

    true_pt = torch.tensor(
        [
            [[50, 50, 150], [100, 100, 100], [10, 10, 20], [200, 200, 200]],
        ]
    )

    sigmas = torch.tensor(
        [
            [5, 5, 5, 5],
        ]
    )

    sim = batch_pairwise_keypoints_iou(pred_pt, true_pt, sigmas)
    assert sim.shape[1] == true_pt.shape[1]
    assert sim.shape[2] == pred_pt.shape[1]
    sim_np = sim.detach().numpy()
    print(sim_np)


def test_weighted_average():
    scores_volume = torch.randn((5, 192, 128, 128))

    weight = AccumulatedObjectDetectionPredictionContainer.compute_weight_matrix(scores_volume, sigma=15)

    weight = weight.numpy()

    print(weight.min(), weight.max())

    fig, ax = plt.subplots(3, 3, figsize=(16, 16))
    ax[0, 0].imshow(weight[0], vmin=0, vmax=1)
    ax[0, 1].imshow(weight[weight.shape[0] // 2], vmin=0, vmax=1)
    ax[0, 2].imshow(weight[-1], vmin=0, vmax=1)

    ax[1, 0].imshow(weight[:, 0, :], vmin=0, vmax=1)
    ax[1, 1].imshow(weight[:, weight.shape[1] // 2, :], vmin=0, vmax=1)
    ax[1, 2].imshow(weight[:, -1, :], vmin=0, vmax=1)

    ax[2, 0].imshow(weight[:, :, 0], vmin=0, vmax=1)
    ax[2, 1].imshow(weight[:, :, weight.shape[2] // 2], vmin=0, vmax=1)
    im = ax[2, 2].imshow(weight[:, :, -1], vmin=0, vmax=1)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.show()

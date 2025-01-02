import math

import einops
import numpy as np
import torch

from cryoet.modelling.detection.detection_head import ObjectDetectionHead
from cryoet.modelling.detection.functional import (
    anchors_for_offsets_feature_map,
    iou_loss,
    keypoint_similarity,
    decode_detections,
)
from cryoet.modelling.detection.task_aligned_assigner import batch_pairwise_keypoints_iou, TaskAlignedAssigner


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

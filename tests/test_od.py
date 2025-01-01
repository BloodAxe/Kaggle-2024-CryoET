import torch

from cryoet.modelling.od_head import ObjectDetectionHead
from cryoet.modelling.task_aligned_assigner import batch_pairwise_keypoints_iou


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

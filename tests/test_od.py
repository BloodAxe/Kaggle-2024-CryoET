import torch

from cryoet.modelling.od_head import batch_pairwise_keypoints_iou


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
    assert sim.shape[1] == pred_pt.shape[1]
    assert sim.shape[2] == true_pt.shape[1]
    sim_np = sim.detach().numpy()
    print(sim_np)

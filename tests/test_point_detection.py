from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from cryoet.data.parsers import (
    get_volume_and_objects,
    ANGSTROMS_IN_PIXEL,
    NUM_CLASSES,
    CLASS_LABEL_TO_CLASS_NAME,
)
from cryoet.data.point_detection_dataset import (
    encode_centers_to_heatmap,
    decoder_centers_from_heatmap,
    centernet_gaussian_3d,
)
from cryoet.metric import score_submission

DATA_ROOT = Path(__file__).parent.parent / "data" / "czii-cryo-et-object-identification"
TRAIN_DATA_DIR = DATA_ROOT / "train" / "static" / "ExperimentRuns"


def test_gaussian():
    radiuses = [60, 90, 150, 130, 135]

    for r in radiuses[:1]:
        d = (r / 10.0) * 2 + 1
        g = centernet_gaussian_3d((d, d, d), sigma=d / 6.0)

        slice = g[g.shape[0] // 2]
        plt.figure(figsize=(16, 16))
        plt.imshow(slice)
        plt.title(f"Radius: {r}")

        # Show values
        for i in range(slice.shape[0]):
            for j in range(slice.shape[1]):
                plt.text(j, i, f"{slice[i, j]:.3f}", ha="center", va="center", color="black")

        plt.tight_layout()
        plt.show()

    for r in radiuses[:1]:
        d = (r / 10.0) * 2 + 1
        g = centernet_gaussian_3d((d, d, d), sigma=d / 3.0)

        slice = g[g.shape[0] // 2]
        plt.figure(figsize=(16, 16))
        plt.imshow(slice)
        plt.title(f"Radius: {r}")

        # Show values
        for i in range(slice.shape[0]):
            for j in range(slice.shape[1]):
                plt.text(j, i, f"{slice[i, j]:.3f}", ha="center", va="center", color="black")

        plt.tight_layout()
        plt.show()


def test_encode_decode():
    study_name = "TS_5_4"

    volume_data, object_centers, object_labels, object_radii = get_volume_and_objects(
        root_dir=DATA_ROOT,
        study_name="TS_5_4",
        mode="denoised",
        split="train",
    )

    # object_centers = object_centers[:2]
    # object_labels = object_labels[:2]
    # object_radii = object_radii[:2]

    solution = defaultdict(list)
    for i, (center, label, radius) in enumerate(zip(object_centers, object_labels, object_radii)):
        solution["id"].append(i)
        solution["experiment"].append(study_name)
        solution["particle_type"].append(CLASS_LABEL_TO_CLASS_NAME[label])
        solution["x"].append(float(center[0]))
        solution["y"].append(float(center[1]))
        solution["z"].append(float(center[2]))

    solution = pd.DataFrame.from_dict(solution).set_index("id")
    print(score_submission(solution, solution, "id", distance_multiplier=0.5, beta=4))

    num_classes = NUM_CLASSES
    labels = encode_centers_to_heatmap(
        object_centers / ANGSTROMS_IN_PIXEL,
        object_labels,
        object_radii / ANGSTROMS_IN_PIXEL,
        shape=volume_data.shape,
        num_classes=num_classes,
    )

    assert labels.shape == (num_classes, *volume_data.shape)
    assert (labels == 1).sum() == len(object_centers)

    topk_scores, topk_clses, topk_coords = decoder_centers_from_heatmap(
        torch.from_numpy(labels).unsqueeze(0),
    )

    topk_scores = topk_scores[0].numpy()
    topk_clses = topk_clses[0].numpy()
    topk_coords = topk_coords[0].numpy()

    print(topk_scores.shape)

    keep_mask = topk_scores > 0.5
    topk_clses = topk_clses[keep_mask]
    topk_coords = topk_coords[keep_mask] * ANGSTROMS_IN_PIXEL

    submission = defaultdict(list)
    for i, (cls, coord) in enumerate(zip(topk_clses, topk_coords)):
        submission["id"].append(i)
        submission["experiment"].append(study_name)
        submission["particle_type"].append(CLASS_LABEL_TO_CLASS_NAME[cls])
        submission["x"].append(float(coord[0]))
        submission["y"].append(float(coord[1]))
        submission["z"].append(float(coord[2]))

    submission = pd.DataFrame.from_dict(submission).set_index("id")

    print(solution.head())
    print(submission.head())
    print(score_submission(solution, submission, "id", distance_multiplier=0.5, beta=4))

import dataclasses
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch

from cryoet.data.cross_validation import split_data_into_folds
from cryoet.data.parsers import CLASS_LABEL_TO_CLASS_NAME, read_annotated_volume
from cryoet.inference.predict_volume import predict_volume
from cryoet.metric import score_submission


@dataclasses.dataclass
class SearchSpace:
    valid_depth_tiles: Tuple[int, ...] = (1, 2)
    valid_spatial_tiles: Tuple[int, ...] = (5, 6, 7, 8)
    use_weighted_average: Tuple[bool, ...] = (True, False)
    use_centernet_nms: Tuple[bool, ...] = (True, False)
    use_single_label_per_anchor: Tuple[bool, ...] = (True, False)
    iou_threshold: Tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def main(
    *checkpoints,
    output_strides: List[int],
    fold: int,
    data_path,
    torch_dtype=torch.float16,
    use_weighted_average: bool = False,
    valid_depth_window_size=192,
    valid_depth_window_step=62,
    valid_spatial_window_size=192,
    valid_spatial_window_step=79,
    split="train",
    iou_threshold=0.6,
    device="cuda",
):
    models = [torch.jit.load(checkpoint, map_location="cuda").to(torch_dtype) for checkpoint in checkpoints]

    data_path = Path(data_path)

    _, valid_studies = split_data_into_folds(data_path)[fold]

    solution = defaultdict(list)
    submissions = []

    window_size = (valid_depth_window_size, valid_spatial_window_size, valid_spatial_window_size)
    window_step = (valid_depth_window_step, valid_spatial_window_step, valid_spatial_window_step)

    for study_name in valid_studies:
        sample = read_annotated_volume(root=data_path, study=study_name, mode="denoised", split="train", use_6_classes=False)

        for i, (center, label, radius) in enumerate(zip(sample.centers, sample.labels, sample.radius)):
            solution["experiment"].append(sample.study)
            solution["particle_type"].append(CLASS_LABEL_TO_CLASS_NAME[label])
            solution["x"].append(float(center[0]))
            solution["y"].append(float(center[1]))
            solution["z"].append(float(center[2]))

        study_sub = predict_volume(
            volume=sample.volume,
            models=models,
            output_strides=output_strides,
            window_size=window_size,
            window_step=window_step,
            device=device,
            use_weighted_average=use_weighted_average,
            study_name=study_name,
            score_thresholds=score_thresholds,
            iou_threshold=iou_threshold,
            batch_size=1,
            num_workers=4,
            torch_dtype=torch_dtype,
            use_centernet_nms=use_centernet_nms,
            use_single_label_per_anchor=use_single_label_per_anchor,
        )

        submissions.append(study_sub)

    solution = pd.DataFrame.from_dict(solution)
    submission = pd.concat(submissions)

    # Add zero-based id column
    submission["id"] = range(len(submission))

    overall_score, scores_dict = score_submission(
        solution,
        submission,
        "id",
        distance_multiplier=0.5,
        beta=4,
    )

    print(f"Overall score: {overall_score}")
    print("Scores per particle type:")
    for particle_type, score in scores_dict.items():
        print(f"{particle_type}: {score}")


if __name__ == "__fire__":
    from fire import Fire

    Fire(main)

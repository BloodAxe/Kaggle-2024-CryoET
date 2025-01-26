import dataclasses
import os
from collections import defaultdict
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv

from cryoet.data.cross_validation import split_data_into_folds
from cryoet.data.parsers import CLASS_LABEL_TO_CLASS_NAME, read_annotated_volume, TARGET_5_CLASSES
from cryoet.ensembling import infer_fold
from cryoet.inference.predict_volume import (
    predict_scores_offsets_from_volume,
    postprocess_scores_offsets_into_submission,
)
from cryoet.metric import score_submission


@dataclasses.dataclass
class PredictionParams:
    valid_depth_window_size: int
    valid_spatial_window_size: int
    valid_depth_tiles: int
    valid_spatial_tiles: int
    use_weighted_average: bool

    use_z_flip_tta: bool
    use_y_flip_tta: bool
    use_x_flip_tta: bool


@dataclasses.dataclass
class PostprocessingParams:
    use_centernet_nms: bool
    use_single_label_per_anchor: bool

    iou_threshold: float
    pre_nms_top_k: int

    min_score_threshold: float


@torch.jit.optimized_execution(False)
def main(
    *checkpoints,
    output_dir: str,
    data_path: str = None,
    output_strides: List[int] = (2,),
    torch_dtype=torch.float16,
    valid_depth_window_size=128,
    valid_spatial_window_size=128,
    valid_depth_tiles=3,
    valid_spatial_tiles=8,
    iou_threshold=0.85,
    pre_nms_top_k=16536,
    device="cuda",
):
    if data_path is None:
        data_path = os.environ.get("CRYOET_DATA_ROOT")

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    data_path = Path(data_path)

    class_names = [cls["name"] for cls in TARGET_5_CLASSES]

    models_per_fold = defaultdict(list)
    for checkpoint_path in checkpoints:
        fold = infer_fold(checkpoint_path)
        models_per_fold[fold].append(checkpoint_path)

    summary_file = open(output_dir / "summary.txt", "w")

    score_thresholds = None
    oof_per_class_scores = []
    oof_averaged_score = []
    oof_best_threshold_per_class = []

    folds = list(sorted(list(models_per_fold.keys())))
    for fold in folds:
        print(f"Evaluating fold {fold}")
        checkpoints = models_per_fold[fold]
        per_class_scores, score_thresholds, best_threshold_per_class, best_score_per_class, averaged_score = (
            evaluate_models_on_fold(
                checkpoints=checkpoints,
                fold=fold,
                data_path=data_path,
                prediction_params=PredictionParams(
                    valid_depth_window_size=valid_depth_window_size,
                    valid_spatial_window_size=valid_spatial_window_size,
                    valid_depth_tiles=valid_depth_tiles,
                    valid_spatial_tiles=valid_spatial_tiles,
                    use_weighted_average=True,
                    use_z_flip_tta=False,
                    use_y_flip_tta=False,
                    use_x_flip_tta=False,
                ),
                postprocess_hparams=PostprocessingParams(
                    use_centernet_nms=True,
                    use_single_label_per_anchor=False,
                    iou_threshold=iou_threshold,
                    pre_nms_top_k=pre_nms_top_k,
                    min_score_threshold=0.05,
                ),
                output_strides=output_strides,
                device=device,
                torch_dtype=torch_dtype,
            )
        )
        oof_per_class_scores.append(per_class_scores)
        oof_averaged_score.append(averaged_score)
        oof_best_threshold_per_class.append(best_threshold_per_class)

        summary_file.write(f"Fold {fold}\n")
        summary_file.write(f"Per class thresholds: {best_threshold_per_class}\n")
        summary_file.write(f"Per class scores:     {best_score_per_class}\n")
        summary_file.write(f"Averaged score:       {averaged_score}\n")
        summary_file.write("\n")

    # Now do something fancy with the OOF scores
    # Compute mean of the best thresholds
    oof_per_class_scores = np.stack(oof_per_class_scores)  # [fold, threshold, class]
    oof_best_threshold_per_class = np.stack(oof_best_threshold_per_class)  # [fold, class]

    # Simple and probably wrong
    mean_of_thresholds = np.mean(oof_best_threshold_per_class, axis=0)
    median_of_thresholds = np.median(oof_best_threshold_per_class, axis=0)

    # More smart - average the oof per class scores and find the best threshold
    avg_per_class_scores = oof_per_class_scores.mean(axis=0)  # [threshold, class]
    max_scores_index = np.argmax(avg_per_class_scores, axis=0)  # [class]
    curve_averaged_thresholds = score_thresholds[max_scores_index]

    print("Mean of thresholds       ", np.array2string(mean_of_thresholds, separator=",", precision=3))
    print("Median of thresholds     ", np.array2string(median_of_thresholds, separator=",", precision=3))
    print("Curve averaged thresholds", np.array2string(curve_averaged_thresholds, separator=",", precision=3))

    num_folds = len(folds)
    f, ax = plt.subplots(1, num_folds + 1, figsize=(5 * (num_folds + 1), 5))

    for i, fold in enumerate(folds):
        for j, cls in enumerate(TARGET_5_CLASSES):
            ax[i].plot(score_thresholds, oof_per_class_scores[i, :, j], label=cls["name"])
        ax[i].set_title(f"Fold {fold}")
        ax[i].set_xlabel("Score threshold")
        ax[i].set_ylabel("Score")
        ax[i].legend(class_names)

    for j, cls in enumerate(TARGET_5_CLASSES):
        ax[-1].plot(score_thresholds, avg_per_class_scores[:, j], label=cls["name"])
    ax[-1].set_title("Average")
    ax[-1].set_xlabel("Score threshold")
    ax[-1].set_ylabel("Score")
    ax[-1].legend(class_names)

    f.tight_layout()
    f.savefig(output_dir / "score_thresholds.png")
    f.show()


def evaluate_models_on_fold(
    checkpoints,
    fold,
    data_path: Path,
    prediction_params: PredictionParams,
    postprocess_hparams: PostprocessingParams,
    output_strides=(2,),
    device="cuda",
    torch_dtype=torch.float16,
):
    models = [torch.jit.load(checkpoint, map_location=device).to(torch_dtype) for checkpoint in checkpoints]

    _, valid_studies = split_data_into_folds(data_path / "train" / "static" / "ExperimentRuns")[fold]

    window_size = (
        prediction_params.valid_depth_window_size,
        prediction_params.valid_spatial_window_size,
        prediction_params.valid_spatial_window_size,
    )

    solution = defaultdict(list)

    valid_samples = []
    for study_name in valid_studies:
        sample = read_annotated_volume(root=data_path, study=study_name, mode="denoised", split="train", use_6_classes=False)
        valid_samples.append(sample)

        for i, (center, label, radius) in enumerate(zip(sample.centers, sample.labels, sample.radius)):
            solution["experiment"].append(sample.study)
            solution["particle_type"].append(CLASS_LABEL_TO_CLASS_NAME[label])
            solution["x"].append(float(center[0]))
            solution["y"].append(float(center[1]))
            solution["z"].append(float(center[2]))

    solution = pd.DataFrame.from_dict(solution)

    pred_scores = []
    pred_offsets = []

    for sample in valid_samples:
        scores, offsets = predict_scores_offsets_from_volume(
            volume=sample.volume,
            models=models,
            output_strides=output_strides,
            window_size=window_size,
            tiles_per_dim=(
                prediction_params.valid_depth_tiles,
                prediction_params.valid_spatial_tiles,
                prediction_params.valid_spatial_tiles,
            ),
            batch_size=1,
            num_workers=0,
            torch_dtype=torch_dtype,
            use_weighted_average=prediction_params.use_weighted_average,
            device=device,
            study_name=sample.study,
            use_z_flip_tta=prediction_params.use_z_flip_tta,
            use_y_flip_tta=prediction_params.use_y_flip_tta,
            use_x_flip_tta=prediction_params.use_x_flip_tta,
        )
        pred_scores.append(scores)
        pred_offsets.append(offsets)

    submission = postprocess_into_submission(pred_scores, pred_offsets, postprocess_hparams, output_strides, valid_samples)

    class_names = [cls["name"] for cls in TARGET_5_CLASSES]
    return compute_optimal_thresholds(class_names, solution, submission)


def postprocess_into_submission(
    pred_scores, pred_offsets, postprocess_hparams: PostprocessingParams, output_strides, valid_samples
):
    submissions = []
    for (
        sample,
        scores,
        offsets,
    ) in zip(valid_samples, pred_scores, pred_offsets):
        submission_for_sample = postprocess_scores_offsets_into_submission(
            scores=scores,
            offsets=offsets,
            output_strides=output_strides,
            study_name=sample.study,
            iou_threshold=postprocess_hparams.iou_threshold,
            score_thresholds=postprocess_hparams.min_score_threshold,
            pre_nms_top_k=postprocess_hparams.pre_nms_top_k,
            use_centernet_nms=postprocess_hparams.use_centernet_nms,
            use_single_label_per_anchor=postprocess_hparams.use_single_label_per_anchor,
        )
        submissions.append(submission_for_sample)
    submission = pd.concat(submissions)
    submission["id"] = range(len(submission))
    return submission


def compute_optimal_thresholds(class_names, solution, submission):
    weights = {
        "apo-ferritin": 1,
        "beta-amylase": 0,
        "beta-galactosidase": 2,
        "ribosome": 1,
        "thyroglobulin": 2,
        "virus-like-particle": 1,
    }

    score_details = []
    score_thresholds = np.linspace(0.05, 0.9, num=171, dtype=np.float32)
    for score_threshold in score_thresholds:
        keep_mask = submission["score"] >= score_threshold
        submission_filtered = submission[keep_mask]
        _, scores_dict = score_submission(
            solution=solution.copy(),
            submission=submission_filtered.copy(),
            row_id_column_name="id",
            distance_multiplier=0.5,
            beta=4,
        )
        score_details.append(scores_dict)

    per_class_scores = []
    for scores_dict in score_details:
        per_class_scores.append([scores_dict[k] for k in class_names])
    per_class_scores = np.array(per_class_scores)  # [threshold, class]

    best_index_per_class = np.argmax(per_class_scores, axis=0)  # [class]
    best_threshold_per_class = np.array([score_thresholds[i] for i in best_index_per_class])  # [class]
    best_score_per_class = np.array([per_class_scores[i, j] for j, i in enumerate(best_index_per_class)])  # [class]
    averaged_score = np.sum([weights[k] * best_score_per_class[i] for i, k in enumerate(class_names)]) / sum(weights.values())

    return per_class_scores, score_thresholds, best_threshold_per_class, best_score_per_class, averaged_score


if __name__ == "__main__":
    from fire import Fire

    load_dotenv()
    Fire(main)

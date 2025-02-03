import dataclasses
import os
from collections import defaultdict
from pathlib import Path
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv

from cryoet.data.cross_validation import split_data_into_folds
from cryoet.data.parsers import CLASS_LABEL_TO_CLASS_NAME, read_annotated_volume, TARGET_5_CLASSES
from cryoet.ensembling import infer_fold, jit_model_from_checkpoint
from cryoet.inference.predict_volume import (
    predict_scores_offsets_from_volume,
    postprocess_scores_offsets_into_submission,
)
from cryoet.metric import score_submission
from cryoet.training.visualization import render_heatmap


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
    valid_depth_window_size=192,
    valid_spatial_window_size=128,
    valid_depth_tiles=1,
    valid_spatial_tiles=9,
    iou_threshold=0.85,
    min_score_threshold=0.05,
    use_weighted_average=True,
    pre_nms_top_k=16536,
    use_single_label_per_anchor=False,
    validate_on_x_flips=False,
    validate_on_y_flips=False,
    validate_on_z_flips=False,
    validate_on_rot90=True,
    device="cuda",
):
    function_args = locals()
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

    summary_file = open(output_dir / "summary.txt", "a")

    score_thresholds = None
    oof_per_class_scores = []
    oof_averaged_score = []
    oof_best_threshold_per_class = []

    summary_file.write(f"Ensemble of {len(checkpoints)} models\n")
    for fold, checkpoints in models_per_fold.items():
        summary_file.write(f"Fold {fold}: {checkpoints}\n")

    summary_file.write("\n")
    summary_file.write("Hyperparameters\n")
    for k, v in function_args.items():
        summary_file.write(f"{k}: {v}\n")

    folds = list(sorted(list(models_per_fold.keys())))
    for fold in folds:
        print(f"Evaluating fold {fold}")
        checkpoints = models_per_fold[fold]
        _, valid_studies = split_data_into_folds(data_path / "train" / "static" / "ExperimentRuns")[fold]

        per_class_scores, score_thresholds, best_threshold_per_class, best_score_per_class, averaged_score = (
            evaluate_models_on_fold(
                checkpoints=checkpoints,
                valid_studies=valid_studies,
                data_path=data_path,
                output_dir=output_dir,
                validate_on_x_flips=validate_on_x_flips,
                validate_on_y_flips=validate_on_y_flips,
                validate_on_z_flips=validate_on_z_flips,
                validate_on_rot90=validate_on_rot90,
                prediction_params=PredictionParams(
                    valid_depth_window_size=valid_depth_window_size,
                    valid_spatial_window_size=valid_spatial_window_size,
                    valid_depth_tiles=valid_depth_tiles,
                    valid_spatial_tiles=valid_spatial_tiles,
                    use_weighted_average=use_weighted_average,
                    use_z_flip_tta=False,
                    use_y_flip_tta=False,
                    use_x_flip_tta=False,
                ),
                postprocess_hparams=PostprocessingParams(
                    use_centernet_nms=True,
                    use_single_label_per_anchor=use_single_label_per_anchor,
                    iou_threshold=iou_threshold,
                    pre_nms_top_k=pre_nms_top_k,
                    min_score_threshold=min_score_threshold,
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

    print("Mean of thresholds        ", np.array2string(mean_of_thresholds, separator=", ", precision=3))
    print("Median of thresholds      ", np.array2string(median_of_thresholds, separator=", ", precision=3))
    print("Curve averaged thresholds ", np.array2string(curve_averaged_thresholds, separator=", ", precision=3))

    summary_file.write(f"Mean of thresholds        {np.array2string(mean_of_thresholds, separator=', ', precision=3)}\n")
    summary_file.write(f"Median of thresholds      {np.array2string(median_of_thresholds, separator=', ', precision=3)}\n")
    summary_file.write(f"Curve averaged thresholds {np.array2string(curve_averaged_thresholds, separator=', ', precision=3)}\n")
    summary_file.write(f"CV score:                 {np.mean(oof_averaged_score):.4f} std: {np.std(oof_averaged_score):.4f}\n")

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
    f.savefig(
        output_dir
        / f"plot_rot{validate_on_rot90}_z{validate_on_z_flips}_y{validate_on_y_flips}_x{validate_on_x_flips}_{valid_depth_window_size}x{valid_depth_tiles}_{valid_spatial_window_size}x{valid_spatial_tiles}_slpa{use_single_label_per_anchor}.png"
    )
    f.show()

    # Now, do the risky thing - use whole train set to find the best thresholds
    train_studies, valid_studies = split_data_into_folds(data_path / "train" / "static" / "ExperimentRuns")[0]
    all_studies = train_studies + valid_studies

    per_class_scores, score_thresholds, best_threshold_per_class, best_score_per_class, averaged_score = evaluate_models_on_fold(
        checkpoints=checkpoints,
        valid_studies=all_studies,
        data_path=data_path,
        output_dir=output_dir,
        validate_on_x_flips=validate_on_x_flips,
        validate_on_y_flips=validate_on_y_flips,
        validate_on_z_flips=validate_on_z_flips,
        validate_on_rot90=validate_on_rot90,
        prediction_params=PredictionParams(
            valid_depth_window_size=valid_depth_window_size,
            valid_spatial_window_size=valid_spatial_window_size,
            valid_depth_tiles=valid_depth_tiles,
            valid_spatial_tiles=valid_spatial_tiles,
            use_weighted_average=use_weighted_average,
            use_z_flip_tta=False,
            use_y_flip_tta=False,
            use_x_flip_tta=False,
        ),
        postprocess_hparams=PostprocessingParams(
            use_centernet_nms=True,
            use_single_label_per_anchor=use_single_label_per_anchor,
            iou_threshold=iou_threshold,
            pre_nms_top_k=pre_nms_top_k,
            min_score_threshold=min_score_threshold,
        ),
        output_strides=output_strides,
        device=device,
        torch_dtype=torch_dtype,
    )

    print("Curve averaged thresholds ", np.array2string(curve_averaged_thresholds, separator=", ", precision=3))
    summary_file.write(f"Whole train eval results\n")
    summary_file.write(f"Thresholds      {np.array2string(best_threshold_per_class, separator=', ', precision=3)}\n")
    summary_file.write(f"Scores          {np.array2string(best_score_per_class, separator=', ', precision=3)}\n")
    summary_file.write(f"Averaged score: {averaged_score:.4f}\n")


def save_scores_heatmap(scores, output_dir, study_name):
    if isinstance(scores, list):
        scores = scores[0]

    heatmap_image = render_heatmap(scores)
    heatmap_image = cv2.cvtColor(heatmap_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_dir / f"{study_name}.png", heatmap_image)


def evaluate_models_on_fold(
    checkpoints,
    valid_studies,
    data_path: Path,
    output_dir: Path,
    validate_on_x_flips,
    validate_on_y_flips,
    validate_on_z_flips,
    validate_on_rot90,
    prediction_params: PredictionParams,
    postprocess_hparams: PostprocessingParams,
    output_strides=(2,),
    device="cuda",
    torch_dtype=torch.float16,
):
    models = [
        jit_model_from_checkpoint(
            checkpoint,
            torch_device=device,
            torch_dtype=torch_dtype,
            window_size=(
                prediction_params.valid_depth_window_size,
                prediction_params.valid_spatial_window_size,
                prediction_params.valid_spatial_window_size,
            ),
        )
        for checkpoint in checkpoints
    ]

    window_size = (
        prediction_params.valid_depth_window_size,
        prediction_params.valid_spatial_window_size,
        prediction_params.valid_spatial_window_size,
    )

    study_names = []
    pred_scores = []
    pred_offsets = []
    solution = defaultdict(list)

    for study_name in valid_studies:
        sample = read_annotated_volume(
            root=data_path, study=study_name, mode="denoised", split="train", use_6_classes=False, normalization="minmax"
        )

        x_options = [False, True] if validate_on_x_flips else [False]
        y_options = [False, True] if validate_on_y_flips else [False]
        z_options = [False, True] if validate_on_z_flips else [False]
        rot_options = [0, 1, 2, 3] if validate_on_rot90 else [0]

        for rot in rot_options:
            for x_flip in x_options:
                for y_flip in y_options:
                    for z_flip in z_options:
                        print("Flipping", x_flip, y_flip, z_flip)
                        maybe_flipped_sample = sample.rot90(rot).flip(x_flip, y_flip, z_flip)

                        for i, (center, label, radius) in enumerate(
                            zip(maybe_flipped_sample.centers, maybe_flipped_sample.labels, maybe_flipped_sample.radius)
                        ):
                            solution["experiment"].append(maybe_flipped_sample.study)
                            solution["particle_type"].append(CLASS_LABEL_TO_CLASS_NAME[label])
                            solution["x"].append(float(center[0]))
                            solution["y"].append(float(center[1]))
                            solution["z"].append(float(center[2]))

                        study_names.append(maybe_flipped_sample.study)

                        scores, offsets = predict_scores_offsets_from_volume(
                            volume=maybe_flipped_sample.volume,
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
                            study_name=maybe_flipped_sample.study,
                            use_z_flip_tta=prediction_params.use_z_flip_tta,
                            use_y_flip_tta=prediction_params.use_y_flip_tta,
                            use_x_flip_tta=prediction_params.use_x_flip_tta,
                        )

                        save_scores_heatmap(scores, output_dir, maybe_flipped_sample.study)
                        pred_scores.append(scores)
                        pred_offsets.append(offsets)

    solution = pd.DataFrame.from_dict(solution)
    submission = postprocess_into_submission(pred_scores, pred_offsets, postprocess_hparams, output_strides, study_names)

    class_names = [cls["name"] for cls in TARGET_5_CLASSES]
    return compute_optimal_thresholds(class_names, solution, submission)


def postprocess_into_submission(
    pred_scores, pred_offsets, postprocess_hparams: PostprocessingParams, output_strides, study_names
):
    submissions = []
    for (
        study_name,
        scores,
        offsets,
    ) in zip(study_names, pred_scores, pred_offsets):
        submission_for_sample = postprocess_scores_offsets_into_submission(
            scores=scores,
            offsets=offsets,
            output_strides=output_strides,
            study_name=study_name,
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
    score_thresholds = np.linspace(0.01, 0.9, num=171, dtype=np.float32)
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

import dataclasses
import os
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from tqdm import tqdm

from cryoet.data.cross_validation import split_data_into_folds
from cryoet.data.parsers import CLASS_LABEL_TO_CLASS_NAME, read_annotated_volume, TARGET_5_CLASSES
from cryoet.ensembling_eval import (
    plot_2d_score_centernet_single_label,
    plot_2d_score_spatial_depth,
    plot_max_score_vs_iou_threshold,
    plot_score_distribution,
)
from cryoet.inference.predict_volume import (
    predict_scores_offsets_from_volume,
    postprocess_scores_offsets_into_submission,
)
from cryoet.metric import score_submission


@dataclasses.dataclass
class PredictionSearchSpace:
    valid_depth_tiles: int | Tuple[int, ...] = dataclasses.field(default=(1,))
    valid_spatial_tiles: int | Tuple[int, ...] = dataclasses.field(default=(7,))
    use_weighted_average: bool | Tuple[bool, ...] = dataclasses.field(default=(True,))

    def product(self):
        for valid_depth_tile in self.valid_depth_tiles:
            for valid_spatial_tile in self.valid_spatial_tiles:
                for use_weighted_average in self.use_weighted_average:
                    yield PredictionSearchSpace(
                        valid_depth_tiles=valid_depth_tile,
                        valid_spatial_tiles=valid_spatial_tile,
                        use_weighted_average=use_weighted_average,
                    )


@dataclasses.dataclass
class PostprocessingSearchSpace:
    use_centernet_nms: bool | Tuple[bool, ...] = (True,)
    use_single_label_per_anchor: bool | Tuple[bool, ...] = (True, False)

    iou_threshold: float | Tuple[float, ...] = (0.85,)
    pre_nms_top_k: int | Tuple[int, ...] = (16536,)  # Does not seems to influence at all (4096, 8192, 16536)

    min_score_threshold: float | Tuple[float, ...] = (0.05,)

    def product(self):
        for use_centernet_nms in self.use_centernet_nms:
            for use_single_label_per_anchor in self.use_single_label_per_anchor:
                for iou_threshold in self.iou_threshold:
                    for pre_nms_top_k in self.pre_nms_top_k:
                        for min_score_threshold in self.min_score_threshold:
                            yield PostprocessingSearchSpace(
                                use_centernet_nms=use_centernet_nms,
                                use_single_label_per_anchor=use_single_label_per_anchor,
                                iou_threshold=iou_threshold,
                                pre_nms_top_k=pre_nms_top_k,
                                min_score_threshold=min_score_threshold,
                            )


@torch.jit.optimized_execution(False)
def main(
    *checkpoints,
    output_dir: str,
    fold: int,
    data_path: str = None,
    output_strides: List[int] = (2,),
    torch_dtype=torch.float16,
    valid_depth_window_size=192,
    valid_spatial_window_size=128,
    device="cuda",
):
    if data_path is None:
        data_path = os.environ.get("CRYOET_DATA_ROOT")

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    models = [torch.jit.load(checkpoint, map_location=device).to(torch_dtype) for checkpoint in checkpoints]

    data_path = Path(data_path)

    _, valid_studies = split_data_into_folds(data_path / "train" / "static" / "ExperimentRuns")[fold]

    solution = defaultdict(list)

    window_size = (valid_depth_window_size, valid_spatial_window_size, valid_spatial_window_size)

    final_dataframe = defaultdict(list)

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

    prediction_search_space = list(PredictionSearchSpace().product())
    postprocess_search_space = list(PostprocessingSearchSpace().product())

    for prediction_hparams in tqdm(prediction_search_space, desc="Prediction search space", position=0):

        pred_scores = []
        pred_offsets = []

        for sample in valid_samples:
            scores, offsets = predict_scores_offsets_from_volume(
                volume=sample.volume,
                models=models,
                output_strides=output_strides,
                window_size=window_size,
                tiles_per_dim=(
                    prediction_hparams.valid_depth_tiles,
                    prediction_hparams.valid_spatial_tiles,
                    prediction_hparams.valid_spatial_tiles,
                ),
                batch_size=1,
                num_workers=0,
                torch_dtype=torch_dtype,
                use_weighted_average=prediction_hparams.use_weighted_average,
                device=device,
                study_name=sample.study,
            )
            pred_scores.append(scores)
            pred_offsets.append(offsets)

        for postprocess_hparams in tqdm(postprocess_search_space, position=1, desc="Postprocessing search space"):

            submission = postprocess_into_submission(
                pred_scores, pred_offsets, postprocess_hparams, output_strides, valid_samples
            )

            class_names = [cls["name"] for cls in TARGET_5_CLASSES]

            per_class_scores, score_thresholds, best_threshold_per_class, best_score_per_class, averaged_score = (
                compute_optimal_thresholds(class_names, solution, submission)
            )

            np.savez(
                per_class_scores=per_class_scores,
                score_thresholds=score_thresholds,
                best_threshold_per_class=best_threshold_per_class,
                best_score_per_class=best_score_per_class,
                averaged_score=averaged_score,
                file=output_dir
                / f"{fold=}_{prediction_hparams.valid_depth_tiles}_{prediction_hparams.valid_spatial_tiles}_{prediction_hparams.use_weighted_average}_{postprocess_hparams.use_centernet_nms}_{postprocess_hparams.use_single_label_per_anchor}_{postprocess_hparams.iou_threshold}_{postprocess_hparams.pre_nms_top_k}_{postprocess_hparams.min_score_threshold}.npz",
            )

            final_dataframe["valid_depth_tiles"].append(prediction_hparams.valid_depth_tiles)
            final_dataframe["valid_spatial_tiles"].append(prediction_hparams.valid_spatial_tiles)
            final_dataframe["use_weighted_average"].append(prediction_hparams.use_weighted_average)

            final_dataframe["use_centernet_nms"].append(postprocess_hparams.use_centernet_nms)
            final_dataframe["use_single_label_per_anchor"].append(postprocess_hparams.use_single_label_per_anchor)
            final_dataframe["iou_threshold"].append(postprocess_hparams.iou_threshold)
            final_dataframe["pre_nms_top_k"].append(postprocess_hparams.pre_nms_top_k)
            final_dataframe["min_score_threshold"].append(postprocess_hparams.min_score_threshold)

            for i, class_name in enumerate(class_names):
                final_dataframe[f"{class_name}_threshold"].append(best_threshold_per_class[i])
                final_dataframe[f"{class_name}_score"].append(best_score_per_class[i])

            final_dataframe["averaged_score"].append(averaged_score)

            print(f"Averaged score: {averaged_score}")

    final_dataframe = pd.DataFrame.from_dict(final_dataframe)
    final_dataframe.to_csv(output_dir / f"fold_{fold}_results.csv", index=True)
    final_dataframe.sort_values("averaged_score", ascending=False).to_markdown(
        output_dir / f"fold_{fold}_results.md", index=False
    )

    plot_2d_score_centernet_single_label(final_dataframe, output_dir / f"fold_{fold}_centernet_single_label.png")
    plot_2d_score_spatial_depth(final_dataframe, output_dir / f"fold_{fold}_spatial_depth.png")
    plot_max_score_vs_iou_threshold(final_dataframe, output_dir / f"fold_{fold}_iou_threshold.png")
    plot_score_distribution(final_dataframe, output_dir / f"fold_{fold}_score_distribution.png")


def postprocess_into_submission(pred_scores, pred_offsets, postprocess_hparams, output_strides, valid_samples):
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

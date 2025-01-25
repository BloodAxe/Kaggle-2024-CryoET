from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cryoet.data.functional import normalize_volume_to_unit_range
from cryoet.data.parsers import TARGET_SIGMAS, ANGSTROMS_IN_PIXEL, CLASS_LABEL_TO_CLASS_NAME
from cryoet.inference.dataset import TileDataset
from cryoet.modelling.detection.functional import decode_detections_with_nms
from cryoet.training.od_accumulator import AccumulatedObjectDetectionPredictionContainer


def infer_num_classes_from_logits(logits):
    if not torch.is_tensor(logits):
        logits = logits[0]

    b, c, d, h, w = logits.size()
    return int(c)


@torch.no_grad()
@torch.jit.optimized_execution(False)
def predict_volume(
    volume: np.ndarray,
    models: List,
    output_strides: List[int],
    window_size: Tuple[int, int, int],
    tiles_per_dim: Tuple[int, int, int],
    device: str,
    study_name: str,
    score_thresholds: Union[float, List[float]],
    iou_threshold,
    batch_size,
    num_workers,
    use_weighted_average,
    use_centernet_nms,
    use_single_label_per_anchor,
    torch_dtype,
    pre_nms_top_k,
):
    scores, offsets = predict_scores_offsets_from_volume(
        volume=volume,
        models=models,
        output_strides=output_strides,
        window_size=window_size,
        tiles_per_dim=tiles_per_dim,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        torch_dtype=torch_dtype,
        study_name=study_name,
        use_weighted_average=use_weighted_average,
    )

    submission = postprocess_scores_offsets_into_submission(
        scores=scores,
        offsets=offsets,
        iou_threshold=iou_threshold,
        output_strides=output_strides,
        score_thresholds=score_thresholds,
        study_name=study_name,
        use_centernet_nms=use_centernet_nms,
        use_single_label_per_anchor=use_single_label_per_anchor,
        pre_nms_top_k=pre_nms_top_k,
    )
    return submission


def postprocess_scores_offsets_into_submission(
    iou_threshold,
    offsets,
    output_strides,
    score_thresholds,
    scores,
    study_name,
    use_centernet_nms,
    use_single_label_per_anchor,
    pre_nms_top_k: int,
):
    topk_coords_px, topk_clses, topk_scores = decode_detections_with_nms(
        scores=scores,
        offsets=offsets,
        strides=output_strides,
        class_sigmas=TARGET_SIGMAS,
        min_score=score_thresholds,
        iou_threshold=iou_threshold,
        use_centernet_nms=use_centernet_nms,
        use_single_label_per_anchor=use_single_label_per_anchor,
        pre_nms_top_k=pre_nms_top_k,
    )
    topk_scores = topk_scores.float().cpu().numpy()
    top_coords = topk_coords_px.float().cpu().numpy() * ANGSTROMS_IN_PIXEL
    topk_clses = topk_clses.cpu().numpy()
    submission = dict(
        experiment=[],
        particle_type=[],
        score=[],
        x=[],
        y=[],
        z=[],
    )
    for cls, coord, score in zip(topk_clses, top_coords, topk_scores):
        submission["experiment"].append(study_name)
        submission["particle_type"].append(CLASS_LABEL_TO_CLASS_NAME[int(cls)])
        submission["score"].append(float(score))
        submission["x"].append(float(coord[0]))
        submission["y"].append(float(coord[1]))
        submission["z"].append(float(coord[2]))
    submission = pd.DataFrame.from_dict(submission)
    return submission


@torch.no_grad()
def predict_scores_offsets_from_volume(
    batch_size,
    device,
    models,
    num_workers,
    output_strides,
    study_name,
    torch_dtype,
    use_weighted_average,
    volume,
    window_size: Tuple[int, int, int],
    tiles_per_dim: Tuple[int, int, int],
    use_z_flip_tta: bool = False,
    use_y_flip_tta: bool = False,
    use_x_flip_tta: bool = False,
):
    torch.cuda.empty_cache()
    container = None
    volume = normalize_volume_to_unit_range(volume)
    # volume = normalize_volume_to_percentile_range(volume)
    ds = TileDataset(volume, window_size, tiles_per_dim, torch_dtype=torch_dtype)
    for tile_volume, tile_offsets in tqdm(
        DataLoader(ds, batch_size=batch_size, num_workers=num_workers, drop_last=False, pin_memory=True),
        desc=f"{study_name} {volume.shape}",
    ):
        tile_volume = tile_volume.to(device=device, non_blocking=True)

        for model in models:
            logits, offsets = model(tile_volume)
            probas = [x.sigmoid() for x in logits]

            if container is None:
                num_classes = infer_num_classes_from_logits(logits)
                print("Num classes", num_classes)

                container = AccumulatedObjectDetectionPredictionContainer.from_shape(
                    shape=volume.shape,
                    num_classes=num_classes,
                    window_size=window_size,
                    use_weighted_average=use_weighted_average,
                    strides=output_strides,
                    device=device,
                    dtype=torch_dtype,
                )

            container.accumulate_batch(probas, offsets, tile_offsets)

            if use_z_flip_tta:
                logits_flip, offsets_flip = model(z_flip_volume(tile_volume))
                probas = [z_flip_volume(x.sigmoid()) for x in logits_flip]
                offsets = [z_flip_offsets(x) for x in offsets_flip]
                container.accumulate_batch(probas, offsets, tile_offsets)

            if use_y_flip_tta:
                logits_flip, offsets_flip = model(y_flip_volume(tile_volume))
                probas = [y_flip_volume(x.sigmoid()) for x in logits_flip]
                offsets = [y_flip_offsets(x) for x in offsets_flip]
                container.accumulate_batch(probas, offsets, tile_offsets)

            if use_x_flip_tta:
                logits_flip, offsets_flip = model(x_flip_volume(tile_volume))
                probas = [x_flip_volume(x.sigmoid()) for x in logits_flip]
                offsets = [x_flip_offsets(x) for x in offsets_flip]
                container.accumulate_batch(probas, offsets, tile_offsets)

    scores, offsets = container.merge_()
    return scores, offsets


def flip_volume(volume, dim):
    """
    Flip the volume along the specified dimension.
    :param volume: Volume to flip. B C D H W shape
    :param dim: Dimension to flip
    """
    return volume.flip(dim)


def flip_offsets(offsets, dim):
    offsets_flip = torch.flip(offsets, [dim]).clone()
    offsets_flip[:, dim] *= -1  # Flip the z-offsets
    return offsets_flip


def z_flip_volume(volume):
    return flip_volume(volume, 2)


def z_flip_offsets(offsets):
    return flip_offsets(offsets, 2)


def y_flip_volume(volume):
    return flip_volume(volume, 3)


def y_flip_offsets(offsets):
    return flip_offsets(offsets, 3)


def x_flip_volume(volume):
    return flip_volume(volume, 4)


def x_flip_offsets(offsets):
    return flip_offsets(offsets, 4)

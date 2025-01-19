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
    window_step: Tuple[int, int, int],
    device: str,
    study_name: str,
    score_thresholds: Union[float, List[float]] = 0.05,
    iou_threshold=0.6,
    batch_size=8,
    num_workers=0,
    use_weighted_average=False,
    use_centernet_nms=True,
    use_single_label_per_anchor=True,
    torch_dtype=torch.float32,
):
    torch.cuda.empty_cache()
    container = None

    volume = normalize_volume_to_unit_range(volume)
    # volume = normalize_volume_to_percentile_range(volume)

    ds = TileDataset(volume, window_size, window_step, torch_dtype=torch_dtype)

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

    scores, offsets = container.merge_()

    topk_coords_px, topk_clses, topk_scores = decode_detections_with_nms(
        scores=scores,
        offsets=offsets,
        strides=output_strides,
        class_sigmas=TARGET_SIGMAS,
        min_score=score_thresholds,
        iou_threshold=iou_threshold,
        use_centernet_nms=use_centernet_nms,
        use_single_label_per_anchor=use_single_label_per_anchor,
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

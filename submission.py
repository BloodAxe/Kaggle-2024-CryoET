import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch.jit
from tqdm import tqdm

from cryoet.data.parsers import ANGSTROMS_IN_PIXEL, CLASS_LABEL_TO_CLASS_NAME, get_volume, NUM_CLASSES
from cryoet.data.heatmap.point_detection_dataset import (
    compute_tiles,
    normalize_volume_to_unit_range,
    decoder_centers_from_heatmap,
)
from cryoet.training.point_detection_module import AccumulatedPredictionContainer


@torch.no_grad()
@torch.jit.optimized_execution(False)
def predict_volume(
    volume: np.ndarray, models: list, window_size: int, stride: int, device: str, study_name: str, top_k=512, threshold=0.05
):
    container = AccumulatedPredictionContainer.from_shape(volume.shape, NUM_CLASSES, device=device)

    tiles = list(compute_tiles(volume.shape, window_size, stride))

    volume = normalize_volume_to_unit_range(volume)

    for tile in tqdm(tiles, desc=study_name):
        tile_volume = volume[tile[0], tile[1], tile[2]]
        pad_z = window_size - tile_volume.shape[0]
        pad_y = window_size - tile_volume.shape[1]
        pad_x = window_size - tile_volume.shape[2]

        tile_volume = np.pad(
            tile_volume,
            ((0, pad_z), (0, pad_y), (0, pad_x)),
            mode="constant",
            constant_values=0,
        )

        tile_volume = torch.from_numpy(tile_volume).unsqueeze(0).unsqueeze(0).float().to(device)
        tile_offsets = (tile[0].start, tile[1].start, tile[2].start)

        for model in models:
            prediction = model(tile_volume).sigmoid()
            container.accumulate(prediction[0], tile_offsets)

        container.probas /= container.counter
        container.probas.masked_fill_(container.counter == 0, 0.0)

    topk_scores, topk_clses, topk_coords_px = decoder_centers_from_heatmap(container.probas.unsqueeze(0), top_k=top_k)
    topk_scores = topk_scores[0].float().cpu().numpy()
    top_coords = topk_coords_px[0].float().cpu().numpy() * ANGSTROMS_IN_PIXEL
    topk_clses = topk_clses[0].cpu().numpy()

    submission = defaultdict(list)
    for cls, coord, score in zip(topk_clses, top_coords, topk_scores):
        submission["experiment"].append(study_name)
        submission["particle_type"].append(CLASS_LABEL_TO_CLASS_NAME[int(cls)])
        submission["score"].append(float(score))
        submission["x"].append(float(coord[0]))
        submission["y"].append(float(coord[1]))
        submission["z"].append(float(coord[2]))

    submission = pd.DataFrame.from_dict(submission)
    submission = submission[submission["score"] >= threshold]
    return submission


def main_inference_entry_point(
    *checkpoints,
    data_path: str,
    split: str = "test",
    window_size: int = 96,
    stride: int = 32,
    top_k: int = 512,
    threshold: float = 0.05,
    device="cuda",
):
    models = [torch.jit.load(checkpoint, map_location="cuda") for checkpoint in checkpoints]

    path = Path(data_path)
    studies_path = path / split / "static" / "ExperimentRuns"

    studies = list(sorted(os.listdir(studies_path)))

    submissions = []

    for study_name in studies:
        study_volume = get_volume(
            root_dir=path,
            study_name=study_name,
            mode="denoised",
            split=split,
        )

        study_sub = predict_volume(
            study_volume,
            models,
            window_size=window_size,
            stride=stride,
            device=device,
            study_name=study_name,
            top_k=top_k,
            threshold=threshold,
        )

        submissions.append(study_sub)

    submission = pd.concat(submissions)
    # Add zero-based id column
    submission["id"] = range(len(submission))
    return submission


if __name__ == "__main__":
    from fire import Fire

    sub = Fire(main_inference_entry_point)
    sub.to_csv("submission.csv", index=False)

from typing import Tuple, Iterable

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from cryoet.data.parsers import (
    get_volume_and_objects,
    TARGET_CLASSES,
    ANGSTROMS_IN_PIXEL,
)


def normalize_volume_to_unit_range(volume):
    volume = volume - volume.min()
    volume = volume / volume.max()
    return volume


class CryoETObjectDetectionDataset(Dataset):
    def __init__(self, root, study, mode, split="train"):
        volume_data, object_centers, object_labels, object_radii = get_volume_and_objects(
            root_dir=root,
            study_name=study,
            mode=mode,
            split=split,
        )

        self.study = study
        self.split = split
        self.mode = mode
        self.volume_data = normalize_volume_to_unit_range(volume_data)
        self.volume_shape = volume_data.shape
        self.object_centers = object_centers
        self.object_labels = object_labels
        self.object_radii = object_radii

        self.object_centers_px = object_centers / ANGSTROMS_IN_PIXEL
        self.object_radii_px = object_radii / ANGSTROMS_IN_PIXEL

        self.num_classes = len(TARGET_CLASSES)


class ObjectDetectionMixin:
    def convert_to_dict(
        self,
        volume,
        centers,
        labels,
        radii,
        study_name,
        volume_shape,
        tile_offsets_zyx,
        mode,
    ):
        volume = torch.from_numpy(volume).unsqueeze(0).float()
        labels = torch.concat(
            [
                torch.from_numpy(centers).float(),
                torch.from_numpy(labels).float(),
                torch.from_numpy(radii).float(),
            ]
        )

        return {
            "volume": volume,
            "labels": labels,
            "tile_offsets_zyx": torch.tensor(tile_offsets_zyx),
            "volume_shape": torch.tensor(volume_shape),
            "study": study_name,
            "mode": mode,
        }

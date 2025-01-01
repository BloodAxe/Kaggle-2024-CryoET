import random

import torch
from sklearn.utils import compute_sample_weight

from cryoet.data.augmentations.functional import (
    rotate_and_scale_volume,
    random_rotate90_volume,
    random_flip_volume,
    get_points_mask_within_cube,
)

from .detection_dataset import ObjectDetectionMixin, CryoETObjectDetectionDataset


class InstanceCropDatasetForPointDetection(CryoETObjectDetectionDataset, ObjectDetectionMixin):
    def __init__(
        self,
        window_size: int,
        root,
        study,
        mode,
        num_crops: int,
        split="train",
        random_rotate: bool = True,
        balance_classes: bool = True,
    ):
        super().__init__(root=root, study=study, mode=mode, split=split)
        self.window_size = window_size
        self.num_crops = num_crops
        self.random_rotate = random_rotate
        self.balance_classes = balance_classes
        if balance_classes:
            self.weights = compute_sample_weight("balanced", self.object_labels)
        else:
            self.weights = None

    def __getitem__(self, idx):
        centers_px = self.object_centers_px  # x y z
        radii_px = self.object_radii_px
        object_labels = self.object_labels

        if self.balance_classes:
            idx = random.choices(range(len(centers_px)), weights=self.weights, k=1)[0]
            center = centers_px[idx]
        else:
            center = random.choice(centers_px)

        scale = 1 + (random.random() - 0.5) / 10.0
        volume, centers_px = rotate_and_scale_volume(
            volume=self.volume_data,
            points=centers_px,
            angles=(
                random.random() * 360,
                0,  # random.random() * 360,
                0,  # random.random() * 360
            ),
            scale=scale,
            center_zyx=(
                center[2] + (random.random() - 0.5) * self.window_size / 5,
                center[1] + (random.random() - 0.5) * self.window_size / 5,
                center[0] + (random.random() - 0.5) * self.window_size / 5,
            ),
            output_shape=(self.window_size, self.window_size, self.window_size),
        )

        radii_px = radii_px * scale

        # Crop the centers to the tile
        keep_mask = get_points_mask_within_cube(centers_px, volume.shape)

        centers_px = centers_px[keep_mask].copy()
        radii_px = radii_px[keep_mask].copy()
        object_labels = object_labels[keep_mask].copy()

        data = self.convert_to_dict(
            volume=volume,
            centers=centers_px,
            labels=object_labels,
            radii=radii_px,
            tile_offsets_zyx=(0, 0, 0),
            study_name=self.study,
            mode=self.mode,
            volume_shape=self.volume_data.shape,
        )

        return data

    def __len__(self):
        return self.num_crops

    def __repr__(self):
        return f"{self.__class__.__name__}(window_size={self.window_size}, study={self.study}, mode={self.mode}, split={self.split}) [{len(self)}]"

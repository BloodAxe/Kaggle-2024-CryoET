import random

import numpy as np

from cryoet.data.augmentations.functional import (
    rotate_and_scale_volume,
    get_points_mask_within_cube,
    random_flip_volume, erase_objects,
)
from .detection_dataset import CryoETObjectDetectionDataset
from .mixin import ObjectDetectionMixin
from ...training.args import DataArguments


class RandomCropForPointDetectionDataset(CryoETObjectDetectionDataset, ObjectDetectionMixin):
    def __init__(
        self,
        window_size: int,
        root,
        study,
        mode,
        num_crops: int,
        data_args: DataArguments,
        split="train",
    ):
        super().__init__(root=root, study=study, mode=mode, split=split)
        self.window_size = window_size
        self.num_crops = num_crops
        self.data_args = data_args

    def __getitem__(self, idx):
        centers_px = self.object_centers_px  # x y z
        radii_px = self.object_radii_px
        object_labels = self.object_labels

        scale = random.uniform(1 - self.data_args.scale_limit, 1 + self.data_args.scale_limit)
        volume, centers_px = rotate_and_scale_volume(
            volume=self.volume_data,
            points=centers_px,
            angles=(
                random.uniform(-self.data_args.z_rotation_limit, self.data_args.z_rotation_limit),
                random.uniform(-self.data_args.y_rotation_limit, self.data_args.y_rotation_limit),
                random.uniform(-self.data_args.x_rotation_limit, self.data_args.x_rotation_limit),
            ),
            scale=scale,
            center_zyx=(
                random.random() * self.volume_shape[0],
                random.random() * self.volume_shape[1],
                random.random() * self.volume_shape[2],
            ),
            output_shape=(self.window_size, self.window_size, self.window_size),
        )

        radii_px = radii_px * scale

        # Crop the centers to the tile
        keep_mask = get_points_mask_within_cube(centers_px, volume.shape)

        centers_px = centers_px[keep_mask].copy()
        radii_px = radii_px[keep_mask].copy()
        object_labels = object_labels[keep_mask].copy()

        if self.data_args.use_random_flips:
            data = random_flip_volume(volume, centers=centers_px)
            volume, centers_px = data["volume"], data["centers"]

        if self.data_args.random_erase_prob > 0:
            mask = np.array([random.random() < self.data_args.random_erase_prob for _ in range(len(centers_px))])
            data = erase_objects(volume, centers_px, radii_px, object_labels, mask)
            volume, centers_px, radii_px, object_labels = data["volume"], data["centers"], data["radii"], data["labels"]

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

import random
from typing import List

from sklearn.utils import compute_sample_weight

from cryoet.data.augmentations.functional import (
    rotate_and_scale_volume,
    get_points_mask_within_cube,
)
from .detection_dataset import CryoETObjectDetectionDataset, apply_augmentations
from .mixin import ObjectDetectionMixin
from ..parsers import AnnotatedVolume
from ...training.args import DataArguments, ModelArguments


class InstanceCropDatasetForPointDetection(CryoETObjectDetectionDataset, ObjectDetectionMixin):
    def __init__(
        self,
        sample: AnnotatedVolume,
        copy_paste_samples: List[AnnotatedVolume],
        num_crops: int,
        model_args: ModelArguments,
        data_args: DataArguments,
        balance_classes: bool = True,
    ):
        super().__init__(sample)
        self.window_size = model_args.depth_window_size, model_args.spatial_window_size, model_args.spatial_window_size
        self.num_crops = num_crops
        self.data_args = data_args
        self.balance_classes = balance_classes
        self.copy_paste_samples = copy_paste_samples

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

        scale = random.uniform(1 - self.data_args.scale_limit, 1 + self.data_args.scale_limit)
        volume, centers_px = rotate_and_scale_volume(
            volume=self.volume_data,
            centers=centers_px,
            angles=(
                random.uniform(-self.data_args.z_rotation_limit, self.data_args.z_rotation_limit),
                random.uniform(-self.data_args.y_rotation_limit, self.data_args.y_rotation_limit),
                random.uniform(-self.data_args.x_rotation_limit, self.data_args.x_rotation_limit),
            ),
            scale=scale,
            center_zyx=(
                center[2] + (random.random() - 0.5) * self.window_size[0] / 5,
                center[1] + (random.random() - 0.5) * self.window_size[1] / 5,
                center[0] + (random.random() - 0.5) * self.window_size[2] / 5,
            ),
            output_shape=self.window_size,
        )

        radii_px = radii_px * scale

        # Crop the centers to the tile
        keep_mask = get_points_mask_within_cube(centers_px, volume.shape)

        centers_px = centers_px[keep_mask].copy()
        radii_px = radii_px[keep_mask].copy()
        object_labels = object_labels[keep_mask].copy()

        data = dict(volume=volume, centers=centers_px, labels=object_labels, radius=radii_px)
        data = apply_augmentations(data, data_args=self.data_args, copy_paste_samples=self.copy_paste_samples, scale=scale)

        data = self.convert_to_dict(
            volume=data["volume"],
            centers=data["centers"],
            labels=data["labels"],
            radii=data["radius"],
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

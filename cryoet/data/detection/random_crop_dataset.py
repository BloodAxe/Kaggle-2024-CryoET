import random
from typing import List

from cryoet.data.augmentations.functional import (
    random_crop_around_point,
)
from .detection_dataset import CryoETObjectDetectionDataset, apply_augmentations, sample_interpolation_mode
from .mixin import ObjectDetectionMixin
from ..parsers import AnnotatedVolume
from ...training.args import DataArguments, ModelArguments


class RandomCropForPointDetectionDataset(CryoETObjectDetectionDataset, ObjectDetectionMixin):
    def __init__(
        self,
        sample: AnnotatedVolume,
        copy_paste_samples: List[AnnotatedVolume],
        num_crops: int,
        model_args: ModelArguments,
        data_args: DataArguments,
    ):
        super().__init__(sample)
        self.window_size = (
            model_args.train_depth_window_size,
            model_args.train_spatial_window_size,
            model_args.train_spatial_window_size,
        )
        self.num_crops = num_crops
        self.model_args = model_args
        self.data_args = data_args
        self.copy_paste_samples = copy_paste_samples

    def __getitem__(self, idx):
        volume_shape = self.sample.volume_shape

        center_xyz = (
            random.random() * volume_shape[2],
            random.random() * volume_shape[0],
            random.random() * volume_shape[1],
        )

        interpolation_mode = sample_interpolation_mode(self.data_args)

        data = random_crop_around_point(
            volume=self.sample.volume,
            centers=self.sample.centers_px,
            labels=self.sample.labels,
            radius=self.sample.radius_px,
            z_rotation_limit=self.data_args.z_rotation_limit,
            y_rotation_limit=self.data_args.y_rotation_limit,
            x_rotation_limit=self.data_args.x_rotation_limit,
            scale_limit=self.data_args.scale_limit,
            anisotropic_scale_limit=self.data_args.anisotropic_scale_limit,
            crop_center_xyz=center_xyz,
            output_shape=self.window_size,
            interpolation_mode=interpolation_mode,
        )

        data = apply_augmentations(
            data, data_args=self.data_args, copy_paste_samples=self.copy_paste_samples, interpolation_mode=interpolation_mode
        )

        data = self.convert_to_dict(
            volume=data["volume"],
            centers=data["centers"],
            labels=data["labels"],
            radii=data["radius"],
            tile_offsets_zyx=(0, 0, 0),
            study_name=self.sample.study,
            mode=self.sample.mode,
            volume_shape=volume_shape,
        )

        return data

    def __len__(self):
        return self.num_crops

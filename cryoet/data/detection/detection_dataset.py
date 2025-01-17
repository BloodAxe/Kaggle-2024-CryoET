import random

from torch.utils.data import Dataset

from cryoet.data.augmentations.functional import (
    random_flip_volume,
    random_erase_objects,
    copy_paste_augmentation,
    gaussian_noise,
    mixup_augmentation,
)
from cryoet.data.parsers import (
    AnnotatedVolume,
)
from cryoet.training.args import DataArguments


class CryoETObjectDetectionDataset(Dataset):
    def __init__(self, sample: AnnotatedVolume):

        self.sample = sample

        self.study = sample.study
        self.split = sample.split
        self.mode = sample.mode
        self.volume_data = sample.volume
        self.volume_shape = sample.volume_shape
        self.object_centers = sample.centers
        self.object_labels = sample.labels
        self.object_radii = sample.radius

        self.object_centers_px = sample.centers_px
        self.object_radii_px = sample.radius_px


def apply_augmentations(data, data_args: DataArguments, copy_paste_samples):
    scale = data["scale"]

    if data_args.use_random_flips:
        data = random_flip_volume(**data)

    if data_args.random_erase_prob > 0:
        data = random_erase_objects(**data, prob=data_args.random_erase_prob)

    if data_args.copy_paste_prob > 0 and random.random() < data_args.copy_paste_prob:
        for _ in range(random.randint(1, data_args.copy_paste_limit)):
            data = copy_paste_augmentation(
                **data,
                samples=copy_paste_samples,
                scale=scale,
                z_rotation_limit=data_args.z_rotation_limit,
                y_rotation_limit=data_args.y_rotation_limit,
                x_rotation_limit=data_args.x_rotation_limit,
                classes_to_paste=[1, 3],  # Only paste beta-galactosidase and thyroglobulin (rare)
            )

    if data_args.mixup_prob > 0 and random.random() < data_args.mixup_prob:
        sample = random.choice(copy_paste_samples)
        data = mixup_augmentation(
            **data,
            sample=sample,
            z_rotation_limit=data_args.z_rotation_limit,
            y_rotation_limit=data_args.y_rotation_limit,
            x_rotation_limit=data_args.x_rotation_limit,
            scale_limit=data_args.scale_limit,
        )

    if data_args.gaussian_noise_sigma > 0:
        data = gaussian_noise(**data, sigma=data_args.gaussian_noise_sigma)

    return data

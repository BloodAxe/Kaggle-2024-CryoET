import random

import torch

from cryoet.data.augmentations.functional import rotate_and_scale_volume, random_rotate90_volume, random_flip_volume
from cryoet.data.point_detection_dataset import CryoETPointDetectionDataset, encode_centers_to_heatmap


class RandomCropCryoETPointDetectionDataset(CryoETPointDetectionDataset):
    def __init__(
            self,
            window_size: int,
            stride: int,
            root,
            study,
            mode,
            num_crops: int,
            split="train",
            random_rotate: bool = False,
    ):
        super().__init__(root, study, mode, split)
        self.window_size = window_size
        self.stride = stride
        self.num_crops = num_crops
        self.random_rotate = random_rotate
    
    def __getitem__(self, idx):
        centers_px = self.object_centers_px  # x y z
        radii_px = self.object_radii_px
        object_labels = self.object_labels
        
        scale = 1 + (random.random() - 0.5) / 10.0
        volume, centers_px = rotate_and_scale_volume(
            volume=self.volume_data,
            points=centers_px,
            angles=(random.random() * 360, random.random() * 360, random.random() * 360),
            scale=scale,
            center=(
                random.random() * self.volume_shape[2],
                random.random() * self.volume_shape[1],
                random.random() * self.volume_shape[0],
            ),
            output_shape=(self.window_size, self.window_size, self.window_size),
        )
        
        radii_px = radii_px * scale
        
        # Crop the centers to the tile
        # fmt: off
        centers_x, centers_y, centers_z = centers_px[:, 0], centers_px[:, 1], centers_px[:, 2]
        keep_mask = (
                (centers_z >= 0) & (centers_z < self.window_size) &
                (centers_y >= 0) & (centers_y < self.window_size) &
                (centers_x >= 0) & (centers_x < self.window_size)
        )
        # fmt: on
        
        centers_px = centers_px[keep_mask].copy()
        radii_px = radii_px[keep_mask].copy()
        object_labels = object_labels[keep_mask].copy()
        
        labels = encode_centers_to_heatmap(
            centers_px,
            object_labels,
            radii_px,
            shape=volume.shape,
            num_classes=self.num_classes,
        )
        
        if self.random_rotate:
            volume, labels = random_rotate90_volume(volume, labels)
            volume, labels = random_flip_volume(volume, labels)
        
        data = {
            "volume": torch.from_numpy(volume).unsqueeze(0),  # C D H W
            "labels": torch.from_numpy(labels),
            "tile_offsets_zyx": torch.tensor([-1, -1, -1]),
            "volume_shape": torch.tensor(self.volume_shape),
            "study": self.study,
            "mode": self.mode,
        }
        return data
    
    def __len__(self):
        return self.num_crops
    
    def __repr__(self):
        return f"{self.__class__.__name__}(window_size={self.window_size}, stride={self.stride}, study={self.study}, mode={self.mode}, split={self.split}) [{len(self)}]"

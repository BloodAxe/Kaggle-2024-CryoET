import numpy as np
import torch

from .augmentations.functional import random_rotate90_volume, random_flip_volume
from .point_detection_dataset import CryoETPointDetectionDataset, compute_tiles, encode_centers_to_heatmap


class SlidingWindowCryoETPointDetectionDataset(CryoETPointDetectionDataset):
    def __repr__(self):
        return f"{self.__class__.__name__}(window_size={self.window_size}, stride={self.stride}, study={self.study}, mode={self.mode}, split={self.split}) [{len(self)}]"
    
    def __init__(
            self,
            window_size: int,
            stride: int,
            root,
            study,
            mode,
            split="train",
            random_rotate: bool = False,
    ):
        super().__init__(root, study, mode, split)
        self.window_size = window_size
        self.stride = stride
        self.tiles = list(compute_tiles(self.volume_data.shape, window_size, stride))
        self.random_rotate = random_rotate
    
    def __getitem__(self, idx):
        tile = self.tiles[idx]  # tiles are z y x order
        centers_px = self.object_centers_px  # x y z
        radii_px = self.object_radii_px
        object_labels = self.object_labels
        
        # Crop the centers to the tile
        # fmt: off
        centers_x, centers_y, centers_z = centers_px[:, 0], centers_px[:, 1], centers_px[:, 2]
        keep_mask = (
                (centers_z >= tile[0].start) & (centers_z < tile[0].stop) &
                (centers_y >= tile[1].start) & (centers_y < tile[1].stop) &
                (centers_x >= tile[2].start) & (centers_x < tile[2].stop)
        )
        # fmt: on
        
        volume = self.volume_data[tile[0], tile[1], tile[2]].copy()
        centers_px = centers_px[keep_mask].copy() - np.array([tile[2].start, tile[1].start, tile[0].start]).reshape(1, 3)
        radii_px = radii_px[keep_mask].copy()
        object_labels = object_labels[keep_mask].copy()
        
        # Pad the volume to the window size
        pad_z = self.window_size - volume.shape[0]
        pad_y = self.window_size - volume.shape[1]
        pad_x = self.window_size - volume.shape[2]
        
        volume = np.pad(
            volume,
            ((0, pad_z), (0, pad_y), (0, pad_x)),
            mode="constant",
            constant_values=0,
        )
        
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
            "tile_offsets_zyx": torch.tensor([tile[0].start, tile[1].start, tile[2].start]),
            "volume_shape": torch.tensor(self.volume_shape),
            "study": self.study,
            "mode": self.mode,
        }
        return data
    
    def __len__(self):
        return len(self.tiles)

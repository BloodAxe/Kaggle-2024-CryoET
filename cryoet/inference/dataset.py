import numpy as np
import torch
from torch.utils.data import Dataset

from cryoet.data.functional import compute_better_tiles


class TileDataset(Dataset):
    def __init__(self, volume, window_size, stride, torch_dtype):
        self.volume = volume
        self.tiles = list(compute_better_tiles(volume.shape, window_size, stride))
        self.window_size = window_size
        self.stride = stride
        self.torch_dtype = torch_dtype

        if isinstance(window_size, int):
            self.window_size_z, self.window_size_y, self.window_size_x = window_size, window_size, window_size
        else:
            self.window_size_z, self.window_size_y, self.window_size_x = window_size

        if isinstance(stride, int):
            self.stide_z, self.stride_y, self.stride_x = stride, stride, stride
        else:
            self.stide_z, self.stride_y, self.stride_x = stride

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, index):
        tile = self.tiles[index]
        tile_volume = self.volume[tile[0], tile[1], tile[2]]

        pad_z = self.window_size_z - tile_volume.shape[0]
        pad_y = self.window_size_y - tile_volume.shape[1]
        pad_x = self.window_size_x - tile_volume.shape[2]

        tile_volume = np.pad(
            tile_volume,
            ((0, pad_z), (0, pad_y), (0, pad_x)),
            mode="constant",
            constant_values=0,
        )

        tile_offsets = (tile[0].start, tile[1].start, tile[2].start)

        return torch.from_numpy(tile_volume).unsqueeze(0).to(self.torch_dtype), torch.tensor(tile_offsets).long()

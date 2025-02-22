from typing import Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from cryoet.data.functional import compute_better_tiles_with_num_tiles, as_tuple_of_3


class TileDataset(Dataset):
    def __init__(self, volume, window_size: Union[int, Tuple[int, int, int]], tiles_per_dim: Tuple[int, int, int], torch_dtype):
        self.volume = volume
        self.tiles = list(compute_better_tiles_with_num_tiles(volume.shape, window_size, tiles_per_dim))
        self.window_size = as_tuple_of_3(window_size)
        self.torch_dtype = torch_dtype

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, index):
        tile = self.tiles[index]
        tile_volume = self.volume[tile[0], tile[1], tile[2]]

        pad_z = self.window_size[0] - tile_volume.shape[0]
        pad_y = self.window_size[1] - tile_volume.shape[1]
        pad_x = self.window_size[2] - tile_volume.shape[2]

        tile_volume = np.pad(
            tile_volume,
            ((0, pad_z), (0, pad_y), (0, pad_x)),
            mode="constant",
            constant_values=0,
        )

        tile_offsets = (tile[0].start, tile[1].start, tile[2].start)

        return torch.from_numpy(tile_volume).unsqueeze(0).to(self.torch_dtype), torch.tensor(tile_offsets).long()

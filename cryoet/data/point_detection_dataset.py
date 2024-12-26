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


class CryoETPointDetectionDataset(Dataset):
    def __init__(self, root, study, mode, split="train"):

        volume_data, object_centers, object_labels, object_radii = (
            get_volume_and_objects(
                root_dir=root,
                study_name=study,
                mode=mode,
                split=split,
            )
        )

        self.study = study
        self.mode = mode
        self.volume_data = volume_data
        self.object_centers = object_centers
        self.object_labels = object_labels
        self.object_radii = object_radii
        self.num_classes = len(TARGET_CLASSES)


def compute_tiles(
    volume_shape: Tuple[int, int, int], window_size: int, stride: int
) -> Iterable[Tuple[slice, slice, slice]]:
    """Compute the slices for a sliding window over a volume.
    A method can output a last slice that is smaller than the window size.
    """
    z, y, x = volume_shape
    for z_start in range(0, z - window_size + 1, stride):
        z_end = z_start + window_size
        for y_start in range(0, y - window_size + 1, stride):
            y_end = y_start + window_size
            for x_start in range(0, x - window_size + 1, stride):
                x_end = x_start + window_size
                yield (
                    slice(z_start, z_end),
                    slice(y_start, y_end),
                    slice(x_start, x_end),
                )


def centernet_gaussian_3d(shape, sigma=1.0):
    d, m, n = [(ss - 1.0) / 2.0 for ss in shape]
    z, y, x = np.ogrid[-d : d + 1, -m : m + 1, -n : n + 1]

    h = np.exp(-(z * z + x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def encode_centers_to_heatmap(centers, labels, radii, shape, num_classes):
    heatmap = np.zeros((num_classes,) + shape, dtype=np.float32)

    depth, height, width = shape
    for center, label, radius in zip(centers, labels, radii):
        z, y, x = center
        z, y, x = int(z + 0.5), int(y + 0.5), int(x + 0.5)

        diameter = 2 * radius
        gaussian = centernet_gaussian_3d(
            (diameter, diameter, diameter), sigma=diameter / 6.0
        )

        front, back = min(z, radius), min(depth - z, radius + 1)
        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_heatmap = heatmap[
            label, z - front : z + back, y - top : y + bottom, x - left : x + right
        ]
        masked_gaussian = gaussian[
            radius - front : radius + back,
            radius - top : radius + bottom,
            radius - left : radius + right,
        ]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            np.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)

    return heatmap


def decoder_centers_from_heatmap(logits: Tensor, kernel=3, top_k=256):
    """

    :param logits: [B,C,D,H,W]
    :param kernel:
    :param top_k: N - number of top candidates per class
    :return:
        Scores [B, N]
        Labels [B, N]
        Coords [B, N, 3] - Coordinates of each peak
    """
    probas = torch.sigmoid(logits)

    # nms
    pad = (kernel - 1) // 2
    maxpool = torch.nn.functional.max_pool3d(probas, kernel_size=kernel, padding=pad)

    #
    mask = probas == maxpool

    peaks = probas * mask

    batch, cat, depth, height, width = peaks.size()

    topk_scores, topk_inds = torch.topk(peaks.view(batch, cat, -1), top_k)

    topk_clses = topk_inds // (depth * height * width)
    topk_inds = topk_inds % (depth * height * width)
    topk_ys = (topk_inds // width) % height
    topk_xs = topk_inds % width
    topk_zs = topk_inds // (width * height)

    topk_scores = topk_scores.view(batch, -1)
    topk_clses = topk_clses.view(batch, -1)
    topk_ys = topk_ys.view(batch, -1)
    topk_xs = topk_xs.view(batch, -1)
    topk_zs = topk_zs.view(batch, -1)

    return topk_scores, topk_clses, torch.stack([topk_zs, topk_ys, topk_xs], dim=-1)


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
    ):

        super().__init__(root, study, mode, split)
        self.window_size = window_size
        self.stride = stride
        self.tiles = list(compute_tiles(self.volume_data.shape, window_size, stride))

    def __getitem__(self, idx):
        tile = self.tiles[idx]
        centers = self.object_centers
        labels = self.object_labels
        radii = self.object_radii

        # Crop the centers to the tile
        # fmt: off
        keep_mask = (
            (centers[:, 0] >= tile[0].start) & (centers[:, 0] < tile[0].stop) &
            (centers[:, 1] >= tile[1].start) & (centers[:, 1] < tile[1].stop) &
            (centers[:, 2] >= tile[2].start) & (centers[:, 2] < tile[2].stop)
        )
        # fmt: on

        volume = self.volume_data[tile[0], tile[1], tile[2]].copy()
        centers = centers[keep_mask].copy() / ANGSTROMS_IN_PIXEL
        radii = radii[keep_mask].copy() / ANGSTROMS_IN_PIXEL
        labels = labels[keep_mask].copy()

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
            centers, labels, radii, shape=volume.shape, num_classes=self.num_classes
        )

        data = {
            "volume": torch.from_numpy(volume).unsqueeze(0),  # C D H W
            "labels": torch.from_numpy(labels),
            "tile_offset": (tile[0].start, tile[1].start, tile[2].start),
            "study": self.study,
            "mode": self.mode,
        }
        return data

    def __len__(self):
        return len(self.tiles)

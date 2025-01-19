import math
from typing import Tuple, Iterable, Union

import numpy as np


def normalize_volume_to_unit_range(volume):
    volume = volume - volume.min()
    volume = volume / volume.max()
    return volume


def normalize_volume_to_percentile_range(volume, low_percentile=1, high_percentile=99):
    low = np.percentile(volume, low_percentile)
    high = np.percentile(volume, high_percentile)
    volume = volume.astype(np.float32, copy=True)
    volume -= low
    volume /= high - low
    return volume


def as_tuple_of_3(value) -> Tuple:
    if isinstance(value, int):
        result = value, value, value
    else:
        a, b, c = value
        result = a, b, c

    return result


def compute_better_tiles_1d(length: int, window_size: int, num_tiles: int):
    """
    Compute the slices for a sliding window over a one dimension.
    Method distribute tiles evenly over the length such that first tile is [0, window_size), and last tile is [length-window_size, length).
    """
    last_tile_start = length - window_size

    starts = np.linspace(0, last_tile_start, num_tiles, dtype=int)
    ends = starts + window_size
    for start, end in zip(starts, ends):
        yield slice(start, end)


def compute_tiles(
    volume_shape: Tuple[int, int, int], window_size: Union[int, Tuple[int, int, int]], stride: Union[int, Tuple[int, int, int]]
) -> Iterable[Tuple[slice, slice, slice]]:
    """
    Compute the slices for a sliding window over a volume.
    A method can output a last slice that is smaller than the window size.
    """
    window_size_z, window_size_y, window_size_x = as_tuple_of_3(window_size)
    stride_z, stride_y, stride_x = as_tuple_of_3(stride)

    z, y, x = volume_shape
    for z_start in range(0, z + stride_z, stride_z):
        if z_start >= z:
            break

        z_end = z_start + window_size_z
        for y_start in range(0, y + stride_y, stride_y):
            if y_start >= y:
                break

            y_end = y_start + window_size_y
            for x_start in range(0, x + stride_x, stride_x):
                if x_start >= x:
                    break

                x_end = x_start + window_size_x
                yield (
                    slice(z_start, z_end),
                    slice(y_start, y_end),
                    slice(x_start, x_end),
                )


def compute_better_tiles(
    volume_shape: Tuple[int, int, int],
    window_size: Union[int, Tuple[int, int, int]],
    window_step: Union[int, Tuple[int, int, int]],
) -> Iterable[Tuple[slice, slice, slice]]:
    """Compute the slices for a sliding window over a volume.
    A method can output a last slice that is smaller than the window size.
    """
    window_size_z, window_size_y, window_size_x = as_tuple_of_3(window_size)
    window_step_z, window_step_y, window_step_x = as_tuple_of_3(window_step)
    z, y, x = volume_shape

    num_z_tiles = math.ceil(z / window_step_z)
    num_y_tiles = math.ceil(y / window_step_y)
    num_x_tiles = math.ceil(x / window_step_x)

    for z_slice in compute_better_tiles_1d(z, window_size_z, num_z_tiles):
        for y_slice in compute_better_tiles_1d(y, window_size_y, num_y_tiles):
            for x_slice in compute_better_tiles_1d(x, window_size_x, num_x_tiles):
                yield (
                    z_slice,
                    y_slice,
                    x_slice,
                )


def compute_better_tiles_with_num_tiles(
    volume_shape: Tuple[int, int, int],
    window_size: Union[int, Tuple[int, int, int]],
    num_tiles: Tuple[int, int, int],
) -> Iterable[Tuple[slice, slice, slice]]:
    """Compute the slices for a sliding window over a volume.
    A method can output a last slice that is smaller than the window size.
    """
    window_size_z, window_size_y, window_size_x = as_tuple_of_3(window_size)
    num_z_tiles, num_y_tiles, num_x_tiles = as_tuple_of_3(num_tiles)
    z, y, x = volume_shape

    for z_slice in compute_better_tiles_1d(z, window_size_z, num_z_tiles):
        for y_slice in compute_better_tiles_1d(y, window_size_y, num_y_tiles):
            for x_slice in compute_better_tiles_1d(x, window_size_x, num_x_tiles):
                yield (
                    z_slice,
                    y_slice,
                    x_slice,
                )

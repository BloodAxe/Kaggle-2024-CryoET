from typing import Tuple, Iterable, Union


def normalize_volume_to_unit_range(volume):
    volume = volume - volume.min()
    volume = volume / volume.max()
    return volume


def as_tuple_of_3(value) -> Tuple:
    if isinstance(value, int):
        result = value, value, value
    else:
        a, b, c = value
        result = a, b, c

    return result


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

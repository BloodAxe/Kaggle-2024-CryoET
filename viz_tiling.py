import numpy as np
from matplotlib import pyplot as plt
from typing import List, Tuple, Union, Any, Iterable, Optional
import torch 

def as_tuple_of_3(value) -> Tuple:
    if isinstance(value, int):
        result = value, value, value
    else:
        a,b,c = value
        result = a,b,c

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

def get_slices_ek(img_shape, roi_size, num_tiles):
    bs , c, *image_size = img_shape
    slices = compute_better_tiles_with_num_tiles(image_size, roi_size, num_tiles)
    slices = [s for s in slices]
    all_slices = [(slice(i,i+1),slice(0,c),) + s for i in range(bs) for s in slices]
    return all_slices

def compute_weight_matrix(shape, sigma=15):
    """
    :param scores_volume: Tensor of shape (C, D, H, W)
    :return: Tensor of shape (D, H, W)
    """
    center = torch.tensor(
        [
            shape[0] / 2,
            shape[1] / 2,
            shape[2] / 2,
        ]
    )

    i = torch.arange(shape[0])
    j = torch.arange(shape[1])
    k = torch.arange(shape[2])

    I, J, K = torch.meshgrid(i, j, k, indexing="ij")
    distances = torch.sqrt((I - center[0]) ** 2 + (J - center[1]) ** 2 + (K - center[2]) ** 2)
    weight = torch.exp(-distances / (sigma**2))

    # I just like the look of heatmap
    return weight**3

def get_counts(slices, original_shape, weight_matrix=True, window_size=(192,128,128)):
    counts = torch.zeros(original_shape, dtype=torch.float, device='cpu')
    if weight_matrix:
        w = compute_weight_matrix(window_size)[None,None]
    else:
        w = 1
    # print(w.shape)
    for i in range(len(slices)):
        counts[slices[i]] += w
    return counts

ORIGINAL_SHAPE = (1,1,192,630,630)
WINDOW_SIZE = (192,128,128)

c = get_counts(get_slices_ek(ORIGINAL_SHAPE, WINDOW_SIZE, (1,9,9)), ORIGINAL_SHAPE, weight_matrix=True , window_size=WINDOW_SIZE)

fix, ax = plt.subplots(1,3,figsize=(30,10))
ax[0].imshow(c[0,0].cpu().numpy().sum(0))
ax[1].imshow(c[0,0].cpu().numpy().sum(1).T)
ax[2].imshow(c[0,0].cpu().numpy().sum(2))
plt.show()
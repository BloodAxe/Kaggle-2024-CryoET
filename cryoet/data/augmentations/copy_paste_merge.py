from scipy import ndimage
import numpy as np


def merge_volume_using_derivatives(volume1, volume2):
    """
    Compute dx, dy, dz derivatives for volume2 and apply them to volume1.
    """
    return volume1 + (volume2 - volume2.mean())


def merge_volume_using_grad_mag(destination, source):
    """
    Compute the gradient magnitude for volume2 in each pixel and use it as weights for blending.
    """
    dx = ndimage.sobel(destination, 0) ** 2  # x derivative
    dy = ndimage.sobel(destination, 1) ** 2  # y derivative
    dz = ndimage.sobel(destination, 2) ** 2  # z derivative
    grad_mag1 = np.sqrt(dx + dy + dz)

    dx = ndimage.sobel(source, 0) ** 2  # x derivative
    dy = ndimage.sobel(source, 1) ** 2  # y derivative
    dz = ndimage.sobel(source, 2) ** 2  # z derivative
    grad_mag2 = np.sqrt(dx + dy + dz)

    return (destination * grad_mag1 + (source - source.mean()) * grad_mag2) / (grad_mag1 + grad_mag2 + 1e-6)


def merge_volume_using_mean(volume1, volume2):
    """
    Compute dx, dy, dz derivatives for volume2 and apply them to volume1.
    """
    return volume1 * 0.5 + volume2 * 0.5


def compute_weighted_matrix(volume, sigma=5.0):
    """
    Compute dx, dy, dz derivatives for volume2 and apply them to volume1.
    """
    # Merge two volumes using weighted sum where weight computed as 3d gaussian with a peak in the center

    # Compute the distance from the center of the volume
    center = np.array(volume.shape) / 2

    i = np.arange(volume.shape[0])
    j = np.arange(volume.shape[1])
    k = np.arange(volume.shape[2])

    I, J, K = np.meshgrid(i, j, k, indexing="ij")
    distances = np.sqrt((I - center[0]) ** 2 + (J - center[1]) ** 2 + (K - center[2]) ** 2)

    # Compute the weight
    weight = np.exp(-distances / (sigma**2))
    mask = distances < sigma * 0.8
    weight[mask] = 1.0
    weight[~mask] -= weight[~mask].min()
    weight[~mask] /= weight[~mask].max()
    return weight


def merge_volume_using_weighted_sum(destination, source):
    """
    Compute dx, dy, dz derivatives for volume2 and apply them to volume1.
    """
    # Merge two volumes using weighted sum where weight computed as 3d gaussian with a peak in the center

    # Compute the weight
    weight = compute_weighted_matrix(source, max(source.shape) * 0.5)

    return destination * (1 - weight) + (source - source.mean() + destination.mean()) * weight


def merge_volume_using_max(volume1, volume2):
    """
    Compute dx, dy, dz derivatives for volume2 and apply them to volume1.
    """
    return np.maximum(volume1, volume2)

from scipy import ndimage
import numpy as np


def merge_volume_using_derivatives(volume1, volume2):
    """
    Compute dx, dy, dz derivatives for volume2 and apply them to volume1.
    """
    return volume1 + (volume2 - volume2.mean())


def merge_volume_using_grad_mag(volume1, volume2):
    """
    Compute the gradient magnitude for volume2 in each pixel and use it as weights for blending.
    """
    dx = ndimage.sobel(volume1, 0) ** 2  # x derivative
    dy = ndimage.sobel(volume1, 1) ** 2  # y derivative
    dz = ndimage.sobel(volume1, 2) ** 2  # z derivative
    grad_mag1 = np.sqrt(dx + dy + dz)

    dx = ndimage.sobel(volume2, 0) ** 2  # x derivative
    dy = ndimage.sobel(volume2, 1) ** 2  # y derivative
    dz = ndimage.sobel(volume2, 2) ** 2  # z derivative
    grad_mag2 = np.sqrt(dx + dy + dz)

    return (volume1 * grad_mag1 + (volume2 - volume2.mean()) * grad_mag2) / (grad_mag1 + grad_mag2 + 1e-6)


def merge_volume_using_mean(volume1, volume2):
    """
    Compute dx, dy, dz derivatives for volume2 and apply them to volume1.
    """
    return volume1 * 0.5 + volume2 * 0.5


def merge_volume_using_max(volume1, volume2):
    """
    Compute dx, dy, dz derivatives for volume2 and apply them to volume1.
    """
    return np.maximum(volume1, volume2)

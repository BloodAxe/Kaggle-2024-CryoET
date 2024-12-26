import random

import numpy as np


def random_rotate90_volume(volume, labels):
    """Randomly rotate the volume and labels by 90 degrees."""
    k1, k2, k3 = random.randint(0, 3), random.randint(0, 3), random.randint(0, 3)

    volume = np.rot90(volume, k=k1, axes=(0, 1))
    labels = np.rot90(labels, k=k1, axes=(0, 1))

    volume = np.rot90(volume, k=k2, axes=(1, 2))
    labels = np.rot90(labels, k=k2, axes=(1, 2))

    volume = np.rot90(volume, k=k3, axes=(0, 2))
    labels = np.rot90(labels, k=k3, axes=(0, 2))

    return volume, labels

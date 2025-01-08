import numpy as np

from cryoet.data.augmentations.functional import erase_objects


def test_random_erase():
    volume = np.ones((32, 32, 32))

    centers = np.array(
        [
            [2, 3, 4],
            [2, 3, 6],
            [20, 20, 20],
            [21, 21, 21],
        ]
    )

    radius = np.array([5, 5, 5, 5])

    labels = np.array([1, 2, 3, 4])

    mask = np.array([1, 0, 1, 0], dtype=bool)

    data = erase_objects(volume, centers, radius, labels, mask)

    volume_new = data["volume"]
    print(data["labels"])
    print(data["radius"])

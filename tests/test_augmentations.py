import numpy as np

from cryoet.data.augmentations.functional import erase_objects


def test_random_erase():
    volume = np.ones((1, 32, 32)) * -1

    centers_xyz = np.array(
        [
            [3, 4, 0],
            [3, 6, 0],
            [20, 20, 0],
            [21, 21, 0],
            [30, 30, 0],
            [0, 0, 0],
            [0, 1, 0],
        ]
    )

    radius = np.array([5, 5, 5, 5, 5, 5, 5])

    labels = np.array([1, 2, 3, 4, 5, 6, 7])

    mask = np.array([1, 0, 1, 0, 1, 1, 1], dtype=bool)

    data = erase_objects(volume, centers_xyz, radius, labels, mask, remove_overlap=True)

    volume_new = data["volume"]

    for (x, y, z), label in zip(data["centers"], data["labels"]):
        volume_new[z, y, x] = label

    print(data["labels"])
    print(data["radius"])
    print(volume_new)


def test_random_erase2():
    volume = np.ones((1, 32, 32)) * -1

    centers_xyz = np.array(
        [
            [3, 4, 0],
            [3, 6, 0],
            [20, 20, 0],
            [21, 21, 0],
            [30, 30, 0],
            [0, 0, 0],
        ]
    )

    radius = np.array([5, 5, 5, 5, 5, 5])

    labels = np.array([1, 2, 3, 4, 5, 6])

    mask = np.array([1, 1, 1, 1, 1, 1], dtype=bool)

    data = erase_objects(volume, centers_xyz, radius, labels, mask, remove_overlap=True)

    volume_new = data["volume"]

    np.testing.assert_array_equal(volume_new, volume)
    assert len(data["centers"]) == 6
    assert len(data["labels"]) == 6
    assert len(data["radius"]) == 6

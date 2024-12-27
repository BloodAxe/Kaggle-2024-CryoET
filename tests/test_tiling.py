from cryoet.data.point_detection_dataset import compute_tiles


def test_tiling():
    volume_shape = (184, 630, 630)

    tiles = compute_tiles(volume_shape, window_size=96, stride=32)
    tiles = list(tiles)

    for tile in tiles:
        print(tile)

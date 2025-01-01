from cryoet.data.heatmap.point_detection_dataset import compute_tiles


def test_tiling():
    volume_shape = (184, 630, 630)

    print()

    tiles = compute_tiles(volume_shape, window_size=(96, 128 + 32, 128 + 32), stride=(92, 128 - 8, 128 - 8))
    tiles = list(tiles)
    print(len(tiles), tiles[-1])

    tiles = compute_tiles(volume_shape, window_size=(96, 128 + 32, 128 + 32), stride=(92, 128 + 16, 128 + 16))
    tiles = list(tiles)
    print(len(tiles), tiles[-1])

    tiles = compute_tiles(volume_shape, window_size=(96, 128 + 32, 128 + 32), stride=(92, 128 + 8, 128 + 8))
    tiles = list(tiles)
    print(len(tiles), tiles[-1])

    tiles = compute_tiles(volume_shape, window_size=(96, 96, 96), stride=(92, 92, 92))
    tiles = list(tiles)
    print(len(tiles), tiles[-1])

    tiles = compute_tiles(volume_shape, window_size=(96, 96, 96), stride=(92, 80, 80))
    tiles = list(tiles)
    print(len(tiles), tiles[-1])

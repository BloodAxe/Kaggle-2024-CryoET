from cryoet.data.heatmap.point_detection_dataset import compute_tiles


def test_tiling():
    volume_shape = (184, 630, 630)

    print()

    tiles = compute_tiles(volume_shape, window_size=(96, 96, 96), stride=(92, 92, 92))
    tiles = list(tiles)
    print(len(tiles), tiles[-1])

    tiles = compute_tiles(volume_shape, window_size=(96, 96, 96), stride=(64, 72, 72))
    tiles = list(tiles)
    print(len(tiles), tiles[-1])

    tiles = compute_tiles(volume_shape, window_size=(96, 96, 96), stride=(64, 71, 71))
    tiles = list(tiles)
    print(len(tiles), tiles[-1])

    tiles = compute_tiles(volume_shape, window_size=(96, 96, 96), stride=(64, 70, 70))
    tiles = list(tiles)
    print(len(tiles), tiles[-1])

    tiles = compute_tiles(volume_shape, window_size=(96, 96, 96), stride=(62, 60, 60))
    tiles = list(tiles)
    print(len(tiles), tiles[-1])

    # tiles = compute_tiles(volume_shape, window_size=(96, 96, 96), stride=(64, 67, 67))
    # tiles = list(tiles)
    # print(len(tiles), tiles[-1])
    #
    # tiles = compute_tiles(volume_shape, window_size=(96, 96, 96), stride=(64, 66, 66))
    # tiles = list(tiles)
    # print(len(tiles), tiles[-1])
    #
    # tiles = compute_tiles(volume_shape, window_size=(96, 96, 96), stride=(64, 65, 65))
    # tiles = list(tiles)
    # print(len(tiles), tiles[-1])
    #
    # tiles = compute_tiles(volume_shape, window_size=(96, 96, 96), stride=(64, 64, 64))
    # tiles = list(tiles)
    # print(len(tiles), tiles[-1])

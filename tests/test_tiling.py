from cryoet.data.functional import compute_better_tiles_1d, compute_tiles, compute_better_tiles


def test_better_tiles():
    tiles = list(compute_better_tiles_1d(184, 96, 4))
    print(len(tiles), tiles)


def test_tiling():
    volume_shape = (184, 630, 630)

    print()

    tiles = compute_tiles(volume_shape, window_size=(96, 128, 128), stride=(62, 80, 80))
    tiles = list(tiles)
    print(len(tiles), tiles[-1])

    tiles = compute_better_tiles(volume_shape, window_size=(192, 128, 128), window_step=(192, 80, 80))
    tiles = list(tiles)
    print(len(tiles), tiles[-1])

    # print("Step Z")
    #
    # for j in range(50, 95):
    #     tiles = compute_tiles(volume_shape, window_size=(96, 96, 96), stride=(j, 70, 70))
    #     tiles = list(tiles)
    #     print(j, len(tiles), tiles[-1])
    #
    # print()
    #
    # tiles = compute_tiles(volume_shape, window_size=(128, 128, 128), stride=(84, 90, 90))
    # tiles = list(tiles)
    # print(len(tiles), tiles[-1])
    #
    # for i in range(70, 120):
    #     tiles = compute_tiles(volume_shape, window_size=(96, 128, 128), stride=(62, i, i))
    #     tiles = list(tiles)
    #     print(i, len(tiles), tiles[-1])

from cryoet.data.functional import (
    compute_better_tiles_1d,
    compute_tiles,
    compute_better_tiles,
    compute_better_tiles_with_num_tiles,
)


def test_better_tiles():
    tiles = list(compute_better_tiles_1d(184, 96, 4))
    print(len(tiles), tiles)


def test_better_tiles_one_tile():
    tiles = list(compute_better_tiles_1d(630, 128, 4))
    print(len(tiles), tiles, "Overlap", (tiles[0].stop - tiles[0].start) - (tiles[1].start - tiles[0].start))

    tiles = list(compute_better_tiles_1d(630, 128, 5))
    print(len(tiles), tiles, "Overlap", (tiles[0].stop - tiles[0].start) - (tiles[1].start - tiles[0].start))

    tiles = list(compute_better_tiles_1d(630, 128, 6))
    print(len(tiles), tiles, "Overlap", (tiles[0].stop - tiles[0].start) - (tiles[1].start - tiles[0].start))

    tiles = list(compute_better_tiles_1d(630, 128, 7))
    print(len(tiles), tiles, "Overlap", (tiles[0].stop - tiles[0].start) - (tiles[1].start - tiles[0].start))

    tiles = list(compute_better_tiles_1d(630, 128, 8))
    print(len(tiles), tiles, "Overlap", (tiles[0].stop - tiles[0].start) - (tiles[1].start - tiles[0].start))

    tiles = list(compute_better_tiles_1d(630, 128, 9))
    print(len(tiles), tiles, "Overlap", (tiles[0].stop - tiles[0].start) - (tiles[1].start - tiles[0].start))


def test_tiling():
    volume_shape = (184, 630, 630)

    print()

    tiles = compute_better_tiles_with_num_tiles(volume_shape, window_size=(192, 128, 128), num_tiles=(1, 7, 7))
    tiles = list(tiles)
    print(len(tiles), tiles[-1])

    tiles = compute_better_tiles_with_num_tiles(volume_shape, window_size=(192, 128, 128), num_tiles=(1, 8, 8))
    tiles = list(tiles)
    print(len(tiles), tiles[-1])

    tiles = compute_better_tiles_with_num_tiles(volume_shape, window_size=(192, 128, 128), num_tiles=(1, 9, 9))
    tiles = list(tiles)
    print(len(tiles), tiles[-1])

from pathlib import Path
from pprint import pprint

from cryoet.data.cross_validation import split_data_into_folds
from cryoet.data.point_detection_dataset import SlidingWindowCryoETPointDetectionDataset

DATA_ROOT = Path(__file__).parent.parent / "data" / "czii-cryo-et-object-identification"
TRAIN_DATA_DIR = DATA_ROOT / "train" / "static" / "ExperimentRuns"


def test_split_data_into_folds():
    folds = split_data_into_folds(TRAIN_DATA_DIR)
    pprint(folds)

    assert len(folds) == 5
    for train_studies, val_studies in folds:
        assert len(train_studies) == 5
        assert len(val_studies) == 2
        assert len(set(train_studies) & set(val_studies)) == 0


def test_dataset():
    folds = split_data_into_folds(TRAIN_DATA_DIR)

    train_studies = folds[0][0]

    dataset = SlidingWindowCryoETPointDetectionDataset(
        window_size=96,
        stride=64,
        root=DATA_ROOT,
        study=train_studies[0],
        mode="denoised",
    )
    assert len(dataset) > 0
    sample = dataset[0]

    assert "volume" in sample
    assert "labels" in sample
    assert sample["volume"].shape == (96, 96, 96)
    assert sample["labels"].shape == (dataset.num_classes, 96, 96, 96)

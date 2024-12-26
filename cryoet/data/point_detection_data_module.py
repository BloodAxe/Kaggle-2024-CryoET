from collections import defaultdict
from pathlib import Path
from typing import List

import lightning as L
import pandas as pd
from torch.utils.data import ConcatDataset, DataLoader

from cryoet.data.cross_validation import split_data_into_folds
from cryoet.data.parsers import CLASS_LABEL_TO_CLASS_NAME
from cryoet.data.point_detection_dataset import SlidingWindowCryoETPointDetectionDataset


class PointDetectionDataModule(L.LightningDataModule):
    def __init__(
        self,
        root: str | Path,
        train_modes: str | List,
        valid_modes: str | List,
        window_size: int,
        stride: int,
        fold: int,
        train_batch_size: int,
        valid_batch_size: int,
        dataloader_num_workers: int = 0,
        dataloader_pin_memory: bool = False,
        dataloader_persistent_workers: bool = False,
    ):
        super().__init__()
        self.root = Path(root)
        self.runs_dir = self.root / "train" / "static" / "ExperimentRuns"
        self.train_modes = [train_modes] if isinstance(train_modes, str) else list(train_modes)
        self.valid_modes = [valid_modes] if isinstance(valid_modes, str) else list(valid_modes)
        self.window_size = window_size
        self.stride = stride
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.dataloader_num_workers = dataloader_num_workers
        self.dataloader_pin_memory = dataloader_pin_memory
        self.dataloader_persistent_workers = dataloader_persistent_workers

        self.train_studies, self.valid_studies = split_data_into_folds(self.runs_dir)[fold]

        self.train = None
        self.val = None
        self.solution = None

    def setup(self, stage):
        train_datasets = []
        for train_study in self.train_studies:
            for mode in self.train_modes:
                dataset = SlidingWindowCryoETPointDetectionDataset(
                    window_size=self.window_size,
                    stride=self.stride,
                    root=self.root,
                    study=train_study,
                    mode=mode,
                    split="train",
                    random_rotate=True,
                )
                train_datasets.append(dataset)

        solution = defaultdict(list)

        valid_datasets = []
        for study_name in self.valid_studies:
            for mode in self.valid_modes:
                dataset = SlidingWindowCryoETPointDetectionDataset(
                    window_size=self.window_size,
                    stride=self.stride,
                    root=self.root,
                    study=study_name,
                    mode=mode,
                    split="train",
                    random_rotate=True,
                )
                valid_datasets.append(dataset)

                for i, (center, label, radius) in enumerate(
                    zip(dataset.object_centers, dataset.object_labels, dataset.object_radii)
                ):
                    solution["experiment"].append(study_name)
                    solution["particle_type"].append(CLASS_LABEL_TO_CLASS_NAME[label])
                    solution["x"].append(float(center[0]))
                    solution["y"].append(float(center[1]))
                    solution["z"].append(float(center[2]))

        self.train = ConcatDataset(train_datasets)
        self.val = ConcatDataset(valid_datasets)
        self.solution = pd.DataFrame.from_dict(solution)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train,
            batch_size=self.train_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.dataloader_num_workers,
            pin_memory=self.dataloader_pin_memory,
            persistent_workers=self.dataloader_persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val,
            batch_size=self.train_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.dataloader_num_workers,
            pin_memory=self.dataloader_pin_memory,
            persistent_workers=self.dataloader_persistent_workers,
        )

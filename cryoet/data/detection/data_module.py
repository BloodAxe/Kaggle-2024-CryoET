from collections import defaultdict
from pathlib import Path
from typing import List, Dict

import lightning as L
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import ConcatDataset, DataLoader, default_collate

from cryoet.data.cross_validation import split_data_into_folds
from cryoet.data.parsers import CLASS_LABEL_TO_CLASS_NAME
from cryoet.training.args import DataArguments, MyTrainingArguments
from .instance_crop_dataset import InstanceCropDatasetForPointDetection
from .random_crop_dataset import RandomCropForPointDetectionDataset

from .sliding_window_dataset import SlidingWindowCryoETObjectDetectionDataset


class ObjectDetectionDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_args: MyTrainingArguments,
        data_args: DataArguments,
        window_size: int,
        stride: int,
    ):
        super().__init__()

        train_modes = data_args.train_modes
        valid_modes = data_args.valid_modes

        self.root = Path(data_args.data_root)
        self.runs_dir = self.root / "train" / "static" / "ExperimentRuns"
        self.train_modes = train_modes.split(",") if isinstance(train_modes, str) else list(train_modes)
        self.valid_modes = valid_modes.split(",") if isinstance(valid_modes, str) else list(valid_modes)
        self.window_size = window_size
        self.stride = stride
        self.train_batch_size = train_args.per_device_train_batch_size
        self.valid_batch_size = train_args.per_device_eval_batch_size
        self.dataloader_num_workers = train_args.dataloader_num_workers
        self.dataloader_pin_memory = train_args.dataloader_pin_memory
        self.dataloader_persistent_workers = train_args.dataloader_persistent_workers
        self.fold = data_args.fold

        self.train_studies, self.valid_studies = split_data_into_folds(self.runs_dir)[self.fold]
        self.data_args = data_args
        self.train_args = train_args
        self.train = None
        self.val = None
        self.solution = None

    def setup(self, stage):
        train_datasets = []
        for train_study in self.train_studies:
            for mode in self.train_modes:
                if self.data_args.use_sliding_crops:
                    sliding_dataset = SlidingWindowCryoETObjectDetectionDataset(
                        window_size=self.window_size,
                        stride=self.stride,
                        root=self.root,
                        study=train_study,
                        mode=mode,
                        split="train",
                        random_rotate=True,
                    )
                    train_datasets.append(sliding_dataset)

                if self.data_args.use_random_crops:
                    random_crop_dataset = RandomCropForPointDetectionDataset(
                        num_crops=self.data_args.num_crops_per_study,
                        window_size=self.window_size,
                        root=self.root,
                        study=train_study,
                        mode=mode,
                        split="train",
                        random_rotate=True,
                    )
                    train_datasets.append(random_crop_dataset)

                if self.data_args.use_instance_crops:
                    crop_around_dataset = InstanceCropDatasetForPointDetection(
                        num_crops=self.data_args.num_crops_per_study,
                        window_size=self.window_size,
                        root=self.root,
                        study=train_study,
                        mode=mode,
                        split="train",
                        random_rotate=True,
                    )
                    train_datasets.append(crop_around_dataset)

        solution = defaultdict(list)

        valid_datasets = []
        for study_name in self.valid_studies:
            for mode in self.valid_modes:
                dataset = SlidingWindowCryoETObjectDetectionDataset(
                    window_size=self.window_size,
                    stride=self.stride,
                    root=self.root,
                    study=study_name,
                    mode=mode,
                    split="train",
                    random_rotate=False,
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
            collate_fn=ObjectDetectionCollate(),
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
            collate_fn=ObjectDetectionCollate(),
        )


class ObjectDetectionCollate:
    def __call__(self, samples: List[Dict]):
        labels = [sample["labels"] for sample in samples]
        num_items_in_batch = sum(len(l) for l in labels)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        all_but_labels: List[Dict] = [{k: v for k, v in sample.items() if k != "labels"} for sample in samples]
        batch = default_collate(all_but_labels)

        return {**batch, "labels": labels, "num_items_in_batch": torch.tensor(num_items_in_batch)}

from collections import defaultdict
from pathlib import Path
from typing import List, Dict

import lightning as L
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import ConcatDataset, DataLoader, default_collate

from cryoet.data.cross_validation import split_data_into_folds, get_ninja_split
from cryoet.data.parsers import read_annotated_volume, AnnotatedVolume, CLASS_LABEL_TO_CLASS_NAME
from cryoet.training.args import DataArguments, MyTrainingArguments, ModelArguments
from .instance_crop_dataset import InstanceCropDatasetForPointDetection
from .random_crop_dataset import RandomCropForPointDetectionDataset

from .sliding_window_dataset import SlidingWindowCryoETObjectDetectionDataset


class ObjectDetectionDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_args: MyTrainingArguments,
        data_args: DataArguments,
        model_args: ModelArguments,
    ):
        super().__init__()

        train_modes = data_args.train_modes
        valid_modes = data_args.valid_modes

        self.root = Path(data_args.data_root)
        self.runs_dir = self.root / "train" / "static" / "ExperimentRuns"
        self.train_modes = train_modes.split(",") if isinstance(train_modes, str) else list(train_modes)
        self.valid_modes = valid_modes.split(",") if isinstance(valid_modes, str) else list(valid_modes)
        self.dataloader_num_workers = train_args.dataloader_num_workers
        self.dataloader_pin_memory = train_args.dataloader_pin_memory
        self.dataloader_persistent_workers = train_args.dataloader_persistent_workers
        self.fold = data_args.fold

        if self.fold == -1:
            self.train_studies, self.valid_studies_original = get_ninja_split(self.runs_dir)
        else:
            self.train_studies, self.valid_studies_original = split_data_into_folds(self.runs_dir)[self.fold]

        self.data_args = data_args
        self.model_args = model_args
        self.train_args = train_args
        self.train = None
        self.val = None
        self.solution = None
        self.train_solution = None
        self.valid_studies = None

    @classmethod
    def build_dataset_from_samples(
        cls,
        samples: List[AnnotatedVolume],
        use_sliding_crops,
        use_random_crops,
        use_instance_crops,
        model_args,
        data_args,
    ):
        datasets = []
        solution = {
            "experiment": [],
            "particle_type": [],
            "x": [],
            "y": [],
            "z": [],
        }

        for sample in samples:

            if use_sliding_crops:
                x_options = [False, True] if data_args.validate_on_x_flips else [False]
                y_options = [False, True] if data_args.validate_on_y_flips else [False]
                z_options = [False, True] if data_args.validate_on_z_flips else [False]
                rot_options = [0, 1, 2, 3] if data_args.validate_on_rot90 else [0]

                for rot in rot_options:
                    for x_flip in x_options:
                        for y_flip in y_options:
                            for z_flip in z_options:
                                print("Flipping", x_flip, y_flip, z_flip)
                                maybe_flipped_sample = sample.rot90(rot).flip(x_flip, y_flip, z_flip)

                                sliding_dataset = SlidingWindowCryoETObjectDetectionDataset(
                                    sample=maybe_flipped_sample,
                                    data_args=data_args,
                                    model_args=model_args,
                                )

                                for i, (center, label, radius) in enumerate(
                                    zip(maybe_flipped_sample.centers, maybe_flipped_sample.labels, maybe_flipped_sample.radius)
                                ):
                                    solution["experiment"].append(maybe_flipped_sample.study)
                                    solution["particle_type"].append(CLASS_LABEL_TO_CLASS_NAME[label])
                                    solution["x"].append(float(center[0]))
                                    solution["y"].append(float(center[1]))
                                    solution["z"].append(float(center[2]))

                                print(f"Added {maybe_flipped_sample.study} objects to the dataset")
                                datasets.append(sliding_dataset)

            if use_random_crops:
                random_crop_dataset = RandomCropForPointDetectionDataset(
                    sample=sample,
                    num_crops=data_args.num_crops_per_study,
                    model_args=model_args,
                    data_args=data_args,
                    copy_paste_samples=samples,
                )
                datasets.append(random_crop_dataset)

            if use_instance_crops:
                crop_around_dataset = InstanceCropDatasetForPointDetection(
                    sample=sample,
                    num_crops=data_args.num_crops_per_study,
                    model_args=model_args,
                    data_args=data_args,
                    copy_paste_samples=samples,
                )
                datasets.append(crop_around_dataset)

        dataset = ConcatDataset(datasets)
        solution = pd.DataFrame.from_dict(solution)
        return dataset, solution

    def setup(self, stage):
        train_samples = []
        for train_study in self.train_studies:
            for mode in self.train_modes:
                sample = read_annotated_volume(
                    root=self.root, study=train_study, mode=mode, split="train", use_6_classes=self.model_args.use_6_classes
                )
                train_samples.append(sample)

        valid_samples = []
        for study_name in self.valid_studies_original:
            for mode in self.valid_modes:
                sample = read_annotated_volume(
                    root=self.root, study=study_name, mode=mode, split="train", use_6_classes=self.model_args.use_6_classes
                )
                valid_samples.append(sample)

        self.train, self.train_solution = self.build_dataset_from_samples(
            train_samples,
            use_sliding_crops=self.data_args.use_sliding_crops,
            use_instance_crops=self.data_args.use_instance_crops,
            use_random_crops=self.data_args.use_random_crops,
            model_args=self.model_args,
            data_args=self.data_args,
        )
        self.val, self.solution = self.build_dataset_from_samples(
            valid_samples,
            use_sliding_crops=True,
            use_instance_crops=False,
            use_random_crops=False,
            model_args=self.model_args,
            data_args=self.data_args,
        )

        # Update the list of validation studies
        self.valid_studies = [dataset.sample.study for dataset in self.val.datasets]
        print("Validation studies:", self.valid_studies)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train,
            batch_size=self.train_args.per_device_train_batch_size,
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
            batch_size=self.train_args.per_device_eval_batch_size,
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

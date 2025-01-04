import dataclasses
from typing import Tuple, List

import torch
from torch import Tensor


@dataclasses.dataclass
class AccumulatedObjectDetectionPredictionContainer:
    scores: List[Tensor]
    offsets: List[Tensor]
    counter: List[Tensor]
    strides: List[int]

    @classmethod
    def from_shape(cls, shape: Tuple[int, int, int], num_classes: int, strides: List[int], device="cpu", dtype=torch.float32):
        d, h, w = shape

        # fmt: off
        return cls(
            scores=[torch.zeros((num_classes, d // stride, h // stride, w // stride), device=device, dtype=dtype) for stride in strides],
            offsets=[torch.zeros((3, d // stride, h // stride, w // stride), device=device, dtype=dtype) for stride in strides],
            counter=[torch.zeros(d // stride, h // stride, w // stride, device=device, dtype=dtype) for stride in strides],
            strides=list(strides),
        )
        # fmt: on

    def __iadd__(self, other):
        if self.strides != other.strides:
            raise ValueError("Strides mismatch")

        for i in range(len(self.scores)):
            self.scores[i] += other.scores[i].to(self.scores[i].device)
            self.offsets[i] += other.offsets[i].to(self.offsets[i].device)
            self.counter[i] += other.counter[i].to(self.counter[i].device)

        return self

    def accumulate_batch(self, batch_scores, batch_offsets, batch_tile_coords):
        for scores_list, offsets_list, tile_coords_zyx in zip(batch_scores, batch_offsets, batch_tile_coords):
            self.accumulate(scores_list, offsets_list, tile_coords_zyx)

    def accumulate(self, scores_list: List[Tensor], offsets_list: List[Tensor], tile_coords_zyx):
        if len(scores_list) != len(self.scores):
            raise ValueError("Number of feature maps mismatch")
        if not isinstance(scores_list, list):
            raise ValueError("Scores should be a list of tensors")
        if not isinstance(offsets_list, list):
            raise ValueError("Offsets should be a list of tensors")

        num_feature_maps = len(self.scores)

        for i in range(num_feature_maps):
            stride = self.strides[i]
            scores = scores_list[i]
            offsets = offsets_list[i]

            if scores.ndim != 4 or offsets.ndim != 4:
                raise ValueError("Scores and offsets should have shape (C, D, H, W)")

            strided_offsets_zyx = tile_coords_zyx // stride
            roi = (
                slice(strided_offsets_zyx[0], strided_offsets_zyx[0] + scores.shape[1]),
                slice(strided_offsets_zyx[1], strided_offsets_zyx[1] + scores.shape[2]),
                slice(strided_offsets_zyx[2], strided_offsets_zyx[2] + scores.shape[3]),
            )

            scores_view = self.scores[i][:, *roi]
            offsets_view = self.offsets[i][:, *roi]
            counter_view = self.counter[i][*roi]

            # Crop tile_scores to the view shape
            scores = scores[:, : scores_view.shape[1], : scores_view.shape[2], : scores_view.shape[3]]
            offsets = offsets[:, : offsets_view.shape[1], : offsets_view.shape[2], : offsets_view.shape[3]]

            scores_view += scores.to(scores_view.device)
            offsets_view += offsets.to(offsets_view.device)
            counter_view += 1

    def merge_(self):
        num_feature_maps = len(self.scores)
        for i in range(num_feature_maps):
            c = self.counter[i].unsqueeze(0)
            zero_mask = c.eq(0)

            self.scores[i] /= c
            self.scores[i].masked_fill_(zero_mask, 0.0)

            self.offsets[i] /= c
            self.offsets[i].masked_fill_(zero_mask, 0.0)

        return self.scores, self.offsets

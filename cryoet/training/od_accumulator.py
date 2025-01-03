import dataclasses
from typing import Tuple, List

import torch
from torch import Tensor


@dataclasses.dataclass
class AccumulatedObjectDetectionPredictionContainer:
    scores: List[Tensor]
    centers: List[Tensor]
    counter: List[Tensor]

    @classmethod
    def from_shape(cls, shape: Tuple[int, int, int], num_classes: int, strides: List[int], device="cpu", dtype=torch.float32):
        d, h, w = shape

        # fmt: off
        return cls(
            scores=[torch.zeros((num_classes, d // stride, h // stride, w // stride), device=device, dtype=dtype) for stride in strides],
            centers=[torch.zeros((3, d // stride, h // stride, w // stride), device=device, dtype=dtype) for stride in strides],
            counter=[torch.zeros(d // stride, h // stride, w // stride, device=device, dtype=dtype) for stride in strides],
        )
        # fmt: on

    def accumulate_batch(self, batch_probas, batch_centers, batch_tile_offsets):
        for tile_scores, tile_centers, tile_offsets in zip(batch_probas, batch_centers, batch_tile_offsets):
            self.accumulate(tile_scores, tile_centers, tile_offsets)

    def accumulate(self, tile_scores: List[Tensor], pred_centers: List[Tensor], tile_offsets_zyx):
        num_feature_maps = len(self.scores)

        for i in range(num_feature_maps):
            stride = self.strides[i]
            strided_offsets_zyx = tile_offsets_zyx // stride
            roi = (
                slice(strided_offsets_zyx[0], strided_offsets_zyx[0] + tile_scores.shape[1]),
                slice(strided_offsets_zyx[1], strided_offsets_zyx[1] + tile_scores.shape[2]),
                slice(strided_offsets_zyx[2], strided_offsets_zyx[2] + tile_scores.shape[3]),
            )

            probas_view = self.scores[i][:, roi]
            centers_view = self.centers[i][:, roi]
            counter_view = self.counter[i][roi]

            tile_offsets_xyz = torch.tensor(
                [
                    tile_offsets_zyx[2],
                    tile_offsets_zyx[1],
                    tile_offsets_zyx[0],
                ],
                device=probas_view.device,
            ).view(3, 1, 1, 1)

            # Crop tile_scores to the view shape
            tile_scores = tile_scores[:, : probas_view.shape[1], : probas_view.shape[2], : probas_view.shape[3]]
            pred_centers = pred_centers[:, : centers_view.shape[1], : centers_view.shape[2], : centers_view.shape[3]]

            probas_view += tile_scores.to(probas_view.device)
            centers_view += (pred_centers + tile_offsets_xyz).to(centers_view.device)
            counter_view += 1

    def merge_(self):
        num_feature_maps = len(self.scores)
        for i in range(num_feature_maps):
            self.scores[i] /= self.counter[i].unsqueeze(0)
            self.scores[i].masked_fill_(self.counter[i] == 0, 0.0)

            self.centers[i] /= self.counter[i].unsqueeze(0)
            self.centers[i].masked_fill_(self.counter[i] == 0, 0.0)

        return self.scores, self.centers

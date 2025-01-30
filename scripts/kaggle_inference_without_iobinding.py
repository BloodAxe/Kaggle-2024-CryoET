%%writefile inference_script.py

from ctypes import *

import os
import torch
import dataclasses
import zarr
import math

from collections import defaultdict
from pathlib import Path
from fire import Fire

import numpy as np
import pandas as pd
import torch.jit
from tqdm import tqdm

import typing
import einops

from torch import Tensor, nn
from typing import List, Tuple, Union, Any, Iterable, Optional
from torch.utils.data import Dataset, DataLoader
import onnxruntime as ort

ANGSTROMS_IN_PIXEL = 10.012

TARGET_CLASSES = (
    {
        "name": "apo-ferritin",
        "label": 0,
        "color": [0, 117, 255],
        "radius": 60,
        "map_threshold": 0.0418,
    },
    {
        "name": "beta-galactosidase",
        "label": 1,
        "color": [176, 0, 192],
        "radius": 90,
        "map_threshold": 0.0578,
    },
    {
        "name": "ribosome",
        "label": 2,
        "color": [0, 92, 49],
        "radius": 150,
        "map_threshold": 0.0374,
    },
    {
        "name": "thyroglobulin",
        "label": 3,
        "color": [43, 255, 72],
        "radius": 130,
        "map_threshold": 0.0278,
    },
    {
        "name": "virus-like-particle",
        "label": 4,
        "color": [255, 30, 53],
        "radius": 135,
        "map_threshold": 0.201,
    },
    {"name": "beta-amylase", "label": 5, "color": [153, 63, 0, 128], "radius": 65, "map_threshold": 0.035},
)

CLASS_LABEL_TO_CLASS_NAME = {c["label"]: c["name"] for c in TARGET_CLASSES}
TARGET_SIGMAS = [c["radius"] / ANGSTROMS_IN_PIXEL for c in TARGET_CLASSES]

def normalize_volume_to_unit_range(volume):
    volume = volume - volume.min()
    volume = volume / volume.max()
    return volume.astype(np.float32)


def as_tuple_of_3(value) -> Tuple:
    if isinstance(value, int):
        result = value, value, value
    else:
        a,b,c = value
        result = a,b,c

    return result


def compute_better_tiles_1d(length: int, window_size: int, num_tiles: int):
    """
    Compute the slices for a sliding window over a one dimension.
    Method distribute tiles evenly over the length such that first tile is [0, window_size), and last tile is [length-window_size, length).
    """
    last_tile_start = length - window_size

    starts = np.linspace(0, last_tile_start, num_tiles, dtype=int)
    ends = starts + window_size
    for start, end in zip(starts, ends):
        yield slice(start, end)


def compute_better_tiles_with_num_tiles(
        volume_shape: Tuple[int, int, int],
        window_size: Union[int, Tuple[int, int, int]],
        num_tiles: Tuple[int, int, int],
) -> Iterable[Tuple[slice, slice, slice]]:
    """Compute the slices for a sliding window over a volume.
    A method can output a last slice that is smaller than the window size.
    """
    window_size_z, window_size_y, window_size_x = as_tuple_of_3(window_size)
    num_z_tiles, num_y_tiles, num_x_tiles = as_tuple_of_3(num_tiles)
    z, y, x = volume_shape

    for z_slice in compute_better_tiles_1d(z, window_size_z, num_z_tiles):
        for y_slice in compute_better_tiles_1d(y, window_size_y, num_y_tiles):
            for x_slice in compute_better_tiles_1d(x, window_size_x, num_x_tiles):
                yield (
                    z_slice,
                    y_slice,
                    x_slice,
                )

class TileDataset(Dataset):
    def __init__(self, volume, window_size: Union[int, Tuple[int, int, int]], tiles_per_dim: Tuple[int, int, int], dtype, return_tensors):
        self.volume = volume.astype(dtype)
        self.tiles = list(compute_better_tiles_with_num_tiles(volume.shape, window_size, tiles_per_dim))
        self.window_size = as_tuple_of_3(window_size)
        self.dtype = dtype
        self.return_tensors = return_tensors

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, index):
        tile = self.tiles[index]
        tile_volume = self.volume[tile[0], tile[1], tile[2]]

        pad_z = self.window_size[0] - tile_volume.shape[0]
        pad_y = self.window_size[1] - tile_volume.shape[1]
        pad_x = self.window_size[2] - tile_volume.shape[2]

        tile_volume = np.pad(
            tile_volume,
            ((0, pad_z), (0, pad_y), (0, pad_x)),
            mode="constant",
            constant_values=0,
        )

        tile_volume = tile_volume[None, :, :, :] # Add channel dim
        tile_offsets = np.array([tile[0].start, tile[1].start, tile[2].start], dtype=int)

        if self.return_tensors == "pt":
            tile_volume = torch.from_numpy(tile_volume)
            tile_offsets = torch.from_numpy(tile_offsets).long()

        return tile_volume, tile_offsets

def get_volume(
        root_dir: str | Path,
        study_name: str,
        mode: str = "denoised",
        split: str = "train",
        voxel_spacing_str: str = "VoxelSpacing10.000",
):
    """
    Opens a Zarr store for the specified study and mode (e.g. denoised, isonetcorrected),
    returns it as a NumPy array (fully loaded).

    :param root_dir: Base directory (e.g., /path/to/czii-cryo-et-object-identification).
    :param study_name: For example, "TS_5_4".
    :param mode: Which volume mode to load, e.g. "denoised", "isonetcorrected", "wbp", etc.
    :param split: "train" or "test".
    :param voxel_spacing_str: Typically "VoxelSpacing10.000" from your structure.
    :return: A 3D NumPy array of the volume data.
    """
    # Example path:
    #   /.../train/static/ExperimentRuns/TS_5_4/VoxelSpacing10.000/denoised.zarr
    zarr_path = os.path.join(
        str(root_dir),
        split,
        "static",
        "ExperimentRuns",
        study_name,
        voxel_spacing_str,
        f"{mode}.zarr",
    )

    # Open the top-level Zarr group
    store = zarr.DirectoryStore(zarr_path)
    zgroup = zarr.open(store, mode="r")

    #
    # Typically, you'll see something like zgroup[0][0][0] or zgroup['0']['0']['0']
    # for the actual volume data, but it depends on how your Zarr store is structured.
    # Let’s assume the final data is at zgroup[0][0][0].
    #
    # You may need to inspect your actual Zarr structure and adjust accordingly.
    #
    volume = zgroup[0]  # read everything into memory

    return np.asarray(volume)


@dataclasses.dataclass
class AccumulatedObjectDetectionPredictionContainer:
    scores: List[Tensor]
    offsets: List[Tensor]
    counter: List[Tensor]
    strides: List[int]
    window_size: Tuple[int, int, int]
    use_weighted_average: bool
    weight_tensor: List[Tensor] = None

    @classmethod
    def from_shape(
            cls,
            shape: Tuple[int, int, int],
            window_size: Tuple[int, int, int],
            num_classes: int,
            strides: List[int],
            device="cpu",
            dtype=torch.float32,
            use_weighted_average: bool = False,
    ):
        d, h, w = shape

        # fmt: off
        return cls(
            scores=[torch.zeros((num_classes, d // stride, h // stride, w // stride), device=device, dtype=dtype) for stride in strides],
            offsets=[torch.zeros((3, d // stride, h // stride, w // stride), device=device, dtype=dtype) for stride in strides],
            counter=[torch.zeros(d // stride, h // stride, w // stride, device=device, dtype=dtype) for stride in strides],
            strides=list(strides),
            window_size=window_size,
            use_weighted_average=use_weighted_average,
        )
        # fmt: on

    def __post_init__(self):
        if self.use_weighted_average:
            output_window_sizes = [
                (self.window_size[0] // s, self.window_size[1] // s, self.window_size[2] // s) for s in self.strides
            ]
            self.weight_tensors = [
                self.compute_weight_matrix(torch.zeros((1, *s), device=self.scores[0].device)) for s in output_window_sizes
            ]

    def __iadd__(self, other):
        if self.strides != other.strides:
            raise ValueError("Strides mismatch")
        if self.use_weighted_average != other.use_weighted_average:
            raise ValueError("use_weighted_average mismatch")
        if self.window_size != other.window_size:
            raise ValueError("Window size mismatch")

        for i in range(len(self.scores)):
            self.scores[i] += other.scores[i].to(self.scores[i].device)
            self.offsets[i] += other.offsets[i].to(self.offsets[i].device)
            self.counter[i] += other.counter[i].to(self.counter[i].device)

        return self

    def accumulate_batch(self, batch_scores, batch_offsets, batch_tile_coords):
        batch_size = len(batch_tile_coords)
        for i in range(batch_size):
            tile_coord = batch_tile_coords[i]
            self.accumulate(
                scores_list=[s[i] for s in batch_scores],
                offsets_list=[o[i] for o in batch_offsets],
                tile_coords_zyx=tile_coord,
            )

    def accumulate(self, scores_list: List[Tensor], offsets_list: List[Tensor], tile_coords_zyx):
        if len(scores_list) != len(self.scores):
            raise ValueError(f"Number of feature maps mismatch. Scores list has size {len(scores_list)}. Number of accumulator scores {len(self.scores)}")
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

            strided_offsets_zyx = tuple(map(int, tile_coords_zyx // stride))

            roi = (
                slice(strided_offsets_zyx[0], strided_offsets_zyx[0] + scores.shape[1]),
                slice(strided_offsets_zyx[1], strided_offsets_zyx[1] + scores.shape[2]),
                slice(strided_offsets_zyx[2], strided_offsets_zyx[2] + scores.shape[3]),
            )

            scores_view = self.scores[i][:, roi[0], roi[1], roi[2]]
            offsets_view = self.offsets[i][:, roi[0], roi[1], roi[2]]
            counter_view = self.counter[i][roi[0], roi[1], roi[2]]

            # Crop tile_scores to the view shape
            scores = scores[:, : scores_view.shape[1], : scores_view.shape[2], : scores_view.shape[3]]
            offsets = offsets[:, : offsets_view.shape[1], : offsets_view.shape[2], : offsets_view.shape[3]]

            if self.use_weighted_average:
                weight_matrix = self.weight_tensors[i]
                weight_view = weight_matrix[: scores.shape[1], : scores.shape[2], : scores.shape[3]] # Crop weight matrix to shape of predicted tensor
            else:
                weight_view = 1

            counter_view += weight_view
            scores_view += scores.to(scores_view.device) * weight_view
            offsets_view += offsets.to(offsets_view.device) * weight_view



    @classmethod
    def compute_weight_matrix(self, scores_volume: Tensor, sigma=15):
        """
        :param scores_volume: Tensor of shape (C, D, H, W)
        :return: Tensor of shape (D, H, W)
        """
        center = torch.tensor(
            [
                scores_volume.shape[1] / 2,
                scores_volume.shape[2] / 2,
                scores_volume.shape[3] / 2,
                ]
        )

        i = torch.arange(scores_volume.shape[1], device=scores_volume.device)
        j = torch.arange(scores_volume.shape[2], device=scores_volume.device)
        k = torch.arange(scores_volume.shape[3], device=scores_volume.device)

        I, J, K = torch.meshgrid(i, j, k, indexing="ij")
        distances = torch.sqrt((I - center[0]) ** 2 + (J - center[1]) ** 2 + (K - center[2]) ** 2)
        weight = torch.exp(-distances / (sigma**2))

        # I just like the look of heatmap
        return weight**3

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


def anchors_for_offsets_feature_map(offsets, stride):
    z, y, x = torch.meshgrid(
        torch.arange(offsets.size(-3), device=offsets.device),
        torch.arange(offsets.size(-2), device=offsets.device),
        torch.arange(offsets.size(-1), device=offsets.device),
        indexing="ij",
    )
    anchors = torch.stack([x, y, z], dim=0)
    anchors = anchors.float().add_(0.5).mul_(stride)

    anchors = anchors[None, ...].repeat(offsets.size(0), 1, 1, 1, 1)
    return anchors

def keypoint_similarity(pts1, pts2, sigmas):
    """
    Compute similarity between two sets of keypoints
    :param pts1: ...x3
    :param pts2: ...x3
    """
    d = ((pts1 - pts2) ** 2).sum(dim=-1, keepdim=False)  # []
    e: Tensor = d / (2 * sigmas**2)
    iou = torch.exp(-e)
    return iou

def decode_detections(logits: Tensor | List[Tensor], offsets: Tensor | List[Tensor], strides: int | List[int]):
    """
    Decode detections from logits and offsets
    :param logits: Predicted logits B C D H W
    :param offsets: Predicted offsets B 3 D H W
    :param anchors: Stride of the network

    :return: Tuple of probas and centers:
             probas - B N C
             centers - B N 3

    """
    if torch.is_tensor(logits):
        logits = [logits]
    if torch.is_tensor(offsets):
        offsets = [offsets]
    if isinstance(strides, int):
        strides = [strides]

    anchors = [anchors_for_offsets_feature_map(offset, s) for offset, s in zip(offsets, strides)]

    logits_flat = []
    centers_flat = []
    anchors_flat = []

    for logit, offset, anchor in zip(logits, offsets, anchors):
        centers = anchor + offset

        logits_flat.append(einops.rearrange(logit, "B C D H W -> B (D H W) C"))
        centers_flat.append(einops.rearrange(centers, "B C D H W -> B (D H W) C"))
        anchors_flat.append(einops.rearrange(anchor, "B C D H W -> B (D H W) C"))

    logits_flat = torch.cat(logits_flat, dim=1)
    centers_flat = torch.cat(centers_flat, dim=1)
    anchors_flat = torch.cat(anchors_flat, dim=1)

    return logits_flat, centers_flat, anchors_flat


def centernet_heatmap_nms(
        scores,
        kernel: Union[int, Tuple[int, int, int]] = 3,
        #kernel: Union[int, Tuple[int, int, int]] = (5,3,3)
):
    kernel = as_tuple_of_3(kernel)
    pad = (kernel[0] - 1) // 2, (kernel[1] - 1) // 2, (kernel[2] - 1) // 2

    maxpool = torch.nn.functional.max_pool3d(scores, kernel_size=kernel, padding=pad, stride=1)

    mask = scores == maxpool
    peaks = scores * mask
    return peaks



@torch.no_grad()
def decode_detections_with_nms(
        scores: List[Tensor],
        offsets: List[Tensor],
        strides: List[int],
        min_score: Union[float, List[float]],
        class_sigmas: List[float],
        iou_threshold: float = 0.25,
        use_single_label_per_anchor: bool = True,
        use_centernet_nms: bool = False,
        pre_nms_top_k: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Decode detections from scores and centers with NMS

    :param scores: Predicted scores of shape (C, D, H, W)
    :param offsets: Predicted offsets of shape (3, D, H, W)
    :param min_score: Minimum score to consider a detection
    :param class_sigmas: Class sigmas (class radius for NMS), length = number of classes
    :param iou_threshold: Threshold above which detections are suppressed

    :return:
        - final_centers [N, 3] (x, y, z)
        - final_labels [N]
        - final_scores [N]
    """

    # Number of classes is the second dimension of `scores`
    # e.g. scores shape = (C, D, H, W)
    num_classes = scores[0].shape[0]  # the 'C' dimension

    # Allow min_score to be a single value or a list of values
    min_score = np.asarray(min_score, dtype=np.float32).reshape(-1)
    if len(min_score) == 1:
        min_score = np.full(num_classes, min_score[0], dtype=np.float32)

    if use_centernet_nms:
        scores = [centernet_heatmap_nms(s.unsqueeze(0)).squeeze(0) for s in scores]

    scores, centers, _ = decode_detections([s.unsqueeze(0) for s in scores], [o.unsqueeze(0) for o in offsets], strides)
    scores = scores.squeeze(0)
    centers = centers.squeeze(0)

    labels_of_max_score = scores.argmax(dim=1)

    # Prepare final outputs
    final_labels_list = []
    final_scores_list = []
    final_centers_list = []

    # NMS per class
    for class_index in range(num_classes):
        sigma_value = float(class_sigmas[class_index])  # Get the sigma for this class
        score_threshold = float(min_score[class_index])
        score_mask = scores[:, class_index] >= score_threshold  # Filter out low-scoring detections

        if use_single_label_per_anchor:
            class_mask = labels_of_max_score.eq(class_index)  # Pick out only detections of this class
            mask = class_mask & score_mask
        else:
            mask = score_mask

        if not mask.any():
            continue

        class_scores = scores[mask, class_index]  # shape: [Nc]
        class_centers = centers[mask]  # shape: [Nc, 3]

        if pre_nms_top_k is not None and len(class_scores) > pre_nms_top_k:
            class_scores, sort_idx = torch.topk(class_scores, pre_nms_top_k, largest=True, sorted=True)
            class_centers = class_centers[sort_idx]
        else:
            class_scores, sort_idx = class_scores.sort(descending=True)
            class_centers = class_centers[sort_idx]

        # Run a simple “greedy” NMS
        keep_indices = []
        suppressed = torch.zeros_like(class_scores, dtype=torch.bool)  # track suppressed

        # print(f"Predictions for class {class_index}: ", torch.count_nonzero(class_mask).item())

        for i in range(class_scores.size(0)):
            if suppressed[i]:
                continue
            # Keep this detection
            keep_indices.append(i)

            # Suppress detections whose IoU with i is above threshold
            iou = keypoint_similarity(class_centers[i : i + 1, :], class_centers, sigma_value)

            high_iou_mask = iou > iou_threshold
            suppressed |= high_iou_mask.to(suppressed.device)

        print(f"Predictions for class {class_index} after NMS", len(keep_indices))

        # Gather kept detections for this class
        keep_indices = torch.as_tensor(keep_indices, dtype=torch.long, device=class_scores.device)
        final_labels_list.append(torch.full((keep_indices.numel(),), class_index, dtype=torch.long))
        final_scores_list.append(class_scores[keep_indices])
        final_centers_list.append(class_centers[keep_indices])

    # Concatenate from all classes
    final_labels = torch.cat(final_labels_list, dim=0) if final_labels_list else torch.empty((0,), dtype=torch.long)
    final_scores = torch.cat(final_scores_list, dim=0) if final_scores_list else torch.empty((0,))
    final_centers = torch.cat(final_centers_list, dim=0) if final_centers_list else torch.empty((0, 3))

    print(f"Final predictions after NMS: {final_centers.size(0)}")
    return final_centers, final_labels, final_scores

def infer_num_classes_from_logits(logits):
    if not torch.is_tensor(logits):
        logits = logits[0]

    b,c,d,h,w = logits.size()
    return int(c)


def flip_volume(volume, dim):
    """
    Flip the volume along the specified dimension.
    :param volume: Volume to flip. B C D H W shape
    :param dim: Dimension to flip
    """
    return volume.flip(dim)


def flip_offsets(offsets, dim, offset_dim):
    offsets_flip = torch.flip(offsets, [dim]).clone()
    offsets_flip[:, offset_dim] *= -1  # Flip the z-offsets
    return offsets_flip


def z_flip_volume(volume):
    return flip_volume(volume, 2)


def z_flip_offsets(offsets):
    return flip_offsets(offsets, 2, 2)


def y_flip_volume(volume):
    return flip_volume(volume, 3)


def y_flip_offsets(offsets):
    return flip_offsets(offsets, 3, 1)


def x_flip_volume(volume):
    return flip_volume(volume, 4)


def x_flip_offsets(offsets):
    return flip_offsets(offsets, 4, 0)


@torch.no_grad()
def predict_scores_offsets_from_volume_using_ort(
        session: ort.InferenceSession,
        volume: np.ndarray,
        batch_size,
        torch_device,
        num_workers,
        output_strides,
        study_name,
        torch_dtype,
        use_weighted_average,
        window_size: Tuple[int, int, int],
        tiles_per_dim: Tuple[int, int, int],
        use_z_flip_tta: bool,
        use_y_flip_tta: bool,
        use_x_flip_tta: bool,
):
    container = None
    volume = normalize_volume_to_unit_range(volume)
    ds = TileDataset(volume, window_size, tiles_per_dim, dtype=np.float16, return_tensors="np")

    container = AccumulatedObjectDetectionPredictionContainer.from_shape(
        shape=volume.shape,
        num_classes=6, # Hard-coded
        window_size=window_size,
        use_weighted_average=use_weighted_average,
        strides=output_strides,
        device=torch_device,
        dtype=torch_dtype,
    )

    for tile_index in tqdm(range(len(ds)), desc=f"{study_name} {volume.shape}"):
        sample = ds[tile_index]
        tile_volume, tile_offsets = sample

        args = dict(volume = tile_volume[None, :,:,:]) # Add batch dimension

        # TODO: For simplicity we don't use io_binding, but we should once everything works
        probas, offsets = session.run(["scores", "offsets"], args)
        probas = [torch.from_numpy(probas).to(device=torch_device)]
        offsets = [torch.from_numpy(offsets).to(device=torch_device)]

        container.accumulate_batch(probas, offsets, [tile_offsets])

    scores, offsets = container.merge_()
    return scores, offsets


def postprocess_scores_offsets_into_submission(
        scores,
        offsets,
        iou_threshold,
        output_strides,
        score_thresholds,
        study_name,
        use_centernet_nms,
        use_single_label_per_anchor,
        pre_nms_top_k: int,
):
    topk_coords_px, topk_clses, topk_scores = decode_detections_with_nms(
        scores=scores,
        offsets=offsets,
        strides=output_strides,
        class_sigmas=TARGET_SIGMAS,
        min_score=score_thresholds,
        iou_threshold=iou_threshold,
        use_centernet_nms=use_centernet_nms,
        use_single_label_per_anchor=use_single_label_per_anchor,
        pre_nms_top_k=pre_nms_top_k,
    )
    topk_scores = topk_scores.float().cpu().numpy()
    top_coords = topk_coords_px.float().cpu().numpy() * ANGSTROMS_IN_PIXEL
    topk_clses = topk_clses.cpu().numpy()
    submission = dict(
        experiment=[],
        particle_type=[],
        score=[],
        x=[],
        y=[],
        z=[],
    )
    for cls, coord, score in zip(topk_clses, top_coords, topk_scores):
        submission["experiment"].append(study_name)
        submission["particle_type"].append(CLASS_LABEL_TO_CLASS_NAME[int(cls)])
        submission["score"].append(float(score))
        submission["x"].append(float(coord[0]))
        submission["y"].append(float(coord[1]))
        submission["z"].append(float(coord[2]))
    submission = pd.DataFrame.from_dict(submission)
    return submission


@torch.no_grad()
@torch.jit.optimized_execution(False)
def predict_volume(
        *,
        volume: np.ndarray,
        session: ort.InferenceSession,
        output_strides: List[int],
        window_size: Tuple[int, int, int],
        tiles_per_dim: Tuple[int, int, int],
        study_name: str,
        score_thresholds: Union[float, List[float]],
        iou_threshold,
        batch_size,
        num_workers,
        use_weighted_average,
        use_centernet_nms,
        use_single_label_per_anchor,
        torch_device: str,
        torch_dtype,
        pre_nms_top_k,
        use_z_flip_tta: bool,
        use_y_flip_tta: bool,
        use_x_flip_tta: bool,
):
    scores, offsets = predict_scores_offsets_from_volume_using_ort(
        volume=volume,
        session=session,
        output_strides=output_strides,
        window_size=window_size,
        tiles_per_dim=tiles_per_dim,
        batch_size=batch_size,
        num_workers=num_workers,
        torch_device=torch_device,
        torch_dtype=torch_dtype,
        study_name=study_name,
        use_weighted_average=use_weighted_average,
        use_z_flip_tta=use_z_flip_tta,
        use_y_flip_tta=use_y_flip_tta,
        use_x_flip_tta=use_x_flip_tta,
    )

    submission = postprocess_scores_offsets_into_submission(
        scores=scores,
        offsets=offsets,
        iou_threshold=iou_threshold,
        output_strides=output_strides,
        score_thresholds=score_thresholds,
        study_name=study_name,
        use_centernet_nms=use_centernet_nms,
        use_single_label_per_anchor=use_single_label_per_anchor,
        pre_nms_top_k=pre_nms_top_k,
    )
    return submission

def main_inference_entry_point(
        *,
        ensemble:str,
        trt_cache_path: str,
        score_thresholds,
        device_id: int,
        world_size: int = 2,
        tiles_per_dim = (1,9,9),
        output_strides:List[int] = (2,),
        window_size = (192, 128, 128),
        iou_threshold: float = 0.85,
        use_weighted_average: bool = True,
        use_centernet_nms: bool = True,
        use_single_label_per_anchor: bool = False,
        use_z_flip_tta: bool = False,
        use_y_flip_tta: bool = False,
        use_x_flip_tta: bool = False,
        pre_nms_top_k: int = 16536,
        batch_size = 1,
        num_workers = 0,
        torch_dtype = torch.float16,
        data_path="/kaggle/input/czii-cryo-et-object-identification",
        split: str = "test",
):
    device_id = int(device_id)
    torch_device = torch.device(f"cuda:{device_id}") # Build torch device that matches device_id

    trt_kwargs = {
        'device_id': device_id,
        'trt_fp16_enable': True,
        'trt_max_workspace_size': 12 * 1073741824,
        'trt_builder_optimization_level': 4,
    }

    if trt_cache_path is not None:
        trt_kwargs.update(
            trt_force_timing_cache = True,
            trt_timing_cache_enable = True,
            trt_engine_cache_enable = True,
            trt_timing_cache_path= os.path.join(trt_cache_path, 'trt_timing_cache'),
            trt_engine_cache_path= os.path.join(trt_cache_path, 'trt_engine_cache'),
            #trt_ep_context_file_path = trt_cache_path,
        )

    sess_options = ort.SessionOptions()
    session = ort.InferenceSession(
        path_or_bytes=ensemble,
        providers=[('TensorrtExecutionProvider', trt_kwargs)],
        sess_options=sess_options
    )

    path = Path(data_path)
    studies_path = path / split / "static" / "ExperimentRuns"

    studies = list(sorted(os.listdir(studies_path)))
    studies = studies[device_id::world_size] # Hopefully this is correct
    print("Process got", len(studies), "to process")

    submissions = []

    for study_name in studies:
        study_volume = get_volume(
            root_dir=path,
            study_name=study_name,
            mode="denoised",
            split=split,
        )

        study_sub = predict_volume(
            session=session,
            volume=study_volume,
            study_name=study_name,
            output_strides=output_strides,

            window_size=window_size,
            tiles_per_dim=tiles_per_dim,
            use_weighted_average=use_weighted_average,
            use_centernet_nms=use_centernet_nms,
            use_single_label_per_anchor=use_single_label_per_anchor,
            pre_nms_top_k=pre_nms_top_k,

            use_z_flip_tta=use_z_flip_tta,
            use_y_flip_tta=use_y_flip_tta,
            use_x_flip_tta=use_x_flip_tta,

            score_thresholds=score_thresholds,
            iou_threshold=iou_threshold,

            batch_size=batch_size,
            num_workers=num_workers,

            torch_device=torch_device,
            torch_dtype=torch_dtype,
        )

        submissions.append(study_sub)

    submission = pd.concat(submissions)
    return submission


def main(device_id):

    cudnn_libc = cdll.LoadLibrary("/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib/libcudnn.so.9")
    cublas_libc = cdll.LoadLibrary("/usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib/libcublas.so.12")
    cublaslt_libc = cdll.LoadLibrary("/usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib/libcublasLt.so.12")
    cudart_libc = cdll.LoadLibrary("/usr/local/lib/python3.10/dist-packages/nvidia/cuda_runtime/lib/libcudart.so.12")
    cufft_libc = cdll.LoadLibrary("/usr/local/lib/python3.10/dist-packages/nvidia/cufft/lib/libcufft.so.11")
    trt = cdll.LoadLibrary("/usr/local/lib/python3.10/dist-packages/tensorrt_libs/libnvinfer.so.10")
    trt_libnvonnxparser = cdll.LoadLibrary("/usr/local/lib/python3.10/dist-packages/tensorrt_libs/libnvonnxparser.so.10")

    import tensorrt
    print(tensorrt.__version__)
    assert tensorrt.Builder(tensorrt.Logger())

    submission = main_inference_entry_point(
        ensemble="/kaggle/input/compile-and-cache-ensembe/v4_segresnet_dynunet_ensemble_1x192x128x128_ctx.onnx",
        trt_cache_path="/kaggle/input/compile-and-cache-ensembe/v4_segresnet_dynunet_ensemble_1x192x128x128",
        tiles_per_dim=(1, 11, 11),
        score_thresholds=[0.255,0.235,0.16 ,0.205,0.225, 0.5], # LB: 784 V4 OOF Computed CV score: 0.8295528641195601 std: 0.01879723638715648
        device_id=device_id,
    )

    submission.to_csv(f"submission_shard_{device_id}.csv", index=False)


if __name__ == "__main__":
    Fire(main)

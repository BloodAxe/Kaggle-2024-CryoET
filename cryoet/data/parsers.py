import dataclasses
import json
import math
import os
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import zarr
from matplotlib.patches import Circle

from .functional import normalize_volume_to_unit_range

ANGSTROMS_IN_PIXEL = 10.012

TARGET_5_CLASSES = (
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
)


# Note we add beta-amylase as the 6th class (last one)
TARGET_6_CLASSES = TARGET_5_CLASSES + (
    {"name": "beta-amylase", "label": 5, "color": [153, 63, 0, 128], "radius": 65, "map_threshold": 0.035},
)

CLASS_LABEL_TO_CLASS_NAME = {c["label"]: c["name"] for c in TARGET_6_CLASSES}
TARGET_SIGMAS = [c["radius"] / ANGSTROMS_IN_PIXEL for c in TARGET_6_CLASSES]


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


def get_annotations(
    root_dir: str,
    study_name: str,
    use_6_classes: bool,
    split: str = "train",
):
    """
    Reads all JSON annotation files in the overlay/Picks directory for the given study,
    and returns object centers (x,y,z), the corresponding class labels, and radii.

    :param root_dir: Base directory (e.g., /path/to/czii-cryo-et-object-identification).
    :param study_name: For example, "TS_5_4".
    :param split: "train" or "test".
    :return: (centers, labels, radii)
             centers = np.array of shape (N, 3)
             labels = np.array of shape (N,)
             radii  = np.array of shape (N,)
    """

    # Build a quick lookup from object "name" -> (label, radius, …)
    target_classes = TARGET_6_CLASSES if use_6_classes else TARGET_5_CLASSES
    class_dict = {c["name"]: {"label": c.get("label", -1), "radius": c.get("radius", 0)} for c in target_classes}

    # e.g., /.../train/overlay/ExperimentRuns/TS_5_4/Picks/
    picks_dir = os.path.join(root_dir, split, "overlay", "ExperimentRuns", study_name, "Picks")

    # Collect data
    centers = []
    labels = []
    radii = []

    json_files = [os.path.join(picks_dir, f"{name}.json") for name in class_dict.keys()]
    for jf in json_files:
        with open(jf, "r") as f:
            data = json.load(f)

        # The JSON schema has:
        #   "pickable_object_name": "apo-ferritin", etc.
        #   "points": [ { "location": {...}, "instance_id": ... }, ... ]
        object_name = data.get("pickable_object_name", None)
        if not object_name or "points" not in data:
            continue

        # Look up label & radius from class_dict
        if object_name in class_dict:
            label_val = class_dict[object_name]["label"]
            radius_val = class_dict[object_name]["radius"]
        else:
            raise RuntimeError(f"Unknown object name: {object_name}")

        # Each "point" has a "location" dict with x,y,z
        for p in data["points"]:
            loc = p["location"]
            x, y, z = loc["x"], loc["y"], loc["z"]
            centers.append([x, y, z])
            labels.append(label_val)
            radii.append(radius_val)

    # Convert to NumPy arrays
    centers = np.array(centers, dtype=np.float32) if centers else np.zeros((0, 3))
    labels = np.array(labels, dtype=np.int32) if labels else np.zeros((0,))
    radii = np.array(radii, dtype=np.float32) if radii else np.zeros((0,))

    return centers, labels, radii


def get_volume_and_objects(
    root_dir: str | Path,
    study_name: str,
    use_6_classes: bool,
    mode: str = "denoised",
    split: str = "train",
):
    """
    Convenience function:
      1) Loads the entire 3D volume for the given study & mode.
      2) Loads the object centers, labels, and radii from the JSON picks.

    Returns (volume, centers, labels, radii).
    """
    volume = get_volume(root_dir, study_name, mode=mode, split=split)
    centers, labels, radii = get_annotations(
        root_dir,
        study_name,
        use_6_classes=use_6_classes,
        split=split,
    )
    return volume, centers, labels, radii


def build_label_to_color_map(target_classes):
    """
    Build a dict mapping from integer label -> (r, g, b, a) in [0,1].
    """
    label_to_color = {}
    for cls in target_classes:
        label = cls["label"]
        # Original color is e.g. [255, 204, 153] in 0–255
        color_255 = cls["color"]  # [R, G, B]
        # Convert to float in 0–1 for Matplotlib
        color_float = tuple(c / 255.0 for c in color_255)
        label_to_color[label] = color_float
    return label_to_color


def visualize_slices_grid(
    volume: np.ndarray,
    centers: np.ndarray,
    labels: np.ndarray,
    radius: np.ndarray,
    target_classes=None,
    slices_to_show=None,
    only_slices_with_objects: bool = False,
    voxel_size=10.0,
    ncols=3,
    figsize=None,
):
    """
    Visualize Z-slices of a 3D volume in a grid and draw circles for object centers.
    Each slice is drawn once, with all objects on that slice.

    :param volume: 3D NumPy array of shape (Z, Y, X).
    :param centers: (N, 3) array of XYZ object centers (in Å).
    :param labels: (N,) array of integer labels.
    :param radius: (N,) array of object radii (in Å).
    :param target_classes: A list of dicts describing classes. Example element:
                          {
                              "name": "apo-ferritin",
                              "label": 1,
                              "color": [0, 117, 220, 128],
                              "radius": 60,
                              ...
                          }
    :param slices_to_show: Optional list of integer Z-slice indices to display.
                           If None, we derive from annotation centers.
    :param only_slices_with_objects: If True, only show slices that contain at least one object.
    :param voxel_size: Real-world size of one voxel in Å (default=10.0).
    :param ncols: Number of columns in the subplot grid.
    :param figsize: Tuple (width, height) for the figure size in inches.
    :return: Matplotlib Figure containing the subplots.
    """
    if target_classes is None:
        target_classes = TARGET_6_CLASSES

    # 1) Build label -> color map from the target_classes
    label_to_color = build_label_to_color_map(target_classes)

    # 2) Convert each annotation's Z from Å to voxel index (rounded).
    z_voxels = np.round(centers[:, 2] / voxel_size + 0.5).astype(int)

    # If slices_to_show is not provided, gather from the annotation Z-coords
    if slices_to_show is None:
        slices_to_show = np.arange(volume.shape[0])

    # If only_slices_with_objects, then filter out any slice not containing an object
    if only_slices_with_objects:
        slices_with_objects = sorted(set(z_voxels.tolist()))
        slices_to_show = [s for s in slices_to_show if s in slices_with_objects]

    if len(slices_to_show) == 0:
        print("No slices to display. Returning None.")
        return None

    # 3) Setup subplots
    num_slices = len(slices_to_show)
    nrows = math.ceil(num_slices / ncols)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols * 6, nrows * 6),
        squeeze=False,
    )

    # Flatten for easy iteration
    axes_flat = axes.ravel()

    # 4) For each slice to show, draw that slice in one subplot
    for i, z_slice in enumerate(slices_to_show):
        if i >= len(axes_flat):
            break

        ax = axes_flat[i]
        ax.clear()

        # If slice is within volume range, draw it
        if 0 <= z_slice < volume.shape[0]:
            slice_img = volume[z_slice, :, :]
            ax.imshow(slice_img, cmap="gray", origin="lower")
            ax.set_title(f"Z-slice = {z_slice}", fontsize=9)

            # 5) Overlay all objects that fall on this slice
            #    We check z_voxels == z_slice
            mask_slice = z_voxels == z_slice
            idx_slice = np.where(mask_slice)[0]  # indices of objects on this slice

            for idx_obj in idx_slice:
                x_vox = centers[idx_obj, 0] / voxel_size
                y_vox = centers[idx_obj, 1] / voxel_size
                r_vox = radius[idx_obj] / voxel_size

                # Get the object label and corresponding color
                lbl = labels[idx_obj]
                color_rgba = label_to_color.get(lbl, (1.0, 0.0, 0.0, 1.0))  # default: red if not found

                circ = Circle(
                    (x_vox, y_vox),
                    radius=r_vox,
                    fill=False,
                    edgecolor=color_rgba,  # pass the RGBA color
                    linewidth=2,
                )
                ax.add_patch(circ)
        else:
            # Out of bounds => blank subplot
            ax.imshow(np.zeros((1, 1)), cmap="gray", origin="lower")
            ax.set_title(f"Z-slice = {z_slice}\n(Out of bounds)", fontsize=9)

        ax.axis("off")

    # Hide any leftover empty subplots
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # 6) Use tight layout
    fig.tight_layout()

    # Return the figure so the caller can show/save
    return fig


@dataclasses.dataclass
class AnnotatedVolume:
    study: str
    mode: str
    split: str

    volume: np.ndarray

    centers: np.ndarray
    labels: np.ndarray
    radius: np.ndarray

    centers_px: np.ndarray
    radius_px: np.ndarray

    @property
    def volume_shape(self) -> Tuple[int, int, int]:
        depth, height, width = self.volume.shape
        return (depth, height, width)


def read_annotated_volume(root, study, mode, use_6_classes: bool, split="train"):
    volume_data, object_centers, object_labels, object_radii = get_volume_and_objects(
        root_dir=root,
        study_name=study,
        mode=mode,
        split=split,
        use_6_classes=use_6_classes,
    )

    return AnnotatedVolume(
        study=study,
        split=split,
        mode=mode,
        volume=normalize_volume_to_unit_range(volume_data),
        centers=object_centers,
        labels=object_labels,
        radius=object_radii,
        centers_px=object_centers / ANGSTROMS_IN_PIXEL,
        radius_px=object_radii / ANGSTROMS_IN_PIXEL,
    )

import random

import numpy as np

import numpy as np
from scipy.ndimage import affine_transform
from typing import Tuple


def rotate_and_scale_volume(
        volume: np.ndarray,
        points: np.ndarray,
        scale: float,
        angles: Tuple[float, float, float],
        center: Tuple[float, float, float],
        output_shape: Tuple[int, int, int]
):
    """
    Applies a uniform scale and rotation (about X, Y, Z) to a 3D volume and 3D points.

    Parameters
    ----------
    volume : np.ndarray
        3D input volume of shape (D, H, W).
    points : np.ndarray
        N×3 array of 3D points. Each row is [z, y, x].
    scale : float
        Uniform scale factor.
    angles : Tuple[float, float, float]
        Rotation angles about X, Y, and Z axes, in degrees (assumed).
        For example: (angle_x, angle_y, angle_z).
    center : Tuple[float, float, float]
        The (z, y, x) coordinate in the input volume to serve as the center of rotation.
    output_shape : Tuple[int, int, int]
        The desired (D, H, W) shape of the output volume.
        The new “rotation center” will lie at the center of this output volume.

    Returns
    -------
    transformed_volume : np.ndarray
        The transformed 3D volume, of shape `output_shape`.
    transformed_points : np.ndarray
        The transformed 3D points (N×3).
    """
    
    # -----------------------------
    # 1) Build the forward transform T (4x4)
    # -----------------------------
    # Convert angles from degrees to radians (if you already have in radians, skip this)
    ax, ay, az = np.deg2rad(angles)
    
    # --- 1.1) Rotation matrices about X, Y, Z ---
    # Rotation about x-axis
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(ax), -np.sin(ax)],
        [0, np.sin(ax), np.cos(ax)]
    ], dtype=np.float64)
    
    # Rotation about y-axis
    Ry = np.array([
        [np.cos(ay), 0, np.sin(ay)],
        [0, 1, 0],
        [-np.sin(ay), 0, np.cos(ay)]
    ], dtype=np.float64)
    
    # Rotation about z-axis
    Rz = np.array([
        [np.cos(az), -np.sin(az), 0],
        [np.sin(az), np.cos(az), 0],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # Combined rotation (apply X, then Y, then Z)
    R = Rz @ Ry @ Rx
    
    # --- 1.2) Scale matrix ---
    S = np.diag([scale, scale, scale])
    
    # Combined scale + rotation
    SR = R @ S  # or S @ R, depending on your definition of scale→rotate order
    # Usually we do "rotate-then-scale" or "scale-then-rotate";
    # either approach is possible, just be consistent for points & volume.
    
    # --- 1.3) Forward transform in homogeneous form ---
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = SR
    
    # -----------------------------
    # 2) Translate so that `center` goes to origin, apply SR, then shift to output center
    # -----------------------------
    
    # We want:
    #    T_final =  Translate_to_output_center * SR * Translate_input_center_to_origin
    #
    # Step A: Translate input center to origin
    T_in = np.eye(4, dtype=np.float64)
    T_in[0, 3] = -center[2]  # x shift
    T_in[1, 3] = -center[1]  # y shift
    T_in[2, 3] = -center[0]  # z shift
    
    # Step B: Translate to output center
    # The center of the output volume = (D_out/2, H_out/2, W_out/2) in (z, y, x) format
    out_center_z = (output_shape[0] - 1) / 2.0
    out_center_y = (output_shape[1] - 1) / 2.0
    out_center_x = (output_shape[2] - 1) / 2.0
    
    T_out = np.eye(4, dtype=np.float64)
    T_out[0, 3] = out_center_x
    T_out[1, 3] = out_center_y
    T_out[2, 3] = out_center_z
    
    # Full forward transform
    # T_final = T_out * T * T_in
    T_final = T_out @ T @ T_in
    
    # -----------------------------
    # 3) Invert the matrix for affine_transform
    # -----------------------------
    T_inv = np.linalg.inv(T_final)
    
    # -----------------------------
    # 4) Apply affine_transform to the volume
    # -----------------------------
    # Note: affine_transform expects the shape ordering = (z, y, x)
    # and the transformation matrix to be the mapping from output->input
    # That is why we pass `T_inv[:3,:3]` and `T_inv[:3, 3]`.
    transformed_volume = affine_transform(
        volume,
        T_inv[:3, :3],  # 3x3 inverse transform
        offset=T_inv[:3, 3],  # translation component
        output_shape=output_shape,
        order=1,  # linear interpolation
        mode='constant',
        cval=0.0
    )
    
    # -----------------------------
    # 5) Transform the points (forward transform)
    # -----------------------------
    #  points are N×3 in [z, y, x].
    #  Convert to homogeneous coordinates => multiply => back to 3D
    n_points = points.shape[0]
    hom_coords = np.ones((n_points, 4), dtype=np.float64)
    hom_coords[:, 0] = points[:, 2]  # x
    hom_coords[:, 1] = points[:, 1]  # y
    hom_coords[:, 2] = points[:, 0]  # z
    
    # Apply forward transform T_final
    transformed_hom = (T_final @ hom_coords.T).T  # shape is N×4
    
    # Convert back to [z, y, x]
    transformed_points = np.zeros_like(points, dtype=np.float64)
    transformed_points[:, 0] = transformed_hom[:, 2]  # z
    transformed_points[:, 1] = transformed_hom[:, 1]  # y
    transformed_points[:, 2] = transformed_hom[:, 0]  # x
    
    return transformed_volume, transformed_points


def random_rotate90_volume(volume, labels):
    """Randomly rotate the volume and labels by 90 degrees.
    :param volume: The volume to rotate. Shape: (D, H, W)
    :param labels: The labels to rotate. Shape: (C, D, H, W)
    """
    k1, k2, k3 = random.randint(0, 3), random.randint(0, 3), random.randint(0, 3)
    
    volume = np.rot90(volume, k=k1, axes=(0, 1))
    labels = np.rot90(labels, k=k1, axes=(1, 2))
    
    volume = np.rot90(volume, k=k2, axes=(1, 2))
    labels = np.rot90(labels, k=k2, axes=(2, 3))
    
    volume = np.rot90(volume, k=k3, axes=(0, 2))
    labels = np.rot90(labels, k=k3, axes=(1, 3))
    
    return np.ascontiguousarray(volume), np.ascontiguousarray(labels)


def random_flip_volume(volume, labels):
    """Randomly flip the volume and labels.
    :param volume: The volume to rotate. Shape: (D, H, W)
    :param labels: The labels to rotate. Shape: (C, D, H, W)
    """
    if random.random() < 0.5:
        volume = np.flip(volume, axis=0)
        labels = np.flip(labels, axis=1)
    
    if random.random() < 0.5:
        volume = np.flip(volume, axis=1)
        labels = np.flip(labels, axis=2)
    
    if random.random() < 0.5:
        volume = np.flip(volume, axis=2)
        labels = np.flip(labels, axis=3)
    
    return np.ascontiguousarray(volume), np.ascontiguousarray(labels)

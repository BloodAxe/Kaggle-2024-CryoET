import random
from typing import Optional
from typing import Tuple

import numpy as np
from scipy.ndimage import affine_transform


def get_points_mask_within_cube(points, cube_shape):
    """
    Return a boolean mask for points that are within the
    :param points: Nx3 array of points in (X, Y, Z)
    :param cube_shape: (D, H, W) = (Z, Y, X)
    """
    return (
        (points[:, 0] >= 0)
        & (points[:, 0] < cube_shape[2])
        & (points[:, 1] >= 0)
        & (points[:, 1] < cube_shape[1])
        & (points[:, 2] >= 0)
        & (points[:, 2] < cube_shape[0])
    )


def euler_matrix_x(angle_deg: float) -> np.ndarray:
    """
    Return the 3x3 rotation matrix for a rotation around the X-axis by angle_deg (degrees).
    """
    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def euler_matrix_y(angle_deg: float) -> np.ndarray:
    """
    Return the 3x3 rotation matrix for a rotation around the Y-axis by angle_deg (degrees).
    """
    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def euler_matrix_z(angle_deg: float) -> np.ndarray:
    """
    Return the 3x3 rotation matrix for a rotation around the Z-axis by angle_deg (degrees).
    """
    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def euler_rotation_matrix(angles: Tuple[float, float, float]) -> np.ndarray:
    """
    Build a 3x3 rotation matrix from Euler angles (x, y, z in degrees),
    applying Rz * Ry * Rx in that order (or adapt as needed).
    """
    ax, ay, az = angles
    Rx = euler_matrix_x(ax)
    Ry = euler_matrix_y(ay)
    Rz = euler_matrix_z(az)
    # Order: X then Y then Z means final = Rz @ Ry @ Rx
    return Rz @ Ry @ Rx


def make_affine_transform_4x4(
    angles: Tuple[float, float, float], scale: float, old_center: np.ndarray, new_center: np.ndarray
) -> np.ndarray:
    """
    Create a 4x4 affine transform matrix that:
      1) Translates old_center to origin,
      2) Rotates by angles (X->Y->Z),
      3) Scales by 'scale',
      4) Translates back to new_center.

    The resulting matrix is the 'forward' transform:
        X_out = A * X_in
    where X_in, X_out are in homogeneous coords (4D).
    """
    # Build 4x4
    M = np.eye(4)
    # Step 1: Translate old_center to origin
    M[:3, 3] = -old_center

    # Step 2+3: Multiply by (R * scale)
    # This is effectively M = RS * (X - old_center), done by right multiplying:
    # We'll insert RS into an identity, then post-multiply M.
    # Rotation matrix (3x3)
    R = euler_rotation_matrix(angles)

    # Combine rotation & scale
    RS = scale * R  # 3x3
    RS_mat = np.eye(4)
    RS_mat[:3, :3] = RS

    # Step 4: Translate origin to new_center
    T_new_center = np.eye(4)
    T_new_center[:3, 3] = new_center

    # Full forward transform
    # (Translate to origin) -> (Rotate+Scale) -> (Translate to new center)
    # Mathematically: T_new_center * RS_mat * M
    return T_new_center @ RS_mat @ M


def rotate_and_scale_volume(
    volume: np.ndarray,
    points: np.ndarray,
    scale: float,
    angles: Tuple[float, float, float],
    center_zyx: Tuple[float, float, float],
    output_shape: Tuple[int, int, int],
    order: int = 1,
):
    """
    Rotate (by angles) then scale a 3D volume about 'center' in (X, Y, Z) coordinate order,
    and resample into 'output_shape' = (D, H, W).
    The new center is the midpoint of output_shape.

    Args:
        volume: 3D numpy array (D, H, W) = (Z, Y, X)
        points: Nx3 array of points in (X, Y, Z)
        scale:  float scale factor
        angles: (angle_x, angle_y, angle_z) in degrees
        center_zyx: (cz, cy, cx) = center of rotation in input volume coords
        output_shape: desired output volume shape (newD, newH, newW)
        order: interpolation order for affine_transform (0=nearest,1=linear,3=cubic,etc)

    Returns:
        new_volume: the transformed volume with shape = output_shape
        new_points: Nx3 array of the transformed points in (X, Y, Z)
    """
    # 1) Convert center to np.array
    old_center = np.array(center_zyx, dtype=np.float32)

    # 2) Figure out new center in output space (X, Y, Z order):
    #    output_shape = (newD, newH, newW) -> (Z, Y, X)
    #    new_center_x = (newW - 1) / 2
    #    new_center_y = (newH - 1) / 2
    #    new_center_z = (newD - 1) / 2
    newD, newH, newW = output_shape
    new_center = np.array([(newD - 1) / 2.0, (newH - 1) / 2.0, (newW - 1) / 2.0], dtype=np.float32)

    # 3) Create the forward 4x4 transform
    forward_M_4x4 = make_affine_transform_4x4(angles, scale, old_center, new_center)

    # 4) For volume resampling with scipy.ndimage.affine_transform, we need the
    #    "inverse" transform that goes from (x_out, y_out, z_out) -> (x_in, y_in, z_in).
    inv_M_4x4 = np.linalg.inv(forward_M_4x4)

    # 5) Build the matrix + offset in the form that affine_transform expects:
    #    output_coord = M_input2output * input_coord + offset
    #    but we have M_output2input. So for a voxel at output coords (ox, oy, oz),
    #    the corresponding input coords is:
    #        (ix, iy, iz, 1) = inv_M_4x4 @ (ox, oy, oz, 1).
    #    That means:
    #        [ix, iy, iz]^T = inv_M_4x4[:3,:3] @ [ox, oy, oz] + inv_M_4x4[:3,3]
    #
    #    Hence:
    #        matrix = inv_M_4x4[:3,:3]
    #        offset = inv_M_4x4[:3, 3]
    #
    #    BUT note scipy uses the convention (z,y,x), so we must reorder slices carefully.
    #    We'll keep dimension order = (z, y, x).

    # 6) Apply the affine transform
    #    volume shape: (D, H, W) = (Z, Y, X)
    #    output_shape is also in (D, H, W).
    new_volume = affine_transform(
        input=volume,
        matrix=inv_M_4x4,
        output_shape=output_shape,
        order=order,
        cval=0.0,  # or any fill value you prefer
        mode="constant",
    )

    # 7) Transform the points using the FORWARD matrix:
    #    points are Nx3, each row is (x, y, z).
    #    We treat them as homogeneous coords to do forward_M_4x4 @ [z, y, x, 1].
    N = points.shape[0]
    ones = np.ones((N, 1), dtype=np.float32)
    pts_hom = np.hstack([points[:, ::-1], ones])  # Nx4
    pts_transformed_hom = (forward_M_4x4 @ pts_hom.T).T  # Nx4
    new_points = pts_transformed_hom[:, :3] / pts_transformed_hom[:, [3]]

    return new_volume, new_points[:, ::-1]  # Back to xyz


def random_rotate90_volume_around_z(volume, labels):
    """Randomly rotate the volume and labels by 90 degrees.
    :param volume: The volume to rotate. Shape: (D, H, W)
    :param labels: The labels to rotate. Shape: (C, D, H, W)
    """
    k = random.randint(0, 3)

    volume = np.rot90(volume, k=k, axes=(1, 2))
    labels = np.rot90(labels, k=k, axes=(2, 3))

    return np.ascontiguousarray(volume), np.ascontiguousarray(labels)


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


def random_flip_volume(volume: np.ndarray, heatmap: Optional[np.ndarray] = None, centers: Optional[np.ndarray] = None):
    """Randomly flip the volume and labels.
    :param volume: The volume to rotate. Shape: (D, H, W)
    :param heatmap: The labels to rotate. Shape: (C, D, H, W)
    :param centers: The centers to rotate. Shape: (N, 3). Each row is (x, y, z)
    """
    if centers is not None:
        centers = centers.copy()

    if random.random() < 0.5:
        volume = np.flip(volume, axis=0)
        if heatmap is not None:
            heatmap = np.flip(heatmap, axis=1)

        if centers is not None:
            centers[:, 2] = (volume.shape[0] - 1) - centers[:, 2]

    if random.random() < 0.5:
        volume = np.flip(volume, axis=1)
        if heatmap is not None:
            heatmap = np.flip(heatmap, axis=2)
        if centers is not None:
            centers[:, 1] = (volume.shape[1] - 1) - centers[:, 1]

    if random.random() < 0.5:
        volume = np.flip(volume, axis=2)
        if heatmap is not None:
            heatmap = np.flip(heatmap, axis=3)
        if centers is not None:
            centers[:, 0] = (volume.shape[2] - 1) - centers[:, 0]

    return dict(volume=np.ascontiguousarray(volume), heatmap=heatmap, centers=centers)


def erase_objects(
    volume: np.ndarray,
    centers_px: np.ndarray,
    radius_px: np.ndarray,
    labels: np.ndarray,
    keep_mask: np.ndarray,
    remove_overlap=True,
):
    """
    Erase objects from the volume and given centers.
    Returns new volume, centers, radius and labels.
    Areas that are erased are set to 0.

    :param volume:         The volume to erase objects from. Shape: (D, H, W)
    :param centers_px:     The centers of the objects to erase. Shape: (N, 3). Each row is (x, y, z)
    :param radius_px:      The radius of the objects to erase. Shape: (N,)
    :param labels:         The labels of the objects to erase. Shape: (N,)
    :param keep_mask:      The mask indicating which points to keep (N,). Note that there are can be more removed points if remove_overlap is True.
    :param remove_overlap: If True, remove points that overlap with other points that are being removed.
                           If points that we keep lies within half the radius of a point being removed, it will also be removed.
                           We don't do chained removals.

    """
    if len(centers_px) != len(radius_px) or len(centers_px) != len(labels) or len(centers_px) != len(keep_mask):
        raise ValueError("centers_px, radius_px and labels must have the same length.")

    # Compute a volume mask for erasion.
    # Step 1 - build meshgrid of the volume
    z, y, x = volume.shape
    z, y, x = np.meshgrid(np.arange(z), np.arange(y), np.arange(x), indexing="ij")

    grid = np.stack([x, y, z], axis=-1)
    radius_sqr = radius_px**2

    erase_mask = ~keep_mask
    mask_volume = np.zeros_like(volume, dtype=bool)
    for center, radius_squared in zip(centers_px[erase_mask], radius_sqr[erase_mask]):
        distances_sqr = np.sum((grid - center[None, None, None, :]) ** 2, axis=-1)
        mask_volume |= distances_sqr < radius_squared

    volume = volume.copy()
    volume[mask_volume] = 0

    if remove_overlap:
        half_radius_sqr = (radius_px * 0.5) ** 2

        # Compute matrix of pairwise distances
        distances_sqr_mask = np.sum((centers_px[:, None] - centers_px[None, :]) ** 2, axis=-1) < half_radius_sqr[:, None]
        np.fill_diagonal(distances_sqr_mask, False)

        distances_sqr_mask_masked = distances_sqr_mask & erase_mask[:, None]
        # Update mask to remove points that are inside the radius of the point being removed
        overlap_mask = np.any(distances_sqr_mask_masked, axis=0) & keep_mask

        # mask out everything that overlaps
        erase_mask = erase_mask | overlap_mask

    keep_mask = ~erase_mask
    return dict(volume=volume, centers=centers_px[keep_mask], radius=radius_px[keep_mask], labels=labels[keep_mask])


def random_erase_objects(volume: np.ndarray, centers_px: np.ndarray, radius_px: np.ndarray, labels: np.ndarray, prob: float):
    """
    :param volume: The volume to erase objects from. Shape: (D, H, W)
    :param centers_px: The centers of the objects to erase. Shape: (N, 3). Each row is (x, y, z)
    :param radius_px: The radius of the objects to erase. Shape: (N,)
    :param labels: The labels of the objects to erase. Shape: (N,)
    :param prob: The probability of erasing each object.
    """
    keep_mask = ~np.array([(random.random() < prob) for _ in range(len(centers_px))], dtype=bool)
    data = erase_objects(volume, centers_px, radius_px, labels, keep_mask, remove_overlap=True)
    return data["volume"], data["centers"], data["radius"], data["labels"]

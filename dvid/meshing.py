# This code is part of dvid-tools (https://github.com/flyconnectome/dvid_tools)
# and is released under GNU GPL3

import numpy as np
import sparsecubes as sc

from scipy.ndimage import binary_erosion, binary_fill_holes


def mesh_from_voxels(voxels, spacing=(1, 1, 1), step_size=1, smooth=True,
                     simplify=False):
    """Generate mesh from voxels using sparse-cubes.

    Unlike marching cubes, sparse-cubes operates directly on the sparse voxel
    coordinates which is faster and much more memory efficient - there is no
    need to rasterize the voxels into a dense 3D matrix or process them in
    chunks.

    Parameters
    ----------
    voxels :        np.array
                    (N, 3) array of x/y/z voxel indices.
    spacing :       np.array
                    (3, ) array with x, y, z voxel size.
    step_size :     int, optional
                    Step size for the meshing. Higher values downsample the
                    voxels first (= faster but coarser).
    smooth :        bool, optional
                    If True (default), uses SurfaceNets to produce a smooth
                    mesh. If False, produces a blocky ("Minecraft"-style) mesh
                    with axis-aligned faces.
    simplify :      bool, optional
                    Only relevant if ``smooth=False``: merges coplanar faces
                    into larger rectangles (greedy meshing) to reduce face
                    count.

    Returns
    -------
    trimesh.Trimesh

    """
    voxels = np.asarray(voxels)

    # sparse-cubes requires an integer (N, 3) array
    if voxels.dtype.kind not in "iu":
        voxels = voxels.astype(np.int64)

    # Pass spacing as float so smooth (fractional) vertices aren't truncated
    spacing = np.asarray(spacing, dtype=float)

    return sc.mesh(voxels,
                   spacing=spacing,
                   step_size=step_size,
                   smooth=smooth,
                   simplify=simplify)


def remove_surface_voxels(voxels, **kwargs):
    """Removes surface voxels."""
    # Generate empty array
    mat, bb_min = _voxels_to_matrix(voxels)

    # Erode
    mat_erode = binary_erosion(mat, **kwargs)

    # Turn back into voxels
    voxels_erode = _matrix_to_voxels(mat_erode) + bb_min

    return voxels_erode


def get_surface_voxels(voxels):
    """Return surface voxels."""
    # Use bounding boxes to keep matrix small
    bb_min = voxels.min(axis=0)
    #bb_max = voxels.max(axis=0)
    #dim = bb_max - bb_min

    # Voxel offset
    voxel_off = voxels - bb_min

    # Generate empty array
    mat = _voxels_to_matrix(voxel_off)

    # Erode
    mat_erode = binary_erosion(mat)

    # Substract
    mat_surface = np.bitwise_and(mat, np.invert(mat_erode))

    # Turn back into voxels
    voxels_surface = _matrix_to_voxels(mat_surface) + bb_min

    return voxels_surface


def parse_obj(obj):
    """Parse .obj string and return vertices and faces."""
    lines = obj.split('\n')
    verts = []
    faces = []
    for l in lines:
        if l.startswith('v '):
            verts.append([float(v) for v in l[2:].split(' ')])
        elif l.startswith('f '):
            f = [v.split('//')[0] for v in l[2:].split(' ')]
            faces.append([int(v) for v in f])

    # `.obj` faces start with vertex indices of 1 -> set to 0
    return np.array(verts), np.array(faces) - 1


def _voxels_to_matrix(voxels, fill=False, pad=1, dtype=bool):
    """Generate matrix from voxels/blocks.

    Parameters
    ----------
    voxels :    (N, 3) numpy array
    fill :      bool, optional
                If True, will use binary fill to fill holes in matrix.

    Returns
    -------
    numpy array
                3D matrix with x, y, z axis (blocks will be converted)

    """
    if not isinstance(voxels, np.ndarray):
        voxels = np.array(voxels)

    offset = voxels.min(axis=0)
    voxels = voxels - offset

    if voxels.shape[1] == 3:
        mat = np.zeros((voxels.max(axis=0) + 1), dtype=dtype)
        mat[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = 1
    else:
        raise ValueError('Unexpected voxel shape')

    # Fill holes
    if fill:
        mat = binary_fill_holes(mat)

    if pad:
        mat = np.pad(mat, pad_width=pad, mode='constant', constant_values=0)

    return mat, offset


def _matrix_to_voxels(matrix):
    """Turn matrix into voxels.

    Assumes that voxels have values True or 1.
    """
    # Turn matrix into voxels
    return np.vstack(np.where(matrix)).T


def _apply_mask(mat, mask):
    """Substracts (logical_xor) mask from mat.

    Assumes that both matrices are (a) of the same voxel size and (b) have the
    same origin.
    """
    # Get maximum dimension between mat and mask
    max_dim = np.max(np.array([mat.shape, mask.shape]), axis=0)

    # Bring mask in shape of mat
    mask_pad = np.vstack([np.array([0, 0, 0]), max_dim - mask.shape]).T
    mask = np.pad(mask, mask_pad, mode='constant', constant_values=0)
    mask = mask[:mat.shape[0], :mat.shape[1], :mat.shape[2]]

    # Substract mask
    return np.bitwise_and(mat, np.invert(mask))


def _mask_voxels(voxels, mask_voxels):
    """Mask voxels with other voxels.

    Assumes that both voxels have the same size and origin.
    """
    # Generate matrices of the voxels
    mat = _voxels_to_matrix(voxels)
    mask = _voxels_to_matrix(mask_voxels)

    # Apply mask
    masked = _apply_mask(mat, mask)

    # Turn matrix back into voxels
    return _matrix_to_voxels(masked)


def _blocks_to_voxels(blocks):
    """Convert blocks to voxels.

    Parameters
    ----------
    blocks :    (N, 4) array
                Blocks of [z, y, x1, x2].

    Return
    ------
    voxels :    (N, 3) array

    """
    if blocks.shape[1] != 4:
        raise ValueError(f'Expected blocks of shape (N, 4), got "{blocks.shape}"')

    # Create empty array
    n_vox = (blocks[:, 3] - blocks[:, 2] + 1)
    voxels = np.zeros((n_vox.sum(), 3), dtype=int)

    # Go over each block and populate voxels
    start = 0
    for i, bl in enumerate(blocks):
        end = start + n_vox[i]
        voxels[start:end, 1] = bl[1]
        voxels[start:end, 2] = bl[0]
        voxels[start:end, 0] = np.arange(bl[2], bl[3] + 1)
        start = end
    return voxels

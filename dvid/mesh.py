# This code is part of dvid-tools (https://github.com/flyconnectome/dvid_tools)
# and is released under GNU GPL3

import numpy as np

from skimage.measure import marching_cubes_lewiner
from scipy.ndimage.morphology import binary_erosion, binary_fill_holes


def mesh_from_voxels(voxels, v_size, step_size=1):
    """Generate mesh from voxels using marching cubes.

    Parameters
    ----------
    voxels :    np.array
                Voxel coordindates. Array of either (N, 3) or (N, 4).
                (N, 3) will be interpreted as single x, y, z voxel coordinates.
                (N, 4) will be interpreted as blocks of z, y, x_start, x_end
                coordindates.
    v_size :    np.array
                (3, ) array with x, y, z voxel size.
    step_size : int, optional
                Step size for marching cube algorithm.
                Higher values = faster but coarser.

    Returns
    -------
    vertices :  (N, 3) numpy array
    faces :     (N, 3) numpy array

    """
    # Turn voxels into matrix
    mat = _voxels_to_matrix(voxels)

    # Add border to matrix - otherwise marching cube generates holes
    mat = np.pad(mat, pad_width=5, mode='constant', constant_values=0)

    # Use marching cubes to create surface model
    verts, faces, normals, values = marching_cubes_lewiner(mat,
                                                           level=.5,
                                                           step_size=step_size,
                                                           allow_degenerate=False,
                                                           gradient_direction='ascent',
                                                           spacing=v_size)

    # Compensate for earlier padding offset
    verts -= np.array(v_size) * 5

    return verts, faces


def remove_surface_voxels(voxels, **kwargs):
    """Removes surface voxels."""
    # Use bounding boxes to keep matrix small
    bb_min = voxels.min(axis=0)
    #bb_max = voxels.max(axis=0)
    #dim = bb_max - bb_min

    # Voxel offset
    voxel_off = voxels - bb_min

    # Generate empty array
    mat = _voxels_to_matrix(voxel_off)

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


def _voxels_to_matrix(voxels, fill=False):
    """Generate matrix from voxels/blocks.

    Parameters
    ----------
    voxels :    numpy array
                Either voxels [[x, y, z], ..] or blocks [[z, y, x1, x2], ...]
    fill :      bool, optional
                If True, will use binary fill to fill holes in matrix.

    Returns
    -------
    numpy array
                3D matrix with x, y, z axis (blocks will be converted)

    """
    if not isinstance(voxels, np.ndarray):
        voxels = np.array(voxels)

    # Populate matrix
    if voxels.shape[1] == 4:
        mat = np.zeros((voxels.max(axis=0) + 1)[[-1, 1, 0]], dtype=np.bool)
        for col in voxels:
            mat[col[2]:col[3] + 1, col[1], col[0]] = 1
    elif voxels.shape[1] == 3:
        mat = np.zeros((voxels.max(axis=0) + 1), dtype=np.bool)
        mat[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = 1
    else:
        raise ValueError('Unexpected voxel shape')

    # Fill holes
    if fill:
        mat = binary_fill_holes(mat)

    return mat


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
    return _matrix_to_voxels(_voxels_to_matrix(blocks))


def read_ngmesh(f, mutable=False):
    """Read vertices and faces from neuroglancer mesh.

    Single-resolution legacy format.

    Parameters
    ----------
    f :         File-like object
                An open binary file object.

    Returns
    -------
    vertices    (N, 3) numpy array
    faces :     (N, 3) numpy array

    """
    num_vertices = np.frombuffer(f.read(4), np.uint32)[0]
    vertices_xyz = np.frombuffer(f.read(int(3*4*num_vertices)), np.float32).reshape(-1, 3)
    faces = np.frombuffer(f.read(), np.uint32).reshape(-1, 3)

    if mutable:
        return vertices_xyz.copy(), faces.copy()
    else:
        return vertices_xyz, faces

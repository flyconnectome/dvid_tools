# This code is part of dvid-tools (https://github.com/flyconnectome/dvid_tools)
# and is released under GNU GPL3

import numpy as np
import trimesh as tm

from skimage import measure
from scipy.ndimage.morphology import binary_erosion, binary_fill_holes

from tqdm.auto import tqdm


def mesh_from_voxels(voxels,
                     spacing,
                     step_size=1,
                     chunk_size=None,
                     pad_chunks=True,
                     merge_fragments=True,
                     progress=True):
    """Generate mesh from voxels using marching cubes.

    Parameters
    ----------
    voxels :        np.array
                    Voxel coordindates. Array of either (N, 3) or (N, 4).
                    (N, 3) will be interpreted as single x, y, z voxel coordinates.
                    (N, 4) will be interpreted as blocks of z, y, x_start, x_end
                    coordindates.
    spacing :       np.array
                    (3, ) array with x, y, z voxel size.
    step_size :     int, optional
                    Step size for marching cube algorithm.
                    Higher values = faster but coarser.
    chunk_size :    int, optional
                    If `None`, will process voxels in one go which can lead to a
                    huge memory footprint. Set a `chunk_size` [in voxels] to
                    process the voxels in cubes of given side length. 200 seems
                    to be a sensible value.

    For `chunk_size != None`:

    pad_chunks :    bool
                    If True, will pad each chunk. This helps making meshes
                    watertight but may introduce internal faces when merging
                    mesh fragments.
    merge_fragments :  bool
                    If True, will attempt to merge fragments at the chunk
                    boundaries.
    progress :      bool
                    Whether to show a progress bar.

    Returns
    -------
    trimesh.Trimesh

    """
    if not chunk_size:
        return _mesh_from_voxels_single(voxels=voxels,
                                        spacing=spacing,
                                        step_size=step_size)
    else:
        return _mesh_from_voxels_chunked(voxels=voxels,
                                         spacing=spacing,
                                         step_size=step_size,
                                         chunk_size=chunk_size,
                                         pad_chunks=pad_chunks,
                                         merge_fragments=merge_fragments,
                                         progress=progress)


def _mesh_from_voxels_single(voxels, spacing=(1, 1, 1), step_size=1):
    """Generate mesh from voxels using marching cubes in one go.

    Better meshes but potentially slower and much less memory efficient.

    Parameters
    ----------
    voxels :        np.array
                    Voxel coordindates. Array of either (N, 3) or (N, 4).
                    (N, 3) will be interpreted as single x, y, z voxel coordinates.
                    (N, 4) will be interpreted as blocks of z, y, x_start, x_end
                    coordindates.
    spacing :       np.array
                    (3, ) array with x, y, z voxel size.
    step_size :     int, optional
                    Step size for marching cube algorithm.
                    Higher values = faster but coarser.

    Returns
    -------
    trimesh.Trimesh

    """
    # Turn voxels into matrix
    # Add border to matrix - otherwise marching cube generates holes
    mat, offset = _voxels_to_matrix(voxels, pad=1)

    # Use marching cubes to create surface model
    # (newer versions of skimage have a "marching cubes" function and
    # the marching_cubes_lewiner is deprecreated)
    marching_cubes = getattr(measure, 'marching_cubes', measure.marching_cubes_lewiner)
    verts, faces, normals, values = marching_cubes(mat,
                                                   level=.5,
                                                   step_size=step_size,
                                                   allow_degenerate=False,
                                                   gradient_direction='ascent',
                                                   spacing=spacing)

    # Compensate for earlier padding offset
    verts -= np.array(spacing) * 1

    # Add offset
    verts += offset * spacing

    return tm.Trimesh(verts, faces)


def _mesh_from_voxels_chunked(voxels,
                              spacing=(1, 1, 1),
                              step_size=1,
                              chunk_size=200,
                              pad_chunks=True,
                              merge_fragments=True,
                              progress=True):
    """Generate mesh from voxels in chunks using marching cubes.

    Potentially faster and much more memory efficient but might introduce
    internal faces and/or holes.

    Parameters
    ----------
    voxels :        np.array
                    Voxel coordindates. Array of (N, 3) XYZ indices.
    spacing :       np.array
                    (3, ) array with x, y, z voxel size.
    step_size :     int, optional
                    Step size for marching cube algorithm.
                    Higher values = faster but coarser.
    chunk_size :    int
                    Size of the cubes in voxels in which to process the data.
                    The bigger the chunks, the more memory is used but there is
                    less chance of errors in the mesh.
    pad_chunks :    bool
                    If True, will pad each chunk. This helps making meshes
                    watertight but may introduce internal faces when merging
                    mesh fragments.
    merge_fragments :  bool
                    If True, will attempt to merge fragments at the chunk
                    boundaries.

    Returns
    -------
    trimesh.Trimesh

    """
    # Use marching cubes to create surface model
    # (newer versions of skimage have a "marching cubes" function and
    # the marching_cubes_lewiner is deprecreated)
    marching_cubes = getattr(measure, 'marching_cubes', measure.marching_cubes_lewiner)

    # Strip the voxels
    offset = voxels.min(axis=0)
    voxels = voxels - offset

    # Find unique chunks
    chunks, inv = np.unique((voxels / chunk_size).astype(int),
                            return_inverse=True,
                            axis=0)

    # For each chunk also find voxels that are directly adjacent (in plus direction)
    inv_fuzzy = np.full((4, len(voxels)), fill_value=-1, dtype=int)
    for k in range(3):
        off = [0, 0, 0]
        off[k] = 1
        chunks_fuzzed = ((voxels - off) / chunk_size).astype(int)
        for i, ch in enumerate(chunks):
            inv_fuzzy[k, np.all(chunks_fuzzed == ch, axis=1)] = i
    inv_fuzzy[-1, :] = inv

    # Generate the fragments
    fragments = []
    end_chunks = chunks.max(axis=0)
    pad = np.array([[1, 1], [1, 1], [1, 1]])
    for i, ch in tqdm(enumerate(chunks),
                      total=len(chunks),
                      disable=not progress,
                      leave=False,
                      desc='Meshing'):
        # Pad the matrices only for the first and last chunks along each axis
        if not pad_chunks:
            pad = np.array([[0, 0], [0, 0], [0, 0]])
            for k in range(3):
                if ch[k] == 0:
                    pad[k][0] = 1
                if ch[k] == end_chunks[k]:
                    pad[k][1] = 1

        # Get voxels in this chunk
        this_vx = voxels[np.any(inv_fuzzy == i, axis=0)]

        # If only a single voxel, skip.
        if this_vx.shape[0] <= 1:
            continue

        # Turn voxels into matrix with given padding
        mat, chunk_offset = _voxels_to_matrix(this_vx, pad=pad.tolist())

        # Marching cubes needs at least a (2, 2, 2) matrix
        # We could in theory make this work by adding padding
        # but we probably don't loose much if we skip them.
        # There is a chance that we might introduce holes in the mesh
        # though
        if any([s < 2 for s in mat.shape]):
            continue

        # Run the actual marching cubes
        v, f, _, _ = marching_cubes(mat,
                                    level=.5,
                                    step_size=step_size,
                                    allow_degenerate=False,
                                    gradient_direction='ascent',
                                    spacing=(1, 1, 1))

        # Remove the padding (if any)
        v -= pad[:, 0]
        # Add chunk offset
        v += chunk_offset

        fragments.append((v, f))

    # Combine into a single mesh
    all_verts = []
    all_faces = []
    verts_offset = 0
    for frag in fragments:
        all_verts.append(frag[0])
        all_faces.append(frag[1] + verts_offset)
        verts_offset += len(frag[0])

    all_verts = np.concatenate(all_verts, axis=0)
    all_faces = np.concatenate(all_faces, axis=0)

    # Make trimesh
    m = tm.Trimesh(all_verts, all_faces)

    # Deduplicate chunk boundaries
    # This is not necessarily the cleanest solution but it should do the
    # trick most of the time
    if merge_fragments:
        m.merge_vertices(digits_vertex=1)

    return m


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


def _voxels_to_matrix(voxels, fill=False, pad=1, dtype=np.bool):
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

    offset = voxels.min(axis=0)
    voxels = voxels - offset

    # Populate matrix
    if voxels.shape[1] == 4:
        mat = np.zeros((voxels.max(axis=0) + 1)[[-1, 1, 0]], dtype=dtype)
        for col in voxels:
            mat[col[2]:col[3] + 1, col[1], col[0]] = 1
    elif voxels.shape[1] == 3:
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
    return _matrix_to_voxels(_voxels_to_matrix(blocks))


"""
# Generate chunk boundaries based on chunk size
x_chunks = np.arange(0, voxels[:, 0].max(), chunksize)
y_chunks = np.arange(0, voxels[:, 1].max(), chunksize)
z_chunks = np.arange(0, voxels[:, 2].max(), chunksize)

# Generate the fragments
fragments = []
for i, x in tqdm(enumerate(x_chunks),
                 desc='Meshing',
                 disable=not progress,
                 total=len(x_chunks),
                 leave=False):
    filt1 = (voxels[:, 0] >= x) & (voxels[:, 0] < x + chunksize + 1)
    if not any(filt1):
        continue

    # Pad the matrices for the first and last chunks along each axis
    pad = np.array([[0, 0], [0, 0], [0, 0]])

    if i == 0:
        pad[0][0] = 1
    elif i == len(x_chunks):
        pad[0][1] = 1

    for j, y in enumerate(y_chunks):
        filt2 = filt1 & (voxels[:, 1] >= y) & (voxels[:, 1] < y + chunksize + 1)
        if not any(filt2):
            continue

        if j == 0:
            pad[1][0] = 1
        elif j == len(y_chunks):
            pad[1][1] = 1

        for k, z in enumerate(z_chunks):
            filt3 = filt2 & (voxels[:, 2] >= z) & (voxels[:, 2] < z + chunksize + 1)
            if not any(filt3):
                continue

            if k == 0:
                pad[2][0] = 1
            elif k == len(z_chunks):
                pad[2][1] = 1

            # Get voxels in this chunk
            this_vx = voxels[filt3]

            # If only a single voxel, skip.
            if this_vx.shape[0] <= 1:
                continue

            # Turn voxels into matrix with given padding
            mat, chunk_offset = _voxels_to_matrix(this_vx, pad=pad.tolist())

            # Marching cubes needs at least a (2, 2, 2) matrix
            # We could in theory make this work by adding padding
            # but we probably don't loose much if we skip them.
            # There is a chance that we might introduce holes in the mesh
            # though
            if any([s < 2 for s in mat.shape]):
                continue

            # Run the actual marching cubes
            v, f, _, _ = marching_cubes(mat,
                                        level=.5,
                                        step_size=step_size,
                                        allow_degenerate=False,
                                        gradient_direction='ascent',
                                        spacing=(1, 1, 1))

            # Remove the padding (if any)
            v -= pad[:, 0]
            # Add chunk offset
            v += chunk_offset

            fragments.append((v, f))
"""

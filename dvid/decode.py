# This code is part of dvid-tools (https://github.com/flyconnectome/dvid_tools)
# and is released under GNU GPL3

import struct

import numpy as np
import trimesh as tm

from io import BytesIO


def decode_sparsevol(b, format='rles', voxels=True):
    """Decode sparsevol binary mesh format.

    Parameters
    ----------
    b :         bytes
                Data to decode.
    format :    "blocks" | "rles"
                Binary format in which the file is encoded. Only "rles" is
                supported at the moment.
    voxels :    bool
                Whether to return x/y/z voxels or original x/y/z/x_run_length rles.

    Returns
    -------
    voxel indices
                Please note that this function is agnostic to voxel size, etc.

    """
    if not isinstance(b, bytes):
        raise TypeError('Need bytes, got "{}"'.format(type(b)))

    if format == 'rles':
        # First get the header
        header = {k: v for k, v in zip(['start_byte', 'n_dims', 'run_dims',
                                        'reserved', 'n_blocks', 'n_spans'],
                                       struct.unpack('bbbbii', b[:12]))}

        rles = np.frombuffer(b, '<i4', offset=12,
                             count=header['n_spans'] * 4).reshape(-1, 4)

        if voxels:
            # Run lengths as int64 to avoid overflowing the cumulative sum
            run_len = rles[:, 3].astype(np.int64)
            n_voxels = int(run_len.sum())

            coords = np.empty((n_voxels, 3), dtype='uint32')

            # y and z are constant within a run
            coords[:, 1] = np.repeat(rles[:, 1], run_len)
            coords[:, 2] = np.repeat(rles[:, 2], run_len)

            # x runs from `x` to `x + run_len`: add the index within each run
            # (a global arange minus the repeated offset at which the run
            # starts) to the repeated run start. Both terms stay non-negative,
            # so this is safe in uint32.
            offsets = (np.cumsum(run_len) - run_len).astype('uint32')
            coords[:, 0] = np.arange(n_voxels, dtype='uint32')
            coords[:, 0] -= np.repeat(offsets, run_len)
            coords[:, 0] += np.repeat(rles[:, 0].astype('uint32'), run_len)
        else:
            coords = rles
        return header, coords
    elif format == 'blocks':
        raise ValueError('Format "blocks" not yet implemented.')
    else:
        raise ValueError('Unknown format "{}"'.format(format))


def read_ngmesh(f, fix=True):
    """Read neuroglancer mesh (single-resolution legacy format).

    See here for specs:
    https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/meshes.md#legacy-single-resolution-mesh-format

    Parameters
    ----------
    f :         File-like object
                An open binary file object.
    fix :       bool
                If True, will try to fix some potential issues with the mesh
                like duplicate vertices, etc.

    Returns
    -------
    trimesh.Trimesh

    """
    if isinstance(f, bytes):
        f = BytesIO(f)

    num_vertices = np.frombuffer(f.read(4),
                                 np.uint32)[0]
    vertices = np.frombuffer(f.read(int(3*4*num_vertices)),
                             np.float32).reshape(-1, 3)
    faces = np.frombuffer(f.read(),
                          np.uint32).reshape(-1, 3)

    m = tm.Trimesh(vertices=vertices, faces=faces)

    if fix:
        # Cater for older and newer versions of trimesh
        if hasattr(m, 'remove_degenerate_faces'):
            m.remove_degenerate_faces()
        else:
            m.update_faces(m.nondegenerate_faces())
        if hasattr(m, 'remove_duplicate_faces'):
            m.remove_duplicate_faces()
        else:
            m.update_faces(m.unique_faces())
        m.remove_infinite_values()
        m.remove_unreferenced_vertices()
        m.merge_vertices()

    return m

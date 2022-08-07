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

        if voxels:
            coords = []
            for i in range(header['n_spans']):
                offset = 12 + (i * 16)
                x, y, z, run_len = struct.unpack('iiii', b[offset: offset + 16])

                this_run = np.array([[x, y, z]] * run_len, dtype='uint32')
                this_run[:, 0] = np.arange(x, x + run_len)

                coords.append(this_run)

            if len(coords):
                coords = np.concatenate(coords)
            else:
                coords = np.zeros((0, 3))
        else:
            coords = np.zeros((header['n_spans'], 4), dtype='uint32')
            for i in range(header['n_spans']):
                offset = 12 + (i * 16)
                coords[i, :] = struct.unpack('iiii', b[offset: offset + 16])
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
        m.remove_degenerate_faces()
        m.remove_duplicate_faces()
        m.remove_infinite_values()
        m.remove_unreferenced_vertices()
        m.merge_vertices()

    return m

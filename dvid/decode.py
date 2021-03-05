# This code is part of dvid-tools (https://github.com/flyconnectome/dvid_tools)
# and is released under GNU GPL3

import struct
import numpy as np


def decode_sparsevol(b, format='rles'):
    """Decode sparsevol binary mesh format.

    Parameters
    ----------
    b :         bytes
                Data to decode.
    format :    "blocks" | "rles"
                Binary format in which the file is encoded. Only "rles" is
                supported at the moment.

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
        coords = []
        for i in range(header['n_spans']):
            offset = 12 + (i * 16)
            x, y, z, run_len = struct.unpack('iiii', b[offset: offset + 16])

            this_run = np.repeat([[x, y, z]], run_len, axis=0)
            this_run[:, 0] = np.arange(x, x + run_len)

            coords.append(this_run)
        coords = np.concatenate(coords)
        return header, coords
    elif format == 'blocks':
        raise ValueError('Format "blocks" not yet implemented.')
    else:
        raise ValueError('Unknown format "{}"'.format(format))

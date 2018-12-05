# This code is part of dvid-tools (https://github.com/flyconnectome/dvid_tools)
# and is released under GNU GPL3

import numpy as np

from skimage.measure import marching_cubes_lewiner

def mesh_from_voxels(voxels, v_size, step_size=1):
    """ Turns voxels into vertices and faces using marching cubes.

    Parameters
    ----------
    voxels :    np.array
                Voxel coordindates. Array of either (N, 3) or (N, 4).
                (N, 3) will be interpreted as single x, y, z voxel coordinates.
                (N, 4) will be interpreted as column of x, y, z_start, z_end
                coordindates.
    v_size :    np.array
                (3, ) array with x, y, z voxel size.
    step_size : int, optional
                Step size for marching cube algorithm.
                Higher values = faster but coarser.

    """
    # Offset block coordinates by +1.
    # Necessary for marching cube producing watertight meshes.
    # Will compensate for this further down
    voxels += 1
    
    # Generate empty array    
    mat = np.zeros((voxels.max(axis=0) + 2)[[0, 1, -1]], dtype=np.float32)
    
    #Populate matrix
    if voxels.shape[1] == 4:
        for col in voxels:
            mat[col[0], col[1], col[2]:col[3]] = 1
    elif voxels.shape[1] == 3:
        mat[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = 1
    else:
        raise ValueError('Unexpected shape')
    
    # Use marching cubes to create surface model
    verts, faces, normals, values = marching_cubes_lewiner(mat,
                                                           level=.5,
                                                           step_size=step_size,
                                                           allow_degenerate=False,
                                                           gradient_direction='ascent',                                                           
                                                           spacing=v_size)

    # Compensate for earlier block offset
    verts -= np.array(v_size)

    # For unknown reasons, the x/y/z order is inverted during marching cubes
    verts = verts[:, [2, 1, 0]]

    return verts, faces
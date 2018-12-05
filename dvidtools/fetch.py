# This code is part of dvid-tools (https://github.com/flyconnectome/dvid_tools)
# and is released under GNU GPL3

from . import decode
from . import mesh

import requests

import numpy as np
import pandas as pd


def set_param(server=None, node=None, user=None):
    for p, n in zip([server, node, user], ['server', 'node', 'user']):
        if not isinstance(p, type(None)):
            globals()[n] = p

def eval_param(server=None, node=None, user=None):
    """ Helper to read globally defined settings."""    
    parsed = {}
    for p, n in zip([server, node, user], ['server', 'node', 'user']):
        if isinstance(p, type(None)):
            parsed[n] = globals().get(n, None)
        else:
            parsed[n] = p
    return [parsed[n] for n in ['server', 'node', 'user']]


def get_skeleton(bodyid, save_to=None, server=None, node=None):
    """ Download skeleton as SWC file.

    Parameters
    ----------
    bodyid :    int | str
                ID of body for which to download skeleton.
    save_to :   str | None, optional
                If provided, will save SWC to file.
    server :    str, optional
                If not provided, will try reading from global.
    node :      str, optional
                If not provided, will try reading from global.

    Returns
    -------
    SWC :       str
                Only if ``save_to=None``.
    """

    server, node, user = eval_param(server, node)
    
    r = requests.get('{}/api/node/{}/segmentation_skeletons/key/{}_swc'.format(server, node, bodyid))
    swc = r.text
                     
    if save_to:
        with open(save_to, 'w') as f:
            f.write(swc)
    else:
        return swc


def get_user_bookmarks(server=None, node=None, user=None):
    """ Get user bookmarks.

    Parameters
    ----------
    server :    str, optional
                If not provided, will try reading from global.
    node :      str, optional
                If not provided, will try reading from global.
    user :      str, optional
                If not provided, will try reading from global.                

    Returns
    -------
    bookmarks : pandas.DataFrame
                
    """
    server, node, user = eval_param(server, node, user)
    
    r = requests.get('{}/api/node/{}/bookmark_annotations/tag/user:{}'.format(server, node, user))

    return pd.DataFrame.from_records(r.json())    


def get_annotation(bodyid, server=None, node=None, verbose=True):
    """ Fetch annotations for given body.

    Parameters
    ----------
    bodyid :    int | str
                ID of body for which to download skeleton.
    server :    str, optional
                If not provided, will try reading from global.
    node :      str, optional
                If not provided, will try reading from global.
    verbose :   bool, optional
                If True, will print error if no annotation for body found.

    Returns
    -------
    annotations :   dict
    """
    server, node, user = eval_param(server, node)
    
    r = requests.get('{}/api/node/{}/segmentation_annotations/key/{}'.format(server, node, bodyid))
    
    try:    
        return r.json()
    except:
        if verbose:
            print(r.text)
        return {}


def get_body_id(pos, server=None, node=None):
    """ Get body ID at given position.

    Parameters
    ----------
    pos :       iterable
                [x, y, z] position to query.
    server :    str, optional
                If not provided, will try reading from global.
    node :      str, optional
                If not provided, will try reading from global.    

    Returns
    -------
    body_id :   str
    """
    server, node, user = eval_param(server, node)
    
    r = requests.get('{}/api/node/{}/segmentation/label/{}_{}_{}'.format(server, node, pos[0], pos[1], pos[2]))
    
    return r.json()['Label']


def get_todo_in_area(offset, size, server=None, node=None):
    """ Get TODO tags in given 3D area.

    Parameters
    ----------
    offset :    iterable
                [x, y, z] position of top left corner of area.
    size :      iterable
                [x, y, z] dimensions of area.
    server :    str, optional
                If not provided, will try reading from global.
    node :      str, optional
                If not provided, will try reading from global.    

    Returns
    -------
    todo tags : pandas.DataFrame
    """
    server, node, user = eval_param(server, node)
    
    r = requests.get('{}/api/node/{}/segmentation_todo/elements/{}_{}_{}/{}_{}_{}'.format(server,
                                                                                          node,
                                                                                          size[0],
                                                                                          size[1],
                                                                                          size[2],                                                                                        
                                                                                          offset[0],
                                                                                          offset[1],
                                                                                          offset[2]))
    
    return pd.DataFrame.from_records(r.json())


def get_roi(roi, voxel_size=(32, 32, 32), server=None, node=None,
            step_size=2):
    """ Get faces and vertices of ROI.

    Uses marching cube algorithm to extract surface model of block ROI.

    Parameters
    ----------
    roi :           str
                    Name of ROI.
    voxel_size :    iterable
                    (3, ) iterable of voxel sizes.
    server :        str, optional
                    If not provided, will try reading from global.
    node :          str, optional
                    If not provided, will try reading from global.
    step_size :     int, optional
                    Step size for marching cube algorithm.
                    Smaller values = higher resolution but slower.

    Returns
    -------
    vertices :      numpy.ndarray
                    Coordinates are in nm.
    faces :         numpy.ndarray
    """
    
    server, node, user = eval_param(server, node)
    
    r = requests.get('{}/api/node/{}/{}/roi'.format(server, node, roi))
    
    # The data returned are block coordinates: [x, y, z_start, z_end]
    blocks = np.array(r.json())
    
    verts, faces = mesh.mesh_from_voxels(blocks, v_size=voxel_size,
                                         step_size=step_size)
    
    return verts, faces


def get_neuron(bodyid, scale='coarse', step_size=2, save_to=None, server=None, node=None):
    """ Get neuron as mesh.

    Parameters
    ----------
    bodyid :    int | str
                ID of body for which to download skeleton.
    scale :     int | "coarse", optional
                Resolution of sparse volume starting with 0 where each level
                beyond 0 has 1/2 resolution of previous level. "coarse" will
                return the volume in block coordinates.
    step_size : int, optional
                Step size for marching cube algorithm.
                Higher values = faster but coarser.
    save_to :   str | None, optional
                If provided, will not convert to verts and faces but instead
                save as response from server as binary file.
    server :    str, optional
                If not provided, will try reading from global.
    node :      str, optional
                If not provided, will try reading from global.

    Returns
    -------
    verts :     np.array
                Vertex coordinates in nm.
    faces :     np.array                
    """

    server, node, user = eval_param(server, node)

    # Get voxel sizes based on scale
    info = get_segmentation_info(server, node)['Extended']

    vsize = {'coarse' : info['BlockSize']}
    vsize.update({i: np.array(info['VoxelSize']) * 2**i for i in range(info['MaxDownresLevel'])})

    if isinstance(scale, int) and scale > info['MaxDownresLevel']:
        raise ValueError('Scale greater than MaxDownresLevel')
    
    if scale == 'coarse':
        r = requests.get('{}/api/node/{}/segmentation/sparsevol-coarse/{}'.format(server, node, bodyid))
    else:
        r = requests.get('{}/api/node/{}/segmentation/sparsevol/{}?scale={}'.format(server, node, bodyid, scale))

    b = r.content
                     
    if save_to:
        with open(save_to, 'wb') as f:
            f.write(b)
        return
    
    # Decode binary format
    header, coords = decode.decode_sparsevol(b, format='rles')

    return coords

    verts, faces = mesh.mesh_from_voxels(coords,
                                         v_size=vsize[scale],
                                         step_size=step_size)

    return verts, faces


def get_segmentation_info(server=None, node=None):
    """ Returns segmentation info as dictionary.
    """

    server, node, user = eval_param(server, node)

    r = requests.get('{}/api/node/{}/segmentation/info'.format(server, node))

    return r.json()

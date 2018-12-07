# This code is part of dvid-tools (https://github.com/flyconnectome/dvid_tools)
# and is released under GNU GPL3

from . import decode
from . import mesh
from . import utils

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

    #r.raise_for_status()

    if 'not found' in r.text:
        print(r.text)
        return None
                 
    if save_to:
        with open(save_to, 'w') as f:
            f.write(r.text)
    else:
        return r.text


def get_user_bookmarks(server=None, node=None, user=None,
                       return_dataframe=True):
    """ Get user bookmarks.

    Parameters
    ----------
    server :            str, optional
                        If not provided, will try reading from global.
    node :              str, optional
                        If not provided, will try reading from global.
    user :              str, optional
                        If not provided, will try reading from global.                
    return_dataframe :  bool, optional
                        If True, will return pandas.DataFrame. If False,
                        returns original json.

    Returns
    -------
    bookmarks : pandas.DataFrame
                
    """
    server, node, user = eval_param(server, node, user)
    
    r = requests.get('{}/api/node/{}/bookmark_annotations/tag/user:{}'.format(server, node, user))

    if return_dataframe:
        return pd.DataFrame.from_records(r.json())
    else:
        return r.json()


def add_bookmarks(data, verify=True, server=None, node=None):
    """ Add or edit user bookmarks.

    Please note that you will have to restart neutu to see the changes to
    your user bookmarks.

    Parameters
    ----------
    data :      list of dicts
                Must be list of dicts. See example::

                    [{'Pos': [21344, 21048, 22824],
                      'Kind': 'Note',
                      'Tags': ['user:schlegelp'],
                      'Prop': {'body ID': '1671952694',
                               'comment': 'mALT',
                               'custom': '1',
                               'status': '',
                               'time': '',
                               'type': 'Other',
                               'user': 'schlegelp'}},
                     ... ]

    verify :    bool, optional
                If True, will sanity check ``data`` against above example.
                Do not skip unless you know exactly what you're doing!
    server :    str, optional
                If not provided, will try reading from global.
    node :      str, optional
                If not provided, will try reading from global.               

    Returns
    -------
    Nothing
                
    """
    server, node, user = eval_param(server, node)

    # Sanity check data
    if not isinstance(data, list):
        raise TypeError('Data must be list of dicts. '
                        'See help(dvidtools.add_bookmarks)')    

    if verify:
        required = {'Pos': list, 'Kind': str, 'Tags': [str],
                'Prop': {'body ID': str, 'comment': str, 'custom': str,
                         'status': str, 'time': str, 'type': str,
                         'user': str}}

        utils.verify_payload(data, required=required, required_only=True)
    
    r = requests.post('{}/api/node/{}/bookmark_annotations/elements'.format(server, node),
                      json=data)

    r.raise_for_status()

    return


def get_annotation(bodyid, server=None, node=None, verbose=True):
    """ Fetch annotations for given body.

    Parameters
    ----------
    bodyid :    int | str
                ID of body for which to get annotations..
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


def edit_annotation(bodyid, annotation, server=None, node=None):
    """ Edit annotations for given body.

    Parameters
    ----------
    bodyid :        int | str
                    ID of body for which to edit annotations.
    annotation :    dict
                    Dictionary of new annotations. Do not omit any of these
                    fields as this will delete it from the entry on the
                    server::

                        {
                         "status": str,
                         "comment": str,
                         "body ID": int,
                         "name": str,
                         "class": str,
                         "user": str,
                         "naming user": str
                        }

    server :        str, optional
                    If not provided, will try reading from global.
    node :          str, optional
                    If not provided, will try reading from global.
    verbose :       bool, optional
                    If True, will print error if no annotation for body found.

    Returns
    -------
    None

    Examples
    --------
    >>> # Get annotations for given body
    >>> an = dvidtools.get_annotation('1700937093')
    >>> # Edit field
    >>> an['name'] = 'New Name'
    >>> # Update annotation 
    >>> dvidtools.edit_annotation('1700937093', an)
    """
    server, node, user = eval_param(server, node)

    # Sanity check
    fields = {"status": str, "comment": str, "body ID": int, "name": str,
              "user": str, "naming user": str} #"class": str

    utils.verify_payload([annotation], fields, required_only=True)
    
    r = requests.post('{}/api/node/{}/segmentation_annotations/key/{}'.format(server, node, bodyid),
                      json=annotation)
    # Check if it worked
    r.raise_for_status()
    
    return None


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
                ID of body for which to download mesh.
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

    Parameters
    ----------
    server :    str, optional
                If not provided, will try reading from global.
    node :      str, optional
                If not provided, will try reading from global.
    """

    server, node, user = eval_param(server, node)

    r = requests.get('{}/api/node/{}/segmentation/info'.format(server, node))

    return r.json()


def get_n_synapses(bodyid, server=None, node=None):
    """ Returns number of pre- and postsynapses associated with given
    body.

    Parameters
    ----------
    bodyid :    int | str
                ID of body for which to get number of synapses.
    server :    str, optional
                If not provided, will try reading from global.
    node :      str, optional
                If not provided, will try reading from global.

    Returns
    -------
    dict 
                ``{'PreSyn': int, 'PostSyn': int}
    """

    server, node, user = eval_param(server, node)

    r = requests.get('{}/api/node/{}/synapses_labelsz/count/{}/PreSyn'.format(server, node, bodyid))
    r.raise_for_status()
    pre = r.json()

    r = requests.get('{}/api/node/{}/synapses_labelsz/count/{}/PostSyn'.format(server, node, bodyid))
    r.raise_for_status()
    post = r.json()

    return {'pre': pre.get('PreSyn', None), 'post': post.get('PostSyn', None)}


def get_synapses(bodyid, server=None, node=None):
    """ Returns table of pre- and postsynapses associated with given body.

    Parameters
    ----------
    bodyid :    int | str
                ID of body for which to get synapses.
    server :    str, optional
                If not provided, will try reading from global.
    node :      str, optional
                If not provided, will try reading from global.

    Returns
    -------
    pandas.DataFrame                 
    """

    if isinstance(bodyid, (list, np.ndarray)):
        tables = [get_synapses(b, server, node) for b in bodyid]
        for b, tbl in zip(bodyid, tables):
            tbl['bodyid'] = b
        return pd.concat(tables, axis=0)

    server, node, user = eval_param(server, node)

    r = requests.get('{}/api/node/{}/synapses/label/{}?relationships=false'.format(server, node, bodyid))

    return pd.DataFrame.from_records(r.json())


def get_connectivity(bodyid, pos_filter=None, server=None, node=None):
    """ Returns connectivity table for given body.

    Parameters
    ----------
    bodyid :    int | str
                ID of body for which to get connectivity.
    filter :    function, optional
                Function to filter synapses by position. Must accept numpy
                array (N, 3) and return array of [True, False, ...]
    server :    str, optional
                If not provided, will try reading from global.
    node :      str, optional
                If not provided, will try reading from global.

    Returns
    -------
    pandas.DataFrame    

    """

    if isinstance(bodyid, (list, np.ndarray)):
        cn = [get_connectivity(b, pos_filter=pos_filter,
                               server=server, node=node) for b in bodyid]
        cn = functools.reduce(lambda left,right: pd.merge(left, right,
                                                          on=['bodyid',
                                                              'relation'],
                                                          how='outer'),
                              cn)
        cn = cn.fillna(0)
        cn.columns = np.append(cn.columns[:2], bodyid)
        cn['total'] = cn[bodyid].sum(axis=1)
        return cn.sort_values(['relation', 'total'], ascending=False).reset_index(drop=True)

    server, node, user = eval_param(server, node)

    # Get synapses
    r = requests.get('{}/api/node/{}/synapses/label/{}?relationships=true'.format(server, node, bodyid))
    syn = r.json()

    if pos_filter:
        # Get filter
        filtered = pos_filter(np.array([s['Pos'] for s in syn]))

        if not any(filtered):
            raise ValueError('No synapses left after filtering.')

        # Filter synapses
        syn = np.array(syn)[filtered]

    # Compile connector table by counting # of synapses between neurons
    connections = {'PreSynTo': {}, 'PostSynTo': {}}

    # Collect positions and query the body IDs
    pos = [cn['To'] for s in syn for cn in s['Rels']]

    bodies = requests.request('GET',
                              url="{}/api/node/{}/segmentation/labels".format(server, node),
                              json=pos).json()

    i = 0
    for s in syn:
        # Connections point to positions -> we have to map this to body IDs
        for k, cn in enumerate(s['Rels']):
            b = bodies[i+k]
            connections[cn['Rel']][b] = connections[cn['Rel']].get(b, 0) + 1
        i += k

    pre = pd.DataFrame.from_dict(connections['PreSynTo'], orient='index')
    pre.columns = ['n_synapses']
    pre['relation'] = 'downstream'
    pre.index.name = 'bodyid'

    post = pd.DataFrame.from_dict(connections['PostSynTo'], orient='index')
    post.columns = ['n_synapses']
    post['relation'] = 'upstream'
    post.index.name = 'bodyid'

    cn_table = pd.concat([pre.reset_index(), post.reset_index()], axis=0)
    cn_table.sort_values(['relation', 'n_synapses'], inplace=True, ascending=False)
    cn_table.reset_index(drop=True, inplace=True)

    return cn_table

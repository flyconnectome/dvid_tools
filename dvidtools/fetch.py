# This code is part of dvid-tools (https://github.com/flyconnectome/dvid_tools)
# and is released under GNU GPL3

from . import decode
from . import mesh
from . import utils

import inspect
import os
import requests

import numpy as np
import pandas as pd

from io import StringIO
from tqdm import tqdm


def set_param(server=None, node=None, user=None):
    """ Set default server, node and/or user."""
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


def get_skeleton(bodyid, save_to=None, xform=None, server=None, node=None,
                 verbose=True):
    """ Download skeleton as SWC file.

    Parameters
    ----------
    bodyid :    int | str
                ID(s) of body for which to download skeleton.
    save_to :   str | None, optional
                If provided, will save SWC to file. If str must be file or
                path.
    xform :     function, optional
                If provided will run this function to transform coordinates
                before saving/returning the SWC file. Function must accept
                ``(N, 3)`` numpy array.
    server :    str, optional
                If not provided, will try reading from global.
    node :      str, optional
                If not provided, will try reading from global.

    Returns
    -------
    SWC :       str
                Only if ``save_to=None``.

    Examples
    --------
    Easy: grab a neuron and save it to a file

    >>> dt.get_skeleton(485775679, save_to='~/Downloads/')

    Grab a neuron and transform to FAFB space before saving to
    file. This requires `navis <https://navis.readthedocs.io>`_
    and ``rpy2`` to be installed.

    >>> from navis.interfaces
    >>> from rpy2.robjects.packages import importr
    >>> importr('nat.jrcfibf')
    >>> dvidtools.get_skeleton(485775679,
    ...                        save_to='~/Downloads/',
    ...                        xform=lambda x: r.xform_brain(x,
    ...                                                      source='JRCFIB2018Fraw',
    ...                                                      target='FAFB14')
    ...                       )


    """

    if isinstance(bodyid, (list, np.ndarray)):
        if save_to and not os.path.isdir(save_to):
            raise ValueError('"save_to" must be path when loading multiple'
                             'multiple bodies')
        resp = {x: get_skeleton(x,
                                save_to=save_to,
                                server=server,
                                node=node) for x in tqdm(bodyid,
                                                         desc='Loading')}
        if not save_to:
            return resp
        else:
            return

    bodyid = utils.parse_bid(bodyid)

    server, node, user = eval_param(server, node)

    r = requests.get('{}/api/node/{}/segmentation_skeletons/key/{}_swc'.format(server, node, bodyid))

    #r.raise_for_status()

    if 'not found' in r.text:
        if verbose:
            print(r.text)
        return None

    if callable(xform):
        # Parse SWC string
        df, header = utils.parse_swc_str(r.text)

        # Add xform function to header for documentation
        header += '#x/y/z coordinates transformed by dvidtools using this:\n'
        header += '\n'.join(['#' + l for l in inspect.getsource(xform).split('\n')])

        # Transform coordinates
        df.iloc[:, 2:5] = xform(df.iloc[:, 2:5].values)

        # Turn DataFrame back into string
        s = StringIO()
        df.to_csv(s, sep=' ', header=False)

        # Replace text
        swc = header + s.getvalue()
    elif not isinstance(xform, type(None)):
        raise TypeError('"xform" must be a function, not "{}"'.format(type(x)))
    else:
        swc = r.text

    if save_to:
        if os.path.isdir(save_to):
            save_to = os.path.join(save_to, '{}.swc'.format(bodyid))
        with open(save_to, 'w') as f:
            f.write(swc)
    else:
        return swc


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
    bookmarks : pandas.DataFrame or json

    """
    server, node, user = eval_param(server, node, user)

    r = requests.get('{}/api/node/{}/bookmark_annotations/tag/user:{}'.format(server, node, user))

    if return_dataframe:
        data = r.json()
        for d in data:
            d.update(d.pop('Prop'))
        return pd.DataFrame.from_records(data)
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
                    Dictionary of new annotations. Possible fields are::

                        {
                         "status": str,
                         "comment": str,
                         "body ID": int,
                         "name": str,
                         "class": str,
                         "user": str,
                         "naming user": str
                        }

                    Fields other than the above will be ignored!

    server :        str, optional
                    If not provided, will try reading from global.
    node :          str, optional
                    If not provided, will try reading from global.

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

    bodyid = utils.parse_bid(bodyid)

    # Get existing annotations
    old_an = get_annotation(bodyid, server=server, node=node)

    # Compile new annotations
    new_an = {k: annotation.get(k, v) for k, v in old_an.items()}

    r = requests.post('{}/api/node/{}/segmentation_annotations/key/{}'.format(server, node, bodyid),
                      json=new_an)

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


def get_multiple_bodyids(pos, server=None, node=None):
    """ Get body IDs at given positions.

    Parameters
    ----------
    pos :       iterable
                [[x1, y1, z1], [x2, y2, z2], ..] positions to query. Must be
                integers!
    server :    str, optional
                If not provided, will try reading from global.
    node :      str, optional
                If not provided, will try reading from global.

    Returns
    -------
    body_ids :  list
    """
    server, node, user = eval_param(server, node)

    if isinstance(pos, np.ndarray):
        pos = pos.tolist()

    r = requests.request('GET',
                         url="{}/api/node/{}/segmentation/labels".format(server, node),
                         json=pos)

    r.raise_for_status()

    return r.json()


def get_body_position(bodyid, server=None, node=None):
    """ Get a single position for given body ID.

    This will (like neutu) use the skeleton. If body has no skeleton, will
    use mesh as fallback.

    Parameters
    ----------
    bodyid :    body ID
                Body for which to find a position.
    server :    str, optional
                If not provided, will try reading from global.
    node :      str, optional
                If not provided, will try reading from global.

    Returns
    -------
    (x, y, z)
    """

    bodyid = utils.parse_bid(bodyid)

    s = get_skeleton(bodyid, server=server, node=node, verbose=False)

    if s:
        s, header = utils.parse_swc_str(s)
        return s.loc[0, ['x', 'y', 'z']].values
    else:
        # First get voxels of the coarse neuron
        voxels = get_neuron(bodyid, scale='coarse', ret_type='INDEX',
                            server=server, node=node)

        # Erode surface voxels to make sure we get a central position
        while True:
            eroded = mesh.remove_surface_voxels(voxels)

            # Stop before no more voxels left
            if eroded.size == 0:
                break

            voxels = eroded

        # Now query the more precise mesh for this coarse voxel
        # Pick a random voxel
        v = voxels[0] * 64
        # Generate a bounding bbox
        bbox = np.vstack([v,v]).T
        bbox[:, 1] += 63

        voxels = get_neuron(bodyid, scale=0, ret_type='INDEX',
                            bbox=bbox.ravel(),
                            server=server, node=node)

        # Erode surface voxels again to make sure we get a central position
        while True:
            eroded = mesh.remove_surface_voxels(voxels)

            # Stop before no more voxels left
            if eroded.size == 0:
                break

            voxels = eroded

        return voxels[0]


def get_body_profile(bodyid, server=None, node=None):
    """ Get body profile (n voxels, n blocks, bounding box)

    Parameters
    ----------
    bodid :     str | int
                Body id.
    server :    str, optional
                If not provided, will try reading from global.
    node :      str, optional
                If not provided, will try reading from global.

    Returns
    -------
    profile :   dict
    """
    server, node, user = eval_param(server, node)

    bodyid = utils.parse_bid(bodyid)

    r = requests.request('GET',
                              url="{}/api/node/{}/segmentation/sparsevol-size/{}".format(server, node, bodyid))

    r.raise_for_status()

    return r.json()


def get_assignment_status(pos, window=None, server=None, node=None):
    """ Returns assignment status at given position.

    Checking/unchecking assigments leaves invisible "bookmarks" at the given
    position. These can be queried using this endpoint.

    Parameters
    ----------
    pos :       tuple
                X/Y/Z Coordinates to query.
    window :    array-like | None, optional
                If provided, will return assigments in bounding box with
                ``pos`` in the center and ``window`` as size in x/y/z.
    bodyid :    int | list, optional
                If provided, will only return assignments that are within the
                given body ID(s). Only relevant if ``window!=None``.
    server :    str, optional
                If not provided, will try reading from global.
    node :      str, optional
                If not provided, will try reading from global.

    Returns
    -------
    dict
                E.g. ``{'checked': True}`` if assignment(s) were found at
                given position/in given bounding box.
    None
                If no assigments found.
    list
                If ``window!=None`` will return a list of of dicts.

    """

    server, node, user = eval_param(server, node)

    if isinstance(window, (list, np.ndarray, tuple)):
        pos = pos if isinstance(pos, np.ndarray) else np.array(pos)
        pos = pos.astype(int)
        window = window if isinstance(window, np.ndarray) else np.array(window)

        r = requests.get('{}/api/node/{}/bookmarks/keyrange/'
                         '{}_{}_{}/{}_{}_{}'.format(server,
                                                    node,
                                                    int(pos[0]-window[0]/2),
                                                    int(pos[1]-window[1]/2),
                                                    int(pos[2]-window[2]/2),
                                                    int(pos[0]+window[0]/2),
                                                    int(pos[1]+window[1]/2),
                                                    int(pos[2]+window[2]/2),))
        r.raise_for_status()

        # Above query returns coordinates that are in lexicographically
        # between key1 and key2 -> we have to filter for those inside the
        # bounding box ourselves
        coords = np.array([c.split('_') for c in r.json()]).astype(int)

        # If provided, make sure all coordinates in window are from given
        # body ID(s)
        if not isinstance(bodyid, type(None)):
            if not isinstance(bodyid, (list, np.ndarray)):
                bodyid = [bodyid]

            bids = np.array(get_multiple_bodyids(coords,
                                                 server=server,
                                                 node=node
                                                 ))

            coords = coords[np.in1d(bids, bodyid)]

        if coords.size == 0:
            return []

        coords = coords[(coords > (pos - window / 2)).all(axis=1)]
        coords = coords[(coords < (pos + window / 2)).all(axis=1)]

        return [get_assignment_status(c,
                                      window=None,
                                      bodyid=bodyid,
                                      server=server,
                                      node=node) for c in coords]

    r = requests.get('{}/api/node/{}/bookmarks/key/{}_{}_{}'.format(server,
                                                                    node,
                                                                    int(pos[0]),
                                                                    int(pos[1]),
                                                                    int(pos[2])))

    # Will raise if key not found -> so just don't
    # r.raise_for_status()

    return r.json() if r.text and 'not found' not in r.text else None


def get_labels_in_area(offset, size, server=None, node=None):
    """ Get labels (todo, to split, etc.) in given bounding box.

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

    r = requests.get('{}/api/node/{}/segmentation_todo/elements/'
                     '{}_{}_{}/{}_{}_{}'.format(server,
                                                node,
                                                int(size[0]),
                                                int(size[1]),
                                                int(size[2]),
                                                int(offset[0]),
                                                int(offset[1]),
                                                int(offset[2])))

    r.raise_for_status()

    j = r.json()

    if j:
        return pd.DataFrame.from_records(r.json())
    else:
        return None


def get_available_rois(server=None, node=None, step_size=2):
    """ Get a list of all available ROIs in given node.

    Parameters
    ----------
    server :        str, optional
                    If not provided, will try reading from global.
    node :          str, optional
                    If not provided, will try reading from global.

    Returns
    -------
    list
    """

    server, node, user = eval_param(server, node)

    r = requests.get('{}/api/node/{}/rois/keys'.format(server, node))

    r.raise_for_status()

    return r.json()


def get_roi(roi, voxel_size=(32, 32, 32), server=None, node=None,
            step_size=2, return_raw=False):
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
    return_raw :    bool, optional
                    If True, will return raw block data instead of faces and
                    verts. Might not exists for all ROIs!

    Returns
    -------
    vertices :      numpy.ndarray
                    Coordinates are in nm.
    faces :         numpy.ndarray
    """

    server, node, user = eval_param(server, node)

    r = requests.get('{}/api/node/{}/{}/roi'.format(server, node, roi))

    r.raise_for_status()

    # The data returned are block coordinates: [x, y, z_start, z_end]
    blocks = r.json()

    if return_raw:
        return blocks

    verts, faces = mesh.mesh_from_voxels(np.array(blocks), v_size=voxel_size,
                                         step_size=step_size)

    return verts, faces


def get_roi2(roi, save_to=None, server=None, node=None):
    """ Get `.obj` for ROI.


    Parameters
    ----------
    roi :           str
                    Name of ROI.
    save_to :       str | None, optional
                    If provided, will not return string but instead
                    save as file.
    server :        str, optional
                    If not provided, will try reading from global.
    node :          str, optional
                    If not provided, will try reading from global.

    Returns
    -------
    str
    """

    server, node, user = eval_param(server, node)

    # Get the key for this roi
    r = requests.get('{}/api/node/{}/rois/key/{}'.format(server, node, roi))
    r.raise_for_status()
    key = r.json()['->']['key']

    # Get the obj string
    r = requests.get('{}/api/node/{}/roi_data/key/{}'.format(server, node, key))
    r.raise_for_status()

    if save_to:
        with open(save_to, 'w') as f:
            f.write(r.text)
        return

    # The data returned is in .obj format
    return r.text


def get_neuron(bodyid, scale='coarse', step_size=2, save_to=None,
               ret_type='MESH', bbox=None, server=None, node=None):
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
    ret_type :  "MESH" | "COORDS" | "INDEX", optional
                If "MESH" will return vertices and faces. If "COORDS" will
                return voxel coordinates. "INDEX" returns voxel indices.
    bbox :      list | None, optional
                Bounding box to which to restrict the query to.
                Format: ``[x_min, x_max, y_min, y_max, z_min, z_max]``.
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

    if ret_type.lower() not in ['mesh', 'coords', 'index']:
        raise ValueError('"ret_type" must be "MESH", "COORDS" or "INDEX"')

    server, node, user = eval_param(server, node)

    bodyid = utils.parse_bid(bodyid)

    # Get voxel sizes based on scale
    info = get_segmentation_info(server, node)['Extended']

    vsize = {'coarse' : info['BlockSize']}
    vsize.update({i: np.array(info['VoxelSize']) * 2**i for i in range(info['MaxDownresLevel'])})

    if isinstance(scale, int) and scale > info['MaxDownresLevel']:
        raise ValueError('Scale greater than MaxDownresLevel')


    if not isinstance(bbox, type(None)):
        url = '{}/api/node/{}/segmentation/sparsevol/{}'.format(server, node, bodyid)
        url += '?minx={}&maxx={}&miny={}&maxy={}&minz={}&maxz={}'.format(int(bbox[0]),
                                                                         int(bbox[1]),
                                                                         int(bbox[2]),
                                                                         int(bbox[3]),
                                                                         int(bbox[4]),
                                                                         int(bbox[5]))
    elif scale == 'coarse':
        url = '{}/api/node/{}/segmentation/sparsevol-coarse/{}'.format(server, node, bodyid)
    else:
        url = '{}/api/node/{}/segmentation/sparsevol/{}?scale={}'.format(server, node, bodyid, scale)

    r = requests.get(url)
    r.raise_for_status()

    b = r.content

    if save_to:
        with open(save_to, 'wb') as f:
            f.write(b)
        return

    # Decode binary format
    header, voxels = decode.decode_sparsevol(b, format='rles')

    if ret_type.lower() == 'index':
        return voxels
    elif ret_type.lower == 'coords':
        return voxels * vsize[scale]

    verts, faces = mesh.mesh_from_voxels(voxels,
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
                ``{'PreSyn': int, 'PostSyn': int}``
    """

    server, node, user = eval_param(server, node)

    bodyid = utils.parse_bid(bodyid)

    if isinstance(bodyid, (list, np.ndarray)):
        syn = {b: get_n_synapses(b, server, node) for b in bodyid}
        return pd.DataFrame.from_records(syn).T

    r = requests.get('{}/api/node/{}/synapses_labelsz/count/{}/PreSyn'.format(server, node, bodyid))
    r.raise_for_status()
    pre = r.json()

    r = requests.get('{}/api/node/{}/synapses_labelsz/count/{}/PostSyn'.format(server, node, bodyid))
    r.raise_for_status()
    post = r.json()

    return {'pre': pre.get('PreSyn', None), 'post': post.get('PostSyn', None)}


def get_synapses(bodyid, pos_filter=None, with_details=False, server=None, node=None):
    """ Returns table of pre- and postsynapses associated with given body.

    Parameters
    ----------
    bodyid :        int | str
                    ID of body for which to get synapses.
    pos_filter :    function, optional
                    Function to filter synapses by position. Must accept
                    numpy array (N, 3) and return array of [True, False, ...]
    with_details :  bool, optional
                    If True, will include more detailed information about
                    connector links.
    server :        str, optional
                    If not provided, will try reading from global.
    node :          str, optional
                    If not provided, will try reading from global.

    Returns
    -------
    pandas.DataFrame

    Examples
    --------
    Get synapses only in the LH (requires navis)
    >>> import navis
    >>> lh = navis.Volume(*dvidtools.get_roi('LH'))
    >>> lh_syn = dvidtools.get_synapses(329566174,
    ...                                 pos_filter=lambda x: navis.in_volume(x, lh))
    """

    if isinstance(bodyid, (list, np.ndarray)):
        tables = [get_synapses(b, pos_filter, server, node) for b in tqdm(bodyid,
                                                              desc='Fetching')]
        for b, tbl in zip(bodyid, tables):
            tbl['bodyid'] = b
        return pd.concat(tables, axis=0)

    server, node, user = eval_param(server, node)

    bodyid = utils.parse_bid(bodyid)

    r = requests.get('{}/api/node/{}/synapses/label/{}?relationships={}'.format(server, node, bodyid, str(with_details).lower()))

    syn = r.json()

    if pos_filter:
        # Get filter
        filtered = pos_filter(np.array([s['Pos'] for s in syn]))

        if not any(filtered):
            raise ValueError('No synapses left after filtering.')

        syn = np.array(syn)[filtered]

    return pd.DataFrame.from_records(syn)


def get_connections(source, target, pos_filter=None, server=None, node=None):
    """ Returns list of connections between source(s) and target(s).

    Parameters
    ----------
    source :            int | str
                        Body ID(s) of sources.
    target :            int | str
                        Body ID(s) of targets.
    pos_filter :        function, optional
                        Function to filter synapses by position. Must accept
                        numpy array (N, 3) and return array of [True, False, ...]
    server :            str, optional
                        If not provided, will try reading from global.
    node :              str, optional
                        If not provided, will try reading from global.

    Returns
    -------
    pandas.DataFrame
                DataFrame containing "bodyid_pre", "tbar_position",
                "tbar_confidence", "psd_position", "bodyid_post".
    """

    if not isinstance(source, (list, np.ndarray)):
        source = [source]

    if not isinstance(target, (list, np.ndarray)):
        target = [target]

    server, node, user = eval_param(server, node)

    cn_data = []

    for s in source:
        r = requests.get('{}/api/node/{}/synapses/label/{}?relationships=true'.format(server, node, s))

        # Raise
        r.raise_for_status()

        # Extract synapses
        synapses = r.json()

        # Collect downstream connections
        cn_data += [[s,
                     syn['Pos'],
                     syn['Prop']['conf'],
                     r['To']] for syn in synapses if syn['Kind'] == 'PreSyn' for r in syn['Rels']]

    cn_data = pd.DataFrame(cn_data,
                           columns=['bodyid_pre', 'tbar_position',
                                    'tbar_confidence', 'psd_position'])

    if pos_filter:
        # Get filter
        filtered = pos_filter(np.vstack(cn_data.tbar_position.values))

        if not any(filtered):
            raise ValueError('No synapses left after filtering.')

        # Filter synapses
        cn_data = cn_data.loc[filtered, :]

    # Get positions of PSDs
    pos = np.vstack(cn_data.psd_position.values)


    # Get postsynaptic body IDs
    bodies = requests.request('GET',
                              url="{}/api/node/{}/segmentation/labels".format(server, node),
                              json=pos.tolist()).json()
    cn_data['bodyid_post'] = bodies

    return cn_data[cn_data.bodyid_post.isin(target)]


def get_connectivity(bodyid, pos_filter=None, ignore_autapses=True,
                     server=None, node=None):
    """ Returns connectivity table for given body.

    Parameters
    ----------
    bodyid :            int | str
                        ID of body for which to get connectivity.
    pos_filter :        function, optional
                        Function to filter synapses by position. Must accept
                        numpy array (N, 3) and return array of [True, False, ...]
    ignore_autapses :   bool, optional
                        If True, will ignore autapses.
    server :            str, optional
                        If not provided, will try reading from global.
    node :              str, optional
                        If not provided, will try reading from global.

    Returns
    -------
    pandas.DataFrame

    """

    if isinstance(bodyid, (list, np.ndarray)):
        bodyid = np.array(bodyid).astype(str)

        cn = [get_connectivity(b, pos_filter=pos_filter,
                               ignore_autapses=ignore_autapses,
                               server=server, node=node) for b in tqdm(bodyid)]

        # Concatenate the DataFrames
        conc = []
        for r in ['upstream', 'downstream']:
            this_r = [d[d.relation==r].set_index('bodyid').drop('relation', axis=1) for d in cn]
            this_r = pd.concat(this_r, axis=1)
            this_r.columns = bodyid
            this_r['relation'] = r
            this_r = this_r[np.append('relation', bodyid)]
            conc.append(this_r.reset_index(drop=False))

        cn = pd.concat(conc, axis=0).reset_index(drop=True)
        cn = cn.fillna(0)
        cn['total'] = cn[bodyid].sum(axis=1)
        return cn.sort_values(['relation', 'total'], ascending=False).reset_index(drop=True)

    server, node, user = eval_param(server, node)

    bodyid = utils.parse_bid(bodyid)

    # Get synapses
    r = requests.get('{}/api/node/{}/synapses/label/{}?relationships=true'.format(server, node, bodyid))

    # Raise
    r.raise_for_status()

    syn = r.json()

    if pos_filter:
        # Get filter
        filtered = pos_filter(np.array([s['Pos'] for s in syn]))

        if not any(filtered):
            pass
            #raise ValueError('No synapses left after filtering.')

        # Filter synapses
        syn = np.array(syn)[filtered]

    # Collect positions and query the body IDs of pre-/postsynaptic neurons
    pos = [cn['To'] for s in syn for cn in s['Rels']]
    bodies = requests.request('GET',
                              url="{}/api/node/{}/segmentation/labels".format(server, node),
                              json=pos).json()

    # Compile connector table by counting # of synapses between neurons
    connections = {'PreSynTo': {}, 'PostSynTo': {}}
    i = 0
    for s in syn:
        # Connections point to positions -> we have to map this to body IDs
        for k, cn in enumerate(s['Rels']):
            b = bodies[i+k]
            connections[cn['Rel']][b] = connections[cn['Rel']].get(b, 0) + 1
        i += k + 1

    if connections['PreSynTo']:
        # Generate connection table
        pre = pd.DataFrame.from_dict(connections['PreSynTo'], orient='index')
        pre.columns = ['n_synapses']
        pre['relation'] = 'downstream'
    else:
        pre = pd.DataFrame([], columns=['n_synapses', 'relation'])
        pre.index = pre.index.astype(np.int64)
    pre.index.name = 'bodyid'

    if connections['PostSynTo']:
        post = pd.DataFrame.from_dict(connections['PostSynTo'], orient='index')
        post.columns = ['n_synapses']
        post['relation'] = 'upstream'
    else:
        post = pd.DataFrame([], columns=['n_synapses', 'relation'])
        post.index = post.index.astype(np.int64)
    post.index.name = 'bodyid'

    # Combine up- and downstream
    cn_table = pd.concat([pre.reset_index(), post.reset_index()], axis=0)
    cn_table.sort_values(['relation', 'n_synapses'], inplace=True, ascending=False)
    cn_table.reset_index(drop=True, inplace=True)

    if ignore_autapses:
        to_drop = cn_table.index[cn_table.bodyid==int(bodyid)]
        cn_table = cn_table.drop(index=to_drop).reset_index()

    return cn_table[['bodyid', 'relation', 'n_synapses']]


def get_adjacency(sources, targets=None, pos_filter=None, ignore_autapses=True,
                  server=None, node=None):
    """ Get adjacency between sources and targets.

    Parameters
    ----------
    sources :       iterable
                    Body IDs of sources.
    targets :       iterable, optional
                    Body IDs of targets. If not provided, targets = sources.
    pos_filter :    function, optional
                    Function to filter synapses by position. Must accept numpy
                    array (N, 3) and return array of [True, False, ...]
    server :        str, optional
                    If not provided, will try reading from global.
    node :          str, optional
                    If not provided, will try reading from global.

    Returns
    -------
    adjacency matrix :  pandas.DataFrame
                        Sources = rows; targets = columns
    """

    server, node, user = eval_param(server, node)

    if not isinstance(sources, (list, tuple, np.ndarray)):
        sources = [sources]

    if isinstance(targets, type(None)):
        targets = sources
    elif not isinstance(targets, (list, tuple, np.ndarray)):
        targets = [targets]

    # Make sure we don't have any duplicates
    sources = np.array(list(set(sources))).astype(str)
    targets = np.array(list(set(targets))).astype(str)

    # Make sure we query the smaller population from the server
    if len(targets) <= len(sources):
        columns, index, relation, to_transpose = targets, sources, 'upstream', False
    else:
        columns, index, relation, to_transpose = sources, targets, 'downstream', True

    # Get connectivity
    cn = get_connectivity(columns, pos_filter=pos_filter,
                          ignore_autapses=ignore_autapses,
                          server=server, node=node)

    # Subset connectivity to source -> target
    cn = cn[cn.relation==relation].set_index('bodyid')
    cn.index = cn.index.astype(str)
    cn = cn.reindex(index=index, columns=columns, fill_value=0)

    if to_transpose:
        cn = cn.T

    return cn

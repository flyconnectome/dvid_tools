# This code is part of dvid-tools (https://github.com/flyconnectome/dvid_tools)
# and is released under GNU GPL3

from . import fetch

import json
import networkx as nx
import pandas as pd
import numpy as np
import re
import requests
import urllib

from io import StringIO
from itertools import combinations
from scipy.spatial.distance import pdist, cdist

__all__ = ['check_skeleton', 'gen_assignments']


def swc_to_graph(x):
    """Generate NetworkX DiGraph from SWC DataFrame.

    Parameters
    ----------
    x :         pandas DataFrame
                SWC table.

    Returns
    -------
    networkx.DiGraph

    """
    if not isinstance(x, pd.DataFrame):
        raise TypeError('Expected DataFrame, got "{}"'.format(type(x)))

    edges = x.loc[x.parent_id > 0, ['node_id', 'parent_id']].values.astype(object)

    nlocs = x.set_index('node_id').loc[edges[:, 0], ['x', 'y', 'z']].values
    plocs = x.set_index('node_id').loc[edges[:, 1], ['x', 'y', 'z']].values

    weights = np.sqrt(np.sum((nlocs - plocs)**2, axis=1))

    weighted_edges = np.append(edges, weights.reshape(len(weights), 1), axis=1)

    g = nx.DiGraph()

    # Add edges
    g.add_weighted_edges_from(weighted_edges)

    # Add disconnected nodes as they do not show in edge list
    disc = x[(x.parent_id < 0) & (~x.node_id.isin(x.parent_id))]
    if not disc.empty:
        g.add_nodes_from(disc.node_id.values)

    # Add positions
    nx.set_node_attributes(g, {r.node_id: np.array([r.x, r.y, r.z]) for r in x.itertuples()}, name='location')
    nx.set_node_attributes(g, {r.node_id: r.radius for r in x.itertuples()}, name='radius')

    return g


def reroot_skeleton(x, new_root, inplace=False):
    """Reroot skeleton to new root.

    For fragmented skeletons, only the root within the new_root's connected
    component will be changed.

    Node order might not conform to SWC standard after healing. Use
    ``dvidtools.refurbish_table`` to fix that.

    Parameters
    ----------
    x :         pandas.DataFrame
                SWC table to reroot.
    new_root :  int
                New root ID.

    Returns
    -------
    SWC:        pandas.DataFrame

    """
    if not isinstance(x, pd.DataFrame):
        raise TypeError('Expected DataFrame, got "{}"'.format(type(x)))

    if new_root not in x.node_id.values:
        raise ValueError('New root node not found')

    if x.set_index('node_id').loc[new_root, 'parent_id'] < 0:
        return x

    # Turn skeleton into Graph
    g = swc_to_graph(x)

    # Walk from new root to old root and keep track of visited edges
    seen = [new_root]
    p = next(g.successors(new_root), None)
    while p:
        seen.append(p)
        p = next(g.successors(p), None)

    # Invert path and change parents
    swc = x.copy() if not inplace else x
    new_p = {v: u for u, v in zip(seen[:-1], seen[1:])}
    swc.loc[swc.node_id.isin(seen), 'parent_id'] = swc.loc[swc.node_id.isin(seen)].node_id.map(lambda x: new_p.get(x, -1))

    if not inplace:
        return swc


def heal_skeleton(x, root=None, inplace=False):
    """Merge fragmented skeleton back together.

    Uses a minimum spanning tree to connect disconnected components.

    Important
    ---------
    Node order might not conform to SWC standard after healing. Use
    ``dvidtools.refurbish_table`` to fix that.

    Parameters
    ----------
    x :         pandas.DataFrame
                SWC table to heal.
    root :      int | None, optional
                New root. If None, will use an existing root.

    Returns
    -------
    SWC:        pandas.DataFrame

    """
    if not isinstance(x, pd.DataFrame):
        raise TypeError('Expected DataFrame, got "{}"'.format(type(x)))

    # If not fragmented, simply return
    if x[x.parent_id < 0].shape[0] <= 1:
        return x

    # Turn skeleton into Graph
    g = swc_to_graph(x)

    # Set existing edges to zero weight to make sure they remain when
    # calculating the minimum spanning tree
    nx.set_edge_attributes(g, 0, 'weight')

    # Calculate distance between all leafs
    leafs = x[~x.node_id.isin(x.parent_id) | (x.parent_id < 0)]
    d = pdist(leafs[['x', 'y', 'z']].values)

    # Generate new edges
    edges = np.array(list(combinations(leafs.node_id.values, 2)))
    edges = np.append(edges.astype(object), d.reshape(d.size, 1), axis=1)

    g.add_weighted_edges_from(edges)

    # Get minimum spanning tree
    st = nx.minimum_spanning_tree(g.to_undirected())

    # Now we have to make sure that all edges are correctly oriented
    if not root:
        root = x.loc[x.parent_id < 0, 'node_id'].values[0]
    tree = nx.bfs_tree(st, source=root)
    tree = nx.reverse(tree, copy=False)

    # Now do some cleaning up
    swc = x.copy() if not inplace else x

    # Map new parents
    lop = {n: next(tree.successors(n)) for n in tree if tree.out_degree(n)}
    swc['parent_id'] = swc.node_id.map(lambda x: lop.get(x, -1))

    # Make sure parent IDs are in ascending order (according to SWC standard)
    swc.sort_values('parent_id', inplace=True)

    # Reorder node IDs
    swc.reset_index(drop=True, inplace=True)
    lon = {k: v for k, v in zip(swc.node_id.values, swc.index.values + 1)}
    swc['node_id'] = swc.index.values + 1
    swc['parent_id'] = swc.parent_id.map(lambda x: lon.get(x, -1))

    if not inplace:
        return swc


def refurbish_table(x, inplace=False):
    """Refurbishes SWC table to keep in conform with official format.

    Important
    ---------
    This operation can change node IDs!

    Parameters
    ----------
    x :         pandas.DataFrame
                SWC table to refurbish.

    Returns
    -------
    SWC:        pandas.DataFrame

    """
    if not isinstance(x, pd.DataFrame):
        raise TypeError('Expected DataFrame, got "{}"'.format(type(x)))

    swc = x.copy() if not inplace else x

    # Make sure parent IDs are in ascending order
    swc.sort_values('parent_id', inplace=True)

    # Reorder node IDs
    swc.reset_index(drop=True, inplace=True)
    lon = {k: v for k, v in zip(swc.node_id.values, swc.index.values + 1)}
    swc['node_id'] = swc.index.values + 1
    swc['parent_id'] = swc.parent_id.map(lambda x: lon.get(x, -1))

    if not inplace:
        return swc


def verify_payload(data, required, required_only=True):
    """Verify payload.

    Parameters
    ----------
    data :      list
                Data to verify.
    required :  dict
                Required entries. Can be nested. For example::
                    {'Note': str, 'User': str, 'Props': {'name': str}}

    Returns
    -------
    None

    """
    if not isinstance(data, list) or any([not isinstance(d, dict) for d in data]):
        raise TypeError('Data must be list of dicts.)')

    for d in data:
        not_required = [str(e) for e in d if e not in required]
        if any(not_required) and required_only:
            raise ValueError('Unallowed entries in data: {}'.format(','.join(not_required)))

        for e, t in required.items():
            if isinstance(t, dict):
                verify_payload([d[e]], t)
            else:
                if e not in d:
                    raise ValueError('Data must contain entry "{}"'.format(e))
                if isinstance(t, type) and not isinstance(d[e], t):
                    raise TypeError('Entry "{}" must be of type "{}"'.format(e, t))
                elif isinstance(t, list):
                    if not isinstance(d[e], list):
                        raise TypeError('"{}" must be list not "{}"'.format(e, type(d[e])))
                    for l in d[e]:
                        if not isinstance(l, tuple(t)):
                            raise TypeError('"{}" must not contain "{}"'.format(e, type(l)))


def parse_swc_str(x):
    """Parse SWC string into a pandas DataFrame.

    Parameters
    ----------
    x :     str

    Returns
    -------
    pandas.DataFrame, header

    """
    def split_iter(string):
        # Returns iterator for line split -> memory efficient
        # Attention: this will not fetch the last line!
        return (x.group(0) for x in re.finditer('.*?\n', string))

    if not isinstance(x, str):
        raise TypeError('x must be str, got "{}"'.format(type(x)))

    # Extract header using a generator -> this way we don't have to iterate
    # over all lines
    lines = split_iter(x)
    header = []
    for l in lines:
        if l.startswith('#'):
            header.append(l)
        else:
            break

    # Turn header back into string
    header = ''.join(header)

    # Turn SWC into a DataFrame
    f = StringIO(x)
    df = pd.read_csv(f, delim_whitespace=True, header=None, comment='#')

    df.columns = ['node_id', 'label', 'x', 'y', 'z', 'radius', 'parent_id']

    return df, header


def save_swc(x, filename, header=''):
    """Save SWC DataFrame to file.

    Parameters
    ----------
    x :         pandas.DataFrame
                SWC table to save.
    filename :  str
    header :    str, optional
                Header to add in front of SWC table. Each line must begin
                with "#"!

    """
    if not isinstance(x, pd.DataFrame):
        raise TypeError('Expected DataFrame, got "{}"'.format(type(x)))

    if not isinstance(header, str):
        raise TypeError('Header must be str, got "{}"'.format(type(header)))

    # Turn DataFrame back into string
    s = StringIO()
    x.to_csv(s, sep=' ', header=False, index=False)

    if not header.endswith('\n'):
        header += '\n'

    # Replace text
    swc = header + s.getvalue()

    with open(filename, 'w') as f:
        f.write(swc)


def gen_assignments(x, save_to=None, meta={}):
    """Generate JSON file that can be imported into neutu as assignments.

    Parameters
    ----------
    x :         pandas.DataFrame
                Must contain columns ``x``, ``y``, ``z`` or ``location``.
                Optional columns: ``body ID``, ``text`` and ``comment``.
    save_to :   None | filepath | filebuffer, optional
                If not None, will save json to file.
    meta :      dict, optional
                Metadata will be stored in json string as ``"metadata"``.

    Returns
    -------
    JSON-formated string
                Only if ``save_to=None``.

    """
    if not isinstance(x, pd.DataFrame):
        raise TypeError('x must be pandas DataFrame, got "{}"'.format(type(x)))

    if 'location' not in x.columns:
        if any([c not in x.columns for c in ['x', 'y', 'z']]):
            raise ValueError('x must have "location" column or "x", "y" '
                             'and "z" columns')
        x['location'] = x[['x', 'y', 'z']].astype(int).apply(list, axis=1)

    for c in ['body ID', 'text', 'comment']:
        if c not in x.columns:
            x[c] = ''

    x = x[['location', 'text', 'body ID', 'comment']]

    j = {'metadata': meta,
         'data': x.to_dict(orient='records')}

    if save_to:
        with open(save_to, 'w') as f:
            json.dump(j, f, indent=2)
    else:
        return j


def parse_bid(x):
    """Force body ID to integer."""
    try:
        return int(x)
    except BaseException:
        raise ValueError('Unable to coerce "{}" into numeric body ID'.format(x))


def _snap_to_skeleton(x, pos):
    """Snap position to closest node.

    Parameters
    ----------
    x :     pandas.DataFrame
            SWC DataFrame.
    pos :   array-like
            x/y/z position.

    Returns
    -------
    node ID :       int

    """
    if not isinstance(x, pd.DataFrame):
        raise TypeError('x must be pandas DataFrame, got "{}"'.format(type(x)))

    if not isinstance(pos, np.ndarray):
        pos = np.array(pos)

    dist = np.sum((x[['x', 'y', 'z']].values - pos)**2, axis=1)

    return x.iloc[np.argmin(dist)].node_id.astype(int)


def check_skeleton(bodyid, sample=False, node=None, server=None):
    """Test if skeleton is up-to-date.

    This works by getting the distance to the closest mesh voxel and skeleton
    node for each skeleton node and mesh voxel, respectively. The function
    returns the difference of the mean between skeleton->mesh and
    mesh->skeleton distances. A negative difference means the skeleton is
    missing branches, a positive difference indicates the skeleton having too
    many branches (e.g. if the mesh has been split).

    For large neurons, this can be costly. Use the ``sample`` parameter to
    speed up calculations.

    Parameters
    ----------
    bodyid :        int | str
                    ID of body for which to check the skeleton.
    sample :        float | int | None, optional
                    Use to restrict number of voxels/nodes used to
                    compare mesh and skeleton. If float, must be between 0
                    and 1 and will use random fraction. If int and >1, will
                    shuffle and cap number of voxels/nodes.
    server :        str, optional
                    If not provided, will try reading from global.
    node :          str, optional
                    If not provided, will try reading from global.

    Returns
    -------
    float
                    Difference of the mean between skeleton->mesh and
                    mesh->skeleton distances. A negative difference means the
                    skeleton is missing branches, a positive differene
                    indicates the skeleton having too many branches (e.g. if
                    the mesh has been split).
    None
                    If no skeleton available.

    """
    # Get SWC
    swc = fetch.get_skeleton(bodyid, node=node, server=server, verbose=False)

    if isinstance(swc, type(None)):
        return None

    swc_co = swc[['x', 'y', 'z']].values

    # Get mesh
    mesh = fetch.get_sparsevol(bodyid, scale='coarse', ret_type='COORDS',
                               node=node, server=server)

    if sample:
        if sample < 1 and sample > 0:
            mesh = mesh[np.random.choice(mesh.shape[0],
                                         int(mesh.shape[0] * sample),
                                         replace=False)]
            swc_co = swc_co[np.random.choice(swc_co.shape[0],
                                             int(swc_co.shape[0] * sample),
                                             replace=False)]
        elif sample > 1:
            if mesh.shape[0] > sample:
                mesh = mesh[np.random.choice(mesh.shape[0],
                                             int(sample),
                                             replace=False)]
            if swc_co.shape[0] > sample:
                swc_co = swc_co[np.random.choice(swc_co.shape[0],
                                                 int(sample),
                                                 replace=False)]

    # Get distances
    dist = cdist(mesh, swc_co)

    # Mesh -> SWC
    mesh_dist = np.min(dist, axis=1)
    # SWC -> Mesh
    swc_dist = np.min(dist, axis=0)

    # Return difference in mean distances
    return np.mean(swc_dist) - np.mean(mesh_dist)


def make_url(*args, **GET):
    """Generates URL.

    Parameters
    ----------
    *args
                Will be turned into the URL. For example::

                    >>> make_url('http://my-server.com', 'skeleton', 'list')
                    'http://my-server.com/skeleton/list'

    **GET
                Keyword arguments are assumed to be GET request queries
                and will be encoded in the url. For example::

                    >>> make_url('http://my-server.com', 'skeleton', node_gt=100)
                    'http://my-server.com/skeleton?node_gt=100'

    Returns
    -------
    url :       str

    """
    # Generate the URL
    url = args[0]
    for arg in args[1:]:
        arg_str = str(arg)
        joiner = '' if url.endswith('/') else '/'
        relative = arg_str[1:] if arg_str.startswith('/') else arg_str
        url = requests.compat.urljoin(url + joiner, relative)
    if GET:
        url += '?{}'.format(urllib.parse.urlencode(GET))
    return url

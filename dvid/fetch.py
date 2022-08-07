# This code is part of dvid-tools (https://github.com/flyconnectome/dvid_tools)
# and is released under GNU GPL3

import getpass
import inspect
import os
import re
import requests
import threading
import urllib
import warnings

import trimesh as tm
import numpy as np
import pandas as pd

from io import StringIO
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import lru_cache, partial
from requests.exceptions import HTTPError
from scipy.spatial.distance import cdist
from tqdm.auto import tqdm

from . import decode, meshing, utils, config

# See if navis is available
try:
    import navis
except ImportError:
    navis = None
except BaseException:
    raise

DVID_SESSIONS = {}
DEFAULT_APPNAME = "dvidtools"


__all__ = ['add_bookmarks', 'edit_annotation', 'get_adjacency', 'get_annotation',
           'get_assignment_status', 'get_available_rois',
           'get_body_position', 'get_connections',
           'get_connectivity', 'get_labels_in_area', 'get_last_mod',
           'locs_to_ids', 'get_n_synapses', 'get_roi', 'get_sparsevol',
           'get_segmentation_info', 'get_skeletons', 'get_skeleton_mutation',
           'get_synapses', 'get_user_bookmarks', 'setup', 'snap_to_body',
           'get_meshes', 'list_projects', 'get_master_node', 'get_sparsevol_size',
           'get_sizes', 'ids_exist', 'skeletonize_neuron', 'mesh_neuron']


def dvid_session(appname=DEFAULT_APPNAME, user=None):
    """Return a default requests.Session() object.

    Automatically appends the 'u' and 'app' query string parameters to every
    request. The Session object is cached, so this function will return the same
    Session object if called again from the same thread with the same arguments.
    """
    # Technically, request sessions are not threadsafe,
    # so we keep one for each thread.
    thread_id = threading.current_thread().ident
    pid = os.getpid()

    # If user not explicitly provided
    if not user:
        # Get globally defined user or fall back to system user
        user = globals().get('user', getpass.getuser())

    try:
        s = DVID_SESSIONS[(appname, user, thread_id, pid)]
    except KeyError:
        s = requests.Session()
        s.params = {'u': user, 'app': appname}
        DVID_SESSIONS[(appname, user, thread_id, pid)] = s

    return s


def setup(server=None, node=None, user=None):
    """Set default server, node and/or user.

    Parameters
    ----------
    server :    str
                URL to the dvid server.
    node :      str
                UUID of the node to query.
    user :      str
                User name. Relevant for e.g. bookmarks.

    """
    for p, n in zip([server, node, user], ['server', 'node', 'user']):
        if not isinstance(p, type(None)):
            globals()[n] = p


def eval_param(server=None, node=None, user=None, raise_missing=True):
    """Parse parameters and fall back to globally defined values."""
    parsed = []
    for p, n in zip([server, node, user], ['server', 'node', 'user']):
        if isinstance(p, type(None)):
            parsed.append(globals().get(n, None))
        else:
            parsed.append(p)

    if raise_missing and not parsed[0]:
        raise ValueError('Must provide `server` (and probably `node`) either '
                         'explicitly or globally via `dvid.setup()`')

    return parsed


def get_meshes(x, fix=True, output='auto', on_error='warn',
               max_threads=5, progress=True, server=None, node=None):
    """Fetch precomputed meshes for given body ID(s).

    Parameters
    ----------
    x :             int | str | list thereof
                    ID(s) of bodies for which to download meshes. Also
                    accepts pandas DataFrames if they have a body ID column.
    output :        "auto" | "navis" | "trimesh" | None
                    Determines the output of this function:
                     - auto = ``navis.MeshNeuron`` if navis is installed else
                       ``trimesh.Trimesh``
                     - navis = ``navis.MeshNeuron`` - raises error if ``navis``
                       not installed
                     - trimesh
    on_error :      "warn" | "skip" | "raise"
                    What to do if fetching a mesh throws an error. Typically
                    this is because there is no mesh for a given body ID
                    but it could also be a more general connection error.
    max_threads :   int
                    Max number of parallel queries to the dvid server.
    progress :      bool
                    Whether to show a progress bar or not.
    server :        str, optional
                    If not provided, will try reading from global.
    node :          str, optional
                    If not provided, will try reading from global.

    Returns
    -------
    mesh            trimesh.Trimesh | navis.MeshNeuron
                    Mutation ID is attached as `mutation_id` property
                    (is ``None`` if not available.)

    See Also
    --------
    :func:`~dvid.fetch.mesh_neuron`
                Use this to create a mesh from scratch - e.g. if there is no
                precomputed mesh for a given body.

    """
    if output == 'navis' and not navis:
        raise ImportError('Please install `navis`: pip3 install navis')

    if on_error not in ("warn", "skip", "raise"):
        raise ValueError('`on_error` must be either "warn", "skip" or "raise"')

    if max_threads < 1:
        raise ValueError('`max_threads` must be >= 1')

    if isinstance(x, pd.DataFrame):
        if 'bodyId' in x.columns:
            x = x['bodyId'].values
        elif 'bodyid' in x.columns:
            x = x['bodyid'].values
        else:
            raise ValueError('DataFrame must have "bodyId" column.')
    elif isinstance(x, int):
        x = [x]
    elif isinstance(x, str):
        x = [int(x)]

    # At this point we expect a list, set or array
    if not isinstance(x, (list, set, np.ndarray)):
        raise TypeError(f'Unexpected data type for body ID(s): "{type(x)}"')

    out = []
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = {}
        for bid in x:
            f = executor.submit(__get_ngmesh,
                                bid,
                                output=output,
                                on_error=on_error,
                                server=server, node=node)
            futures[f] = bid

        with tqdm(desc='Fetching',
                  total=len(x),
                  leave=False,
                  disable=not progress) as pbar:
            for f in as_completed(futures):
                res = f.result()
                # Skip neurons that caused an error
                if res is not None:
                    out.append(res)
                pbar.update(1)

    if (output == 'auto' and navis) or (output == 'navis'):
        out = navis.NeuronList(out)

    return out


def __get_ngmesh(bodyid, output='auto', on_error='raise',
                 check_mutation=True, server=None, node=None):
    """Load a single mesh."""
    bodyid = utils.parse_bid(bodyid)

    server, node, user = eval_param(server, node)

    url = utils.make_url(server, 'api/node', node, f'{config.segmentation}_meshes',
                         f'key/{bodyid}.ngmesh')
    r = dvid_session().get(url)

    try:
        r.raise_for_status()
    except BaseException:
        if on_error == 'raise':
            raise
        elif on_error == 'warn':
            warnings.warn(f'{bodyid}: {r.text.strip()}')
        return

    # Decode mesh
    m = decode.read_ngmesh(r.content)

    # Grab mutation ID
    if check_mutation:
        url = utils.make_url(server, 'api/node/', node, f'{config.segmentation}_meshes_mutid',
                             f'key/{bodyid}')
        r = dvid_session().get(url)
        # This query won't work with older nodes
        # -> hence wrapping it in a try/except block
        try:
            r.raise_for_status()
            m.mutation_id = int(r.json())
        except HTTPError:
            m.mutation_id = None
        except BaseException:
            raise

    if (output == 'auto' and navis) or (output == 'navis'):
        n = navis.MeshNeuron(m, id=bodyid, mutation_id=m.mutation_id)
        return n

    return m


def get_skeletons(x, save_to=None, output='auto', on_error='warn',
                  check_mutation=False, max_threads=5, progress=True,
                  server=None, node=None):
    """Fetch skeleton for given body ID.

    Parameters
    ----------
    x :             int | str | list thereof
                    ID(s) of bodies for which to download skeletons. Also
                    accepts pandas DataFrames if they have a body ID column.
    save_to :       str | None, optional
                    If provided, will save raw SWC to file. If str must be file
                    or path.
    output :        "auto" | "navis" | "swc" | None
                    Determines the output of this function:
                     - auto = ``navis.TreeNeuron`` if navis is installed else
                       SWC table as ``pandas.DataFrame``
                     - navis = ``navis.TreeNeuron`` - raises error if ``navis``
                       not installed
                     - swc = SWC table as ``pandas.DataFrame``
                     - None = no direct output - really only relevant if you
                       want to only save the SWC to a file
    on_error :      "warn" | "skip" | "raise"
                    What to do if fetching a skeleton throws an error. Typically
                    this is because there is no skeleton for a given body ID
                    but it could also be a more general connection error.
    check_mutation : bool, optional
                    If True, will check if skeleton and body are still in-sync
                    using the mutation IDs. Will warn if mismatch found.
    max_threads :   int
                    Max number of parallel queries to the dvid server.
    progress :      bool
                    Whether to show a progress bar or not.
    server :        str, optional
                    If not provided, will try reading from global.
    node :          str, optional
                    If not provided, will try reading from global.

    Returns
    -------
    SWC :           pandas.DataFrame
                    Only if ``save_to=None`` else ``True``.
    None
                    If no skeleton found.

    See Also
    --------
    :func:`dvid.get_skeleton_mutation`
                    If you want to create a skeleton based on the current
                    voxels yourself.
    :func:`~dvid.fetch.skeletonize_neuron`
                    Use this to create a skeleton from scratch - e.g. if there
                    is no precomputed skeleton for a given body.

    Examples
    --------
    Fetch neuron as navis skeleton

    >>> dt.get_skeleton(485775679)

    Grab a neuron and save it directly to a file

    >>> dt.get_skeleton(485775679, save_to='~/Downloads/', output=None)

    """
    if output == 'navis' and not navis:
        raise ImportError('Please install `navis`: pip3 install navis')

    if on_error not in ("warn", "skip", "raise"):
        raise ValueError('`on_error` must be either "warn", "skip" or "raise"')

    if max_threads < 1:
        raise ValueError('`max_threads` must be >= 1')

    if isinstance(x, pd.DataFrame):
        if 'bodyId' in x.columns:
            x = x['bodyId'].values
        elif 'bodyid' in x.columns:
            x = x['bodyid'].values
        else:
            raise ValueError('DataFrame must have "bodyId" column.')
    elif isinstance(x, int):
        x = [x]
    elif isinstance(x, str):
        x = [int(x)]

    # At this point we expect a list, set or array
    if not isinstance(x, (list, set, np.ndarray)):
        raise TypeError(f'Unexpected data type for body ID(s): "{type(x)}"')

    if len(x) > 1:
        if save_to and not os.path.isdir(save_to):
            raise ValueError('"save_to" must be path when loading multiple'
                             'multiple skeletons')

    out = []
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = {}
        for bid in x:
            f = executor.submit(__get_skeleton,
                                bid,
                                save_to=save_to,
                                output=output,
                                on_error=on_error,
                                check_mutation=check_mutation,
                                server=server, node=node)
            futures[f] = bid

        with tqdm(desc='Fetching',
                  total=len(x),
                  leave=False,
                  disable=not progress) as pbar:
            for f in as_completed(futures):
                res = f.result()
                pbar.update(1)
                if on_error == "skip" and res is None:
                    continue
                out.append(res)

    if (output == 'auto' and navis) or (output == 'navis'):
        out = navis.NeuronList(out)

    return out


def __get_skeleton(bodyid, save_to=None, output='auto', on_error='raise',
                   check_mutation=False, server=None, node=None):
    """Load a single skeleton."""
    bodyid = utils.parse_bid(bodyid)

    server, node, user = eval_param(server, node)

    url = urllib.parse.urljoin(server,
                               'api/node/{}/{}_skeletons/key/{}_swc'.format(node,
                                                                            config.segmentation,
                                                                            bodyid))

    try:
        r = dvid_session().get(url)
        r.raise_for_status()
    except BaseException:
        if on_error == 'raise':
            raise
        elif on_error == 'warn':
            warnings.warn(f'{bodyid}: {r.text.strip()}')
        return

    # Save raw SWC before making any changes
    if save_to:
        # Generate proper filename if necessary
        if os.path.isdir(save_to):
            save_raw_to = os.path.join(save_to, '{}.swc'.format(bodyid))
        else:
            # ... else assume it's a file
            save_raw_to = save_to

        with open(save_raw_to, 'w') as f:
            f.write(r.text)

    # Stop here is no further output required
    if not output:
        return

    # Parse SWC string
    swc, header = utils.parse_swc_str(r.text)
    swc.header = header

    if 'mutation id' in header:
        swc.mutation_id = int(re.search('"mutation id": (.*?)}', header).group(1))
    else:
        swc.mutation_id = None

    if check_mutation:
        if not getattr(swc, 'mutation_id', None):
            print('{} - Unable to check mutation: mutation ID not in '
                  'SWC header'.format(bodyid))
        else:
            body_mut = get_last_mod(bodyid,
                                    server=server,
                                    node=node).get('mutation id')
            if swc.mutation_id != body_mut:
                print("{}: mutation IDs of skeleton and mesh don't match. "
                      "The skeleton might not be up-to-date.".format(bodyid))

    if (output == 'auto' and navis) or (output == 'navis'):
        n = navis.TreeNeuron(swc, id=bodyid)
        n.header = header
        n.mutation_id = getattr(swc, 'mutation_id', None)
        return n

    return swc


def __old_get_skeleton(bodyid, save_to=None, xform=None, root=None, soma=None,
                       heal=False, check_mutation=True, server=None, node=None,
                       verbose=True, **kwargs):
    """Download skeleton as SWC file.

    Parameters
    ----------
    bodyid :        int | str
                    ID(s) of body for which to download skeleton.
    save_to :       str | None, optional
                    If provided, will save SWC to file. If str must be file or
                    path. Please note that using ``heal`` or ``reroot`` will
                    require the SWC table to be cleaned-up using
                    ``dvidtools.refurbish_table`` before saving to keep it in
                    line with SWC format. This will change node IDs!
    xform :         function, optional
                    If provided will run this function to transform coordinates
                    before saving/returning the SWC file. Function must accept
                    ``(N, 3)`` numpy array. Nodes that don't transform properly
                    will be removed and disconnected piece will be healed.
    soma :          array-like | function, optional
                    Use to label ("1") and reroot to soma node:
                      - array-like is interpreted as x/y/z position and will be
                        mapped to the closest node
                      - ``function`` must accept ``bodyid`` and return x/y/z
                        array-like
    root :          array-like | function, optional
                    Use to reroot the neuron to given node:
                      - array-like is interpreted as x/y/z position and will be
                        mapped to the closest node
                      - ``function`` must accept ``bodyid`` and return x/y/z
                        array-like
                    This will override ``soma``.
    heal :          bool, optional
                    If True, will heal fragmented neurons using
                    ``dvidtools.heal_skeleton``.
    check_mutation : bool, optional
                    If True, will check if skeleton and body are still in-sync
                    using the mutation IDs. Will warn if mismatch found.
    server :        str, optional
                    If not provided, will try reading from global.
    node :          str, optional
                    If not provided, will try reading from global.

    Returns
    -------
    SWC :       pandas.DataFrame
                Only if ``save_to=None`` else ``True``.
    None
                If no skeleton found.

    Examples
    --------
    Easy: grab a neuron and save it to a file

    >>> dt.get_skeleton(485775679, save_to='~/Downloads/')

    """
    if isinstance(bodyid, (list, np.ndarray)):
        if save_to and not os.path.isdir(save_to):
            raise ValueError('"save_to" must be path when loading multiple'
                             'multiple bodies')
        resp = {x: __old_get_skeleton(x,
                                      save_to=save_to,
                                      check_mutation=check_mutation,
                                      verbose=verbose,
                                      heal=heal,
                                      soma=soma,
                                      xform=xform,
                                      server=server,
                                      node=node,
                                      **kwargs) for x in tqdm(bodyid,
                                                              desc='Loading')}
        # Give summary
        missing = [str(k) for k, v in resp.items() if isinstance(v, type(None))]
        print('{}/{} skeletons successfully downloaded.'.format(len(resp) - len(missing),
                                                                len(resp)))
        if missing:
            print('Missing skeletons for: {}'.format(','.join(missing)))

        if not save_to:
            return resp
        else:
            return

    bodyid = utils.parse_bid(bodyid)

    server, node, user = eval_param(server, node)

    r = dvid_session().get(urllib.parse.urljoin(server,
                                                'api/node/{}/{}_skeletons/key/{}_swc'.format(node,
                                                                                             config.segmentation,
                                                                                             bodyid)))

    if 'not found' in r.text:
        if verbose:
            print(r.text)
        return None

    # Save raw SWC before making any changes
    save_raw_to = kwargs.get('save_raw_to', False)
    if save_raw_to:
        # Generate proper filename if necessary
        if os.path.isdir(save_raw_to):
            save_raw_to = os.path.join(save_raw_to, '{}.swc'.format(bodyid))

        with open(save_raw_to, 'w') as f:
            f.write(r.text)

    # Parse SWC string
    df, header = utils.parse_swc_str(r.text)

    if 'mutation id' in header:
        df.mutation_id = int(re.search('"mutation id": (.*?)}', header).group(1))

    if check_mutation:
        if not getattr(df, 'mutation_id', None):
            print('{} - Unable to check mutation: mutation ID not in '
                  'SWC header'.format(bodyid))
        else:
            body_mut = get_last_mod(bodyid,
                                    server=server,
                                    node=node).get('mutation id')
            if df.mutation_id != body_mut:
                print("{}: mutation IDs of skeleton and mesh don't match. "
                      "The skeleton might not be up-to-date.".format(bodyid))

    # Heal first as this might change node IDs
    if heal:
        utils.heal_skeleton(df, inplace=True)

    # If soma is function, call it
    if callable(soma):
        soma = soma(bodyid)

    # If root is function, call it
    if callable(root):
        root = root(bodyid)

    # If we have a soma
    if isinstance(soma, (list, tuple, np.ndarray)):
        # Get soma node
        soma_node = utils._snap_to_skeleton(df, soma)

        # Set label
        df.loc[df.node_id == soma_node, 'label'] = 1

        # If root is not explicitly provided, reroot to soma
        if not isinstance(root, (list, tuple, np.ndarray)):
            root = soma

    # If we have a root
    if isinstance(root, (list, tuple, np.ndarray)):
        # Get soma node
        root_node = utils._snap_to_skeleton(df, root)

        # Reroot
        utils.reroot_skeleton(df, root_node, inplace=True)

    if callable(xform):
        # Add xform function to header for documentation
        header += '#x/y/z coordinates transformed by dvidtools using this:\n'
        header += '\n'.join(['#' + l for l in inspect.getsource(xform).split('\n') if l])

        # Transform coordinates
        df.iloc[:, 2:5] = xform(df.iloc[:, 2:5].values)

        # Check if any coordinates got messed up
        nans = df[np.any(df.iloc[:, 2:5].isnull(), axis=1)]
        if not nans.empty:
            if verbose:
                print('{} nodes did not xform - removing & stitching...'.format(nans.shape[0]))
            # Drop nans
            df.drop(nans.index, inplace=True)
            # Keep track of existing root (if any left)
            root = df.loc[df.parent_id < 0, 'node_id']
            root = root.values[0] if not root.empty else None
            # Set orphan nodes to roots
            df.loc[~df.parent_id.isin(df.node_id), 'parent_id'] = -1
            # Heal fragments
            utils.heal_skeleton(df, root=root, inplace=True)
    elif not isinstance(xform, type(None)):
        raise TypeError('"xform" must be a function, not "{}"'.format(type(xform)))

    if save_to:
        # Make sure table is still conform with SWC format
        if heal or not isinstance(root, type(None)):
            df = utils.refurbish_table(df)

        # Generate proper filename if necessary
        if os.path.isdir(save_to):
            save_to = os.path.join(save_to, '{}.swc'.format(bodyid))

        # Save SWC file
        utils.save_swc(df, filename=save_to, header=header)
        return True
    else:
        return df


def get_user_bookmarks(server=None, node=None, user=None,
                       return_dataframe=True):
    """Get user bookmarks.

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

    r = dvid_session().get(urllib.parse.urljoin(server,
                                                'api/node/{}/bookmark_annotations/tag/user:{}'.format(node,
                                                                                                      user)))

    if return_dataframe:
        data = r.json()
        for d in data:
            d.update(d.pop('Prop'))
        return pd.DataFrame.from_records(data)
    else:
        return r.json()


def add_bookmarks(data, verify=True, server=None, node=None):
    """Add or edit user bookmarks.

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

    r = dvid_session().post(urllib.parse.urljoin(server,
                                                 'api/node/{}/bookmark_annotations/elements'.format(node)),
                            json=data)

    r.raise_for_status()

    return


def get_annotation(bodyid, server=None, node=None, verbose=True):
    """Fetch annotations for given body.

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

    r = dvid_session().get(urllib.parse.urljoin(server,
                                                'api/node/{}/{}_annotations/key/{}'.format(node,
                                                                                           config.body_labels,
                                                                                           bodyid)))

    try:
        return r.json()
    except BaseException:
        if verbose:
            print(r.text)
        return {}


def edit_annotation(bodyid, annotation, server=None, node=None, verbose=True):
    """Edit annotations for given body.

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
    verbose :       bool, optional
                    If True, will warn if entirely new annotations are added
                    (as opposed to just updating existing annotations)

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
    if not isinstance(annotation, dict):
        raise TypeError('Annotation must be dictionary, not "{}"'.format(type(annotation)))

    server, node, user = eval_param(server, node)

    bodyid = utils.parse_bid(bodyid)

    # Get existing annotations
    old_an = get_annotation(bodyid, server=server, node=node, verbose=False)

    # Raise non-standard payload
    new_an = [k for k in annotation if k not in old_an]
    if new_an and verbose:
        warnings.warn('Adding new annotation(s) to {}: {}'.format(bodyid,
                                                                  ', '.join(new_an)))

    # Update annotations
    old_an.update(annotation)

    r = dvid_session().post(urllib.parse.urljoin(server,
                                                 'api/node/{}/{}_annotations/key/{}'.format(node,
                                                                                            config.body_labels,
                                                                                            bodyid)),
                            json=old_an)

    # Check if it worked
    r.raise_for_status()

    return None


def __old_get_body_id(pos, server=None, node=None):
    """Get body ID at given position.

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

    r = dvid_session().get(urllib.parse.urljoin(server,
                                                'api/node/{}/{}/label/{}_{}_{}'.format(node,
                                                                                       config.segmentation,
                                                                                       pos[0], pos[1],
                                                                                       pos[2])))

    return r.json()['Label']


def locs_to_ids(pos, chunk_size=10e3, progress=True, server=None, node=None):
    """Get body IDs at given positions.

    Parameters
    ----------
    pos :           iterable
                    [[x1, y1, z1], [x2, y2, z2], ..] positions in voxel space.
                    Must be integers!
    chunk_size :    int, optional
                    Splits query into chunks of a given size to reduce strain on
                    server.
    progress :      bool
                    If True, show progress bar.
    server :        str, optional
                    If not provided, will try reading from global.
    node :          str, optional
                    If not provided, will try reading from global.

    Returns
    -------
    body_ids :      np.ndarray

    """
    server, node, user = eval_param(server, node)

    pos = np.asarray(pos)

    if pos.ndim == 1:
        pos.reshape(1, 3)

    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f'Expected (N, 3) array of positions, got {pos.shape}')

    # Make sure we are working on integers
    pos = pos.astype(int)

    data = []
    with tqdm(desc='Querying positions', total=len(pos), disable=not progress,
              leave=False) as pbar:
        for ix in range(0, len(pos), int(chunk_size)):
            chunk = pos[ix: ix + int(chunk_size)]

            url = utils.make_url(server, f'api/node/{node}/{config.segmentation}/labels')
            r = dvid_session().get(url, json=chunk.tolist())

            r.raise_for_status()
            data += r.json()
            pbar.update(len(chunk))

    return np.array(data)


def get_sizes(ids, chunk_size=1e3, progress=True, server=None, node=None):
    """Get sizes (in supervoxels) for given body IDs.

    Parameters
    ----------
    ids :           iterable
                    [12345, 455677, ...] IDs. Must be integers!
    chunk_size :    int, optional
                    Splits query into chunks of a given size to reduce strain on
                    server.
    progress :      bool
                    If True, show progress bar.
    server :        str, optional
                    If not provided, will try reading from global.
    node :          str, optional
                    If not provided, will try reading from global.

    Returns
    -------
    sizes :         np.ndarray
                    For IDs that do not exist the size will be 0.

    """
    server, node, user = eval_param(server, node)

    ids = np.asarray(ids)

    if ids.ndim != 1:
        raise ValueError(f'Expected (N, ) array of IDs, got {ids.shape}')

    # Make sure we are working on integers
    ids = ids.astype(int)

    data = []
    with tqdm(desc='Querying sizes', total=len(ids), disable=not progress,
              leave=False) as pbar:
        for ix in range(0, len(ids), int(chunk_size)):
            chunk = ids[ix: ix + int(chunk_size)]

            url = utils.make_url(server, f'api/node/{node}/{config.segmentation}/sizes')
            r = dvid_session().get(url, json=chunk.tolist())

            r.raise_for_status()
            data += r.json()
            pbar.update(len(chunk))

    return np.array(data)


def ids_exist(ids, chunk_size=1e3, progress=True, server=None, node=None):
    """Check if given IDs exist.

    Parameters
    ----------
    ids :           iterable
                    [12345, 455677, ...] IDs. Must be integers!
    chunk_size :    int, optional
                    Splits query into chunks of a given size to reduce strain on
                    server.
    progress :      bool
                    If True, show progress bar.
    server :        str, optional
                    If not provided, will try reading from global.
    node :          str, optional
                    If not provided, will try reading from global.

    Returns
    -------
    exists :        np.ndarray
                    Array of booleans.

    """
    server, node, user = eval_param(server, node)

    ids = np.asarray(ids)

    if ids.ndim != 1:
        raise ValueError(f'Expected (N, ) array of IDs, got {ids.shape}')

    # Make sure we are working on integers
    ids = ids.astype(int)

    data = []
    with tqdm(desc='Querying', total=len(ids), disable=not progress,
              leave=False) as pbar:
        for ix in range(0, len(ids), int(chunk_size)):
            chunk = ids[ix: ix + int(chunk_size)]

            url = utils.make_url(server, f'api/node/{node}/{config.segmentation}/sizes')
            r = dvid_session().get(url, json=chunk.tolist())

            r.raise_for_status()
            data += r.json()
            pbar.update(len(chunk))

    sizes = np.array(data)

    return sizes != 0


def get_body_position(bodyid, server=None, node=None):
    """Get a single position for given body ID.

    This will (like neutu) use the skeleton. If body has no skeleton, will
    use mesh as fallback.

    Parameters
    ----------
    bodyid :    int | str
                Body ID for which to find a position.
    server :    str, optional
                If not provided, will try reading from global.
    node :      str, optional
                If not provided, will try reading from global.

    Returns
    -------
    (x, y, z)

    """
    bodyid = utils.parse_bid(bodyid)

    s = get_skeletons(bodyid, server=server, node=node)

    if isinstance(s, pd.DataFrame) and not s.empty:
        # Return the root (more likely to be actually within the mesh?)
        return s.loc[0, ['x', 'y', 'z']].values
    else:
        # First get voxels of the coarse neuron
        voxels = get_sparsevol(bodyid, scale='coarse', ret_type='INDEX',
                               server=server, node=node)

        # Erode surface voxels to make sure we get a central position
        while True:
            eroded = meshing.remove_surface_voxels(voxels)

            # Stop before no more voxels left
            if eroded.size == 0:
                break

            voxels = eroded

        # Get voxel sizes based on scale
        info = get_segmentation_info(server, node)['Extended']

        # Now query the more precise mesh for this coarse voxel
        # Pick a random voxel
        v = voxels[0] * info['BlockSize'] # turn into locs for scale 0
        # Generate a bounding bbox
        bbox = np.vstack([v, v]).T
        bbox[:, 1] += info['BlockSize']

        voxels = get_sparsevol(bodyid, scale=0, ret_type='INDEX',
                               bbox=bbox.ravel(),
                               server=server, node=node)

        # Erode surface voxels again to make sure we get a central position
        while True:
            eroded = meshing.remove_surface_voxels(voxels)

            # Stop before no more voxels left
            if eroded.size == 0:
                break

            voxels = eroded

        return voxels[0]


def get_assignment_status(pos, window=None, bodyid=None, server=None, node=None):
    """Return assignment status at given position.

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

        r = dvid_session().get(urllib.parse.urljoin(server,
                                                    'api/node/{}/bookmarks/keyrange/'
                                                    '{}_{}_{}/{}_{}_{}'.format(node,
                                                                               int(pos[0]-window[0]/2),
                                                                               int(pos[1]-window[1]/2),
                                                                               int(pos[2]-window[2]/2),
                                                                               int(pos[0]+window[0]/2),
                                                                               int(pos[1]+window[1]/2),
                                                                               int(pos[2]+window[2]/2),)))
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

            bids = np.array(locs_to_ids(coords,
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
    r = dvid_session().get(urllib.parse.urljoin(server,
                                                'api/node/{}/bookmarks/key/{}_{}_{}'.format(node,
                                                                                            int(pos[0]),
                                                                                            int(pos[1]),
                                                                                            int(pos[2]))))

    # Will raise if key not found -> so just don't
    # r.raise_for_status()

    return r.json() if r.text and 'not found' not in r.text else None


def get_labels_in_area(offset, size, server=None, node=None):
    """Get labels (todo, to split, etc.) in given bounding box.

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

    r = dvid_session().get(urllib.parse.urljoin(server,
                                                'api/node/{}/{}_todo/elements/'
                                                '{}_{}_{}/{}_{}_{}'.format(node,
                                                                           config.segmentation,
                                                                           int(size[0]),
                                                                           int(size[1]),
                                                                           int(size[2]),
                                                                           int(offset[0]),
                                                                           int(offset[1]),
                                                                           int(offset[2]))))

    r.raise_for_status()

    j = r.json()

    if j:
        return pd.DataFrame.from_records(r.json())
    else:
        return None


def get_available_rois(server=None, node=None, step_size=2):
    """Get a list of all available ROIs in given node.

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

    r = dvid_session().get(urllib.parse.urljoin(server,
                                                'api/node/{}/rois/keys'.format(node)))

    r.raise_for_status()

    return r.json()


def get_roi(roi, step_size=2, form='MESH', save_to=None, server=None, node=None):
    """Get ROI as mesh or voxels.

    Uses marching cube algorithm to extract surface model of ROI voxels if no
    precomputed mesh available.

    Note that some ROIs exist only as voxels or as mesh. If voxels are requested
    but not available, will raise "Bad Request" HttpError.

    Parameters
    ----------
    roi :           str
                    Name of ROI.
    form :          "MESH" | "VOXELS"| "BLOCKS", optional
                    Returned format - see ``Returns``.
    step_size :     int, optional
                    Step size for marching cube algorithm. Only relevant for
                    ``form="MESH"``. Smaller values = higher resolution but
                    slower.
    save_to :       filename
                    If provided will also write mesh straight to file. Should
                    end with `.obj`.
    server :        str, optional
                    If not provided, will try reading from global.
    node :          str, optional
                    If not provided, will try reading from global.

    Returns
    -------
    mesh :          trimesh.Trimesh
                    If ``format=='MESH'``. Coordinates in nm.
    voxels :        numpy array
                    If ``format=='VOXELS'``.
    blocks :        numpy array
                    If ``format=='BLOCKS'``. Encode blocks of voxels as
                    4 coordinates: ``[z, y, x_start, x_end]``

    """
    if form.upper() not in ('MESH', 'VOXELS', 'BLOCKS'):
        raise ValueError('Unknown return format "{}"'.format(form))

    server, node, user = eval_param(server, node)

    if form.upper() == 'MESH':
        # Check if we can get the OBJ file directly
        try:
            # Get the key for this roi
            r = dvid_session().get(urllib.parse.urljoin(server, 'api/node/{}/rois/key/{}'.format(node, roi)))
            r.raise_for_status()
            key = r.json()['->']['key']

            # Get the obj string
            r = dvid_session().get(urllib.parse.urljoin(server, 'api/node/{}/roi_data/key/{}'.format(node, key)))
            r.raise_for_status()

            if save_to:
                with open(save_to, 'w') as f:
                    f.write(r.text)
                return

            # The data returned is in .obj format
            return tm.load(StringIO(r.text), file_type='obj')
        except BaseException:
            if r.status_code != 400:
                raise

    # Get the voxels
    r = dvid_session().get(urllib.parse.urljoin(server, 'api/node/{}/{}/roi'.format(node, roi)))
    r.raise_for_status()

    # The data returned are block coordinates: [z, y, x_start, x_end]
    blocks = np.array(r.json())

    if form.upper() == 'BLOCKS':
        return blocks

    voxels = meshing._blocks_to_voxels(blocks)

    if form.upper() == 'VOXELS':
        return voxels

    # Try getting voxel size
    meta = get_segmentation_info(node=node, server=server)
    if 'Extended' in meta and 'BlockSize' in meta['Extended']:
        voxel_size = tuple(meta['Extended']['BlockSize'])
    else:
        print('No voxel size found. Mesh returned in raw voxels.')
        voxel_size = (1, 1, 1)

    mesh = meshing.mesh_from_voxels(voxels,
                                    spacing=voxel_size,
                                    step_size=step_size)

    if save_to:
        mesh.export(save_to)

    return mesh


def skeletonize_neuron(bodyid,
                       scale=4,
                       server=None,
                       node=None,
                       progress=True,
                       **kwargs):
    """Skeletonize given body.

    Fetches voxels from DVID, creates a mesh (via `mesh_neuron`) and then
    skeletonizes it. This can be useful if the precomputed skeletons are not
    up-to-date or have incorrect topology. This function requires `skeletor` to
    be installed::

      pip3 install skeletor

    Parameters
    ----------
    bodyid :    int | str | trimesh
                ID of body for which to generate skeleton.
    scale :     int | "COARSE"
                Resolution of sparse volume to use for skeletonization.
                Lower = higher res. Higher resolutions tend to produce more
                accurate but also more noisy (e.g. tiny free-floating fragments)
                skeletons. In my experience, `scale="COARSE"` for quick & dirty
                and `scale=4` for high-quality skeletons make the most sense.
                Scales 5 and 6 are too coarse, and below 3 becomes prohibitively
                slow.
    server :    str, optional
                If not provided, will try reading from global.
    node :      str, optional
                If not provided, will try reading from global.
    progress :  bool
                Whether to show progress bars.
    **kwargs
                Keyword arguments are passed through to `mesh_from_voxels`.

    Returns
    -------
    skeletor.Skeleton
                In "nm" coordinates.

    See Also
    --------
    :func:`dvid.get_skeletons`
                To download precomputed skeletons.

    """
    try:
        import skeletor as sk
    except ImportError:
        raise ImportError('`skeletonize_neuron` requires `skeletor` to be installed')
    except BaseException:
        raise

    defaults = dict(step_size=1)
    defaults.update(kwargs)

    # Get the sparse-vol mesh
    if isinstance(bodyid, tm.Trimesh):
        mesh = bodyid
    else:
        mesh = mesh_neuron(bodyid, scale=scale,
                           server=server, node=node,
                           progress=progress,
                           **defaults)

    # Skeletonize
    return sk.skeletonize.by_wavefront(mesh, radius_agg='median',
                                       progress=progress)


def get_sparsevol_size(bodyid, server=None, node=None):
    """Fetch sparsevol (voxel) info for given neuron.

    Parameters
    ----------
    bodyid :    int | str
                ID of body for which to download mesh.
    server :    str, optional
                If not provided, will try reading from global.
    node :      str, optional
                If not provided, will try reading from global.


    Returns
    -------
    dict
                Dict with number of voxels and coarse bounding box in voxel
                space.

    """
    server, node, user = eval_param(server, node)

    bodyid = utils.parse_bid(bodyid)

    url = urllib.parse.urljoin(server, 'api/node/{}/{}/sparsevol-size/{}'.format(node,
                                                                                 config.segmentation,
                                                                                 bodyid))

    r = dvid_session().get(url)
    r.raise_for_status()

    return r.json()


def get_sparsevol(bodyid,
                  scale='COARSE',
                  ret_type='INDEX',
                  voxels=True,
                  save_to=None,
                  bbox=None,
                  server=None,
                  node=None):
    """Fetch sparsevol (voxel) representation for given neuron.

    Parameters
    ----------
    bodyid :    int | str
                ID of body for which to download mesh.
    scale :     int | "COARSE", optional
                Resolution of sparse volume starting with 0 where each level
                beyond 0 has 1/2 resolution of previous level. "COARSE" will
                return the volume in block coordinates.
    save_to :   str | None, optional
                If provided, will save response from server as binary file.
    ret_type :  "INDEX" | "COORDS" | "RAW"
                "INDEX" returns x/y/z indices. "COORDS" returns x/y/z
                coordinates. "RAW" will return server response as raw bytes.
    voxels :    bool
                If False, will return x/y/z/x_run_length instead of x/y/z voxels.
    bbox :      list | None, optional
                Bounding box to which to restrict the query to. Must be in
                `scale=0` index coordinates.
                Format: ``[x_min, x_max, y_min, y_max, z_min, z_max]``.
    server :    str, optional
                If not provided, will try reading from global.
    node :      str, optional
                If not provided, will try reading from global.


    Returns
    -------
    voxels :    (N, 3) np.array
                If ``voxels=True``: array with x/y/z coordinates/indices
    rles :      (N, 4) np.array
                If ``voxels=False``: array with x/y/z/x_run_length
                coordinates/indices.
    raw :       bytes string
                If ``ret_type='RAW'``: server response as raw bytes.

    """
    if ret_type.upper() not in ('COORDS', 'INDEX', 'RAW'):
        raise ValueError('"ret_type" must be "COORDS", "INDEX" or "RAW"')

    server, node, user = eval_param(server, node)

    bodyid = utils.parse_bid(bodyid)

    # Get voxel sizes based on scale
    info = get_segmentation_info(server, node)['Extended']

    vsize = {'COARSE': [s * 8 for s in info['BlockSize']]}
    vsize.update({i: np.array(info['VoxelSize']) * 2**i for i in range(info['MaxDownresLevel'])})

    if isinstance(scale, int) and scale > info['MaxDownresLevel']:
        raise ValueError('Scale greater than MaxDownresLevel')
    elif isinstance(scale, str):
        scale = scale.upper()

    options = {}
    if scale == 'COARSE':
        url = urllib.parse.urljoin(server, 'api/node/{}/{}/sparsevol-coarse/{}'.format(node,
                                                                                       config.segmentation,
                                                                                       bodyid))
    elif isinstance(scale, (int, np.number)):
        url = urllib.parse.urljoin(server, 'api/node/{}/{}/sparsevol/{}'.format(node,
                                                                                config.segmentation,
                                                                                bodyid))
        options['scale'] = scale
    else:
        raise TypeError(f'scale must be "COARSE" or integer, not "{scale}"')

    if not isinstance(bbox, type(None)):
        bbox = np.asarray(bbox).astype(int)
        if bbox.shape != (6, ):
            raise ValueError(f'Bounding box must be shape (6, ), got {bbox.shape}')
        for key, co in zip(['minx', 'maxx', 'miny', 'maxy', 'minz', 'maxz'], bbox):
            options[key] = co

    if options:
        url += '?' + '&'.join([f'{k}={v}' for k, v in options.items()])

    if not isinstance(bbox, type(None)):
        check = dvid_session().head(url)
        if check.status_code != 200:
            raise ValueError(f'There is no sparse volume for {bodyid} '
                             'within the specified bounds.')

    r = dvid_session().get(url)
    r.raise_for_status()

    b = r.content

    if save_to:
        with open(save_to, 'wb') as f:
            f.write(b)
        return

    if ret_type.upper() == 'RAW':
        return b

    # Decode binary format
    header, voxels = decode.decode_sparsevol(b, format='rles', voxels=voxels)

    if ret_type.upper() == 'COORDS':
        voxels = voxels[:, :3] * vsize[scale]
        if voxels.shape[1] == 4:
            voxels[:, 4] *= vsize[scale][0]

    return voxels


def mesh_neuron(bodyid,
                scale='COARSE',
                step_size=1,
                bbox=None,
                parallel=False,
                progress=True,
                server=None,
                node=None,
                **kwargs):
    """Create mesh for given neuron(s).

    Parameters
    ----------
    bodyid :    int | str | list-like
                Body ID(s) for which to generate mesh.
    scale :     int | "COARSE", optional
                Resolution of sparse volume starting with 0 where each level
                beyond 0 has 1/2 resolution of previous level. "COARSE" will
                return the volume in block coordinates.
    step_size : int, optional
                Step size for marching cube algorithm.
                Higher values = faster but coarser.
    bbox :      list | None, optional
                Bounding box to which to restrict the meshing to. Must be in
                `scale=0` coordinates.
                Format: ``[x_min, x_max, y_min, y_max, z_min, z_max]``.
    parallel :  bool | int
                Whether to run meshing in parallel on multiple cores if
                `bodyid` is more than one neuron. If `parallel` is integer will
                use that many cores. Otherwise defaults to half the available
                cores.
    progress :  bool
                Whether to show a progress bar when meshing multiple neurons.
    server :    str, optional
                If not provided, will try reading from global.
    node :      str, optional
                If not provided, will try reading from global.
    **kwargs
                Keyword arguments are passed through to
                `dv.meshing.mesh_from_voxels`.


    Returns
    -------
    trimesh.Trimesh

    See Also
    --------
    :func:`~dvid.fetch.get_meshes`
                Use this to fetch precomputed meshes instead of making our own.

    """
    server, node, user = eval_param(server, node)

    if isinstance(bodyid, (list, tuple, np.ndarray)):
        if len(bodyid) == 1:
            bodyid = bodyid[0]
        else:
            func = partial(mesh_neuron,
                           scale=scale,
                           step_size=step_size,
                           bbox=bbox,
                           server=server,
                           node=node,
                           **kwargs)

            if not parallel:
                return [func(b) for b in tqdm(bodyid,
                                              desc='Meshing',
                                              disable=not progress,
                                              leave=False)]
            else:
                meshes = []
                n_cores = parallel if isinstance(parallel, int) else max(1, int(os.n_cores() // 2))
                with ProcessPoolExecutor(max_workers=n_cores) as executor:
                    futures = {}
                    for bid in bodyid:
                        f = executor.submit(func, bid)
                        futures[f] = bid

                    with tqdm(desc='Meshing',
                              total=len(bodyid),
                              leave=False,
                              disable=not progress) as pbar:
                        for f in as_completed(futures):
                            meshes.append(f.result())
                            pbar.update(1)
                return meshes

    bodyid = utils.parse_bid(bodyid)

    voxels = get_sparsevol(bodyid,
                           scale=scale,
                           ret_type='INDEX',
                           save_to=None,
                           bbox=bbox,
                           server=server,
                           node=node)

    defaults = dict(chunk_size=200 if scale in (0, 1, 2, 3, 4) else None,
                    merge_fragments=True,
                    pad_chunks=True)
    defaults.update(kwargs)

    # Get voxel sizes based on scale
    info = get_segmentation_info(server, node)['Extended']

    vsize = {'COARSE': [s * 8 for s in info['BlockSize']]}
    vsize.update({i: np.array(info['VoxelSize']) * 2**i for i in range(info['MaxDownresLevel'])})

    mesh = meshing.mesh_from_voxels(voxels,
                                    spacing=vsize[scale],
                                    step_size=step_size,
                                    **defaults)

    # Track the ID just in case
    mesh.id = bodyid

    return mesh


@lru_cache(None)
def get_segmentation_info(server=None, node=None):
    """Return segmentation info as dictionary (cached).

    Parameters
    ----------
    server :    str, optional
                If not provided, will try reading from global.
    node :      str, optional
                If not provided, will try reading from global.

    """
    server, node, user = eval_param(server, node)

    r = dvid_session().get(urllib.parse.urljoin(server, 'api/node/{}/{}/info'.format(node, config.segmentation)))

    return r.json()


def get_n_synapses(bodyid, server=None, node=None):
    """Return number of pre- and postsynapses associated with given body.

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

    r = dvid_session().get(urllib.parse.urljoin(server, 'api/node/{}/{}_labelsz/count/{}/PreSyn'.format(node,
                                                                                                        config.synapses,
                                                                                                        bodyid)))
    r.raise_for_status()
    pre = r.json()

    r = dvid_session().get(urllib.parse.urljoin(server, 'api/node/{}/{}_labelsz/count/{}/PostSyn'.format(node,
                                                                                                         config.synapses,
                                                                                                         bodyid)))
    r.raise_for_status()
    post = r.json()

    return {'pre': pre.get('PreSyn', None), 'post': post.get('PostSyn', None)}


def get_synapses(bodyid, pos_filter=None, with_details=False, server=None, node=None):
    """Return table of pre- and postsynapses associated with given body.

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

    r = dvid_session().get(urllib.parse.urljoin(server, 'api/node/{}/{}/label/{}?relationships={}'.format(node,
                                                                                                          config.synapses, bodyid, str(with_details).lower())))

    syn = r.json()

    if pos_filter:
        # Get filter
        filtered = pos_filter(np.array([s['Pos'] for s in syn]))

        if not any(filtered):
            raise ValueError('No synapses left after filtering.')

        syn = np.array(syn)[filtered]

    return pd.DataFrame.from_records(syn)


def get_connections(source, target, pos_filter=None, server=None, node=None):
    """Return list of connections between source(s) and target(s).

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

    if len(source) <= len(target):
        to_query = source
        query_rel = 'PreSyn'
    else:
        to_query = target
        query_rel = 'PostSyn'

    cn_data = []
    for q in to_query:
        r = dvid_session().get(urllib.parse.urljoin(server, 'api/node/{}/{}/label/{}?relationships=true'.format(node, config.synapses, q)))

        # Raise
        r.raise_for_status()

        # Extract synapses
        syn = r.json()

        if not syn:
            continue

        # Find arbitrary properties
        props = list(set([k for s in syn for k in s['Prop'].keys()]))

        # Collect downstream connections
        this_cn = [[s['Pos'], r['To']] + [s['Prop'].get(p, None) for p in props]
                   for s in syn if s['Kind'] == query_rel and s['Rels'] for r in s['Rels']]

        df = pd.DataFrame(this_cn)

        # Add columns
        if query_rel == 'PreSyn':
            df.columns = ['tbar_position', 'psd_position'] + props
            # If we queried sources, we now the identity of presynaptic neuron
            df['bodyid_pre'] = q
        else:
            df.columns = ['psd_position', 'tbar_position'] + props

        cn_data.append(df)

    cn_data = pd.concat(cn_data, axis=0, sort=True)

    if pos_filter:
        # Get filter
        filtered = pos_filter(np.vstack(cn_data.tbar_position.values))

        if not any(filtered):
            raise ValueError('No synapses left after filtering.')

        # Filter synapses
        cn_data = cn_data.loc[filtered, :]

    # Add body positions
    if 'bodyid_pre' not in cn_data.columns:
        # Get positions of PSDs
        pos = np.vstack(cn_data.tbar_position.values)

        # Get postsynaptic body IDs
        bodies = locs_to_ids(pos, server=server, node=node)
        cn_data['bodyid_pre'] = bodies

        # Filter to sources of interest
        cn_data = cn_data[cn_data.bodyid_pre.isin(source)]

    if 'bodyid_post' not in cn_data.columns:
        # Get positions of PSDs
        pos = np.vstack(cn_data.psd_position.values)

        # Get presynaptic body IDs
        bodies = locs_to_ids(pos, server=server, node=node)
        cn_data['bodyid_post'] = bodies

        # Filter to targets of interest
        cn_data = cn_data[cn_data.bodyid_post.isin(target)]

    return cn_data


def get_connectivity(bodyid, pos_filter=None, ignore_autapses=True,
                     server=None, node=None):
    """Return connectivity table for given body.

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
            this_r = [d[d.relation == r].set_index('bodyid').drop('relation', axis=1) for d in cn]
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
    r = dvid_session().get(urllib.parse.urljoin(server, 'api/node/{}/{}/label/{}?relationships=true'.format(node, config.synapses, bodyid)))

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
    bodies = locs_to_ids(pos, server=server, node=node)

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
        to_drop = cn_table.index[cn_table.bodyid == int(bodyid)]
        cn_table = cn_table.drop(index=to_drop).reset_index()

    return cn_table[['bodyid', 'relation', 'n_synapses']]


def get_adjacency(sources, targets=None, pos_filter=None, ignore_autapses=True,
                  server=None, node=None):
    """Get adjacency between sources and targets.

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
    cn = cn[cn.relation == relation].set_index('bodyid')
    cn.index = cn.index.astype(str)
    cn = cn.reindex(index=index, columns=columns, fill_value=0)

    if to_transpose:
        cn = cn.T

    return cn


def snap_to_body(bodyid, positions, server=None, node=None):
    """Snap a set of positions to the closest voxels on a given body.

    Parameters
    ----------
    bodyid :    body ID
                Body for which to find positions.
    positions : array-like
                List/Array of (x, y, z) raw(!) coordinates.
    server :    str, optional
                If not provided, will try reading from global.
    node :      str, optional
                If not provided, will try reading from global.

    Returns
    -------
    (x, y, z)

    """
    # Parse body ID
    bodyid = utils.parse_bid(bodyid)

    if isinstance(positions, pd.DataFrame):
        positions = positions[['x', 'y', 'z']].values
    elif not isinstance(positions, np.ndarray):
        positions = np.array(positions)

    positions = positions.astype(int)

    # Find those that are not already within the body
    bids = locs_to_ids(positions, server=server, node=node)
    mask = np.array(bids) != int(bodyid)
    to_snap = positions[mask]

    # First get voxels of the coarse neuron
    voxels = get_sparsevol(bodyid, scale='coarse', ret_type='INDEX',
                           server=server, node=node)

    # Get voxel sizes based on scale
    info = get_segmentation_info(server, node)['Extended']
    voxels = voxels * info['BlockSize']

    # For each position find a corresponding coarse voxel
    dist = cdist(to_snap, voxels)
    closest = voxels[np.argmin(dist, axis=1)]

    # Now query the more precise mesh for these coarse voxels
    snapped = []
    for v in tqdm(closest, leave=False, desc='Snapping'):
        # Generate a bounding bbox to only fetch the voxels we actually need
        bbox = np.vstack([v, v]).T
        bbox[:, 1] += info['BlockSize']

        fine = get_sparsevol(bodyid, scale=0, ret_type='INDEX',
                             bbox=bbox.ravel(),
                             server=server, node=node)

        dist = cdist([v], fine)

        snapped.append(fine[np.argmin(dist, axis=1)][0])

    positions[mask] = np.vstack(snapped)

    return positions


def get_last_mod(bodyid, server=None, node=None):
    """Fetch details on the last modification to given body.

    Parameters
    ----------
    bodyid :    body ID
                Body for which to find positions.
    server :    str, optional
                If not provided, will try reading from global.
    node :      str, optional
                If not provided, will try reading from global.

    Returns
    -------
    dict
                {'mutation id': int,
                 'last mod user': str,
                 'last mod app': str,
                 'last mod time': timestamp isoformat str}

    """
    # Parse body ID
    bodyid = utils.parse_bid(bodyid)

    server, node, user = eval_param(server, node)

    r = dvid_session().get(urllib.parse.urljoin(server, 'api/node/{}/{}/lastmod/{}'.format(node,
                                                                                           config.segmentation,
                                                                                           bodyid)))
    r.raise_for_status()

    return r.json()


def get_skeleton_mutation(bodyid, server=None, node=None):
    """Fetch mutation ID of given body.

    Works by downloading the SWC file and parsing only the header.

    Parameters
    ----------
    bodyid :        int | str
                    ID(s) of body for which to download skeleton.
    server :        str, optional
                    If not provided, will try reading from global.
    node :          str, optional
                    If not provided, will try reading from global.

    Returns
    -------
    int
                    Mutation ID. Returns ``None`` if no skeleton available.

    """
    if isinstance(bodyid, (list, np.ndarray)):
        resp = {x: get_skeleton_mutation(x,
                                         server=server,
                                         node=node) for x in tqdm(bodyid,
                                                                  desc='Loading')}
        return resp

    def split_iter(string):
        # Returns iterator for line split -> memory efficient
        # Attention: this will not fetch the last line!
        return (x.group(0) for x in re.finditer('.*?\n', string))

    bodyid = utils.parse_bid(bodyid)

    server, node, user = eval_param(server, node)

    r = dvid_session().get(urllib.parse.urljoin(server, 'api/node/{}/{}_skeletons/key/{}_swc'.format(node,
                                                                                                     config.segmentation,
                                                                                                     bodyid)))

    if 'not found' in r.text:
        print(r.text)
        return None

    # Extract header using a generator -> this way we don't have to iterate
    # over all lines
    lines = split_iter(r.text)
    header = []
    for l in lines:
        if l.startswith('#'):
            header.append(l)
        else:
            break
    # Turn header back into string
    header = '\n'.join(header)

    if 'mutation id' not in header:
        print('{} - Unable to check mutation: mutation ID not in SWC header'.format(bodyid))
    else:
        swc_mut = re.search('"mutation id": (.*?)}', header).group(1)
        return int(swc_mut)


def list_projects(server=None):
    """List available projects on the server.

    Parameters
    ----------
    server :    str
                If not provided will fall back to globally defined server.

    Returns
    -------
    pandas.DataFrame

    """
    server, _, _ = eval_param(server)

    url = utils.make_url(server, 'api/repos/info')

    r = dvid_session().get(url)

    r.raise_for_status()

    return pd.DataFrame.from_records(list(r.json().values()))


def get_master_node(id, server=None):
    """Get UUID of the current master node.

    Parameters
    ----------
    id :            str
                    UUID of a node or a project you want the current master
                    node for. You can get use `list_projects` to get the root
                    ID for a project and then use this function to get the
                    master.
    server :        str, optional
                    If not provided will fall back to globally defined server.

    Returns
    -------
    master :        str
                    ID of master node.

    """
    server, _, _ = eval_param(server)

    url = utils.make_url(server, f'api/repo/{id}/branch-versions/master')

    r = dvid_session().get(url)

    try:
        r.raise_for_status()
    except HTTPError:
        raise HTTPError(f'{r.status_code} {r.reason}: {r.text}')
    except BaseException:
        raise

    return r.json()[0]

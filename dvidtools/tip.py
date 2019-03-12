# This code is part of dvid-tools (https://github.com/flyconnectome/dvid_tools)
# and is released under GNU GPL3

from . import utils
from . import fetch

import datetime as dt
import numpy as np
import pandas as pd

from collections import OrderedDict
from scipy.spatial.distance import cdist, pdist, squareform
from tqdm import tqdm


def detect_tips(x, psd_dist=10, done_dist=50, checked_dist=50, tip_dist=50,
                pos_filter=None, save_to=None, verbose=True, snap=True,
                server=None, node=None):
    """ Detects potential open ends on a given neuron.

    In brief, the workflow is as follows:
      1. Extract tips from the neuron's skeleton
      2. Remove duplicate tips (see ``tip_dist``)
      3. Snap tip positions back to mesh (see ``snap``)
      4. Remove tips in vicinity of PSDs (see ``psd_dist``)
      5. Remove tips close to DONE tags (see ``done_dist``)
      6. Remove tips close to a previously checked assignment
         (see ``checked_dist``)
      7. Sort tips by radius and return

    Parameters
    ----------
    x :             single body ID
    psd_dist :      int | None, optional
                    Minimum distance (in raw units) to a postsynaptic density
                    (PSD) for a tip to be considered "done".
    done_dist :     int | None,
                    Minimum distance (in raw units) to a DONE tag for a tip
                    to be considered "done".
    checked_dist :  int | None, optional
                    Minimum distance (in raw units) to a bookmark that has
                    previously been "Set Checked" in the "Assigned bookmarks"
                    table in Neutu.
    tip_dist :      int | None, optional
                    If a given pair of tips is closer than this distance they
                    will be considered duplicates and one of them will be
                    dropped.
    pos_filter :    function, optional
                    Function to tips by position. Must accept
                    numpy array (N, 3) and return array of [True, False, ...]
    save_to :       filepath, optional
                    If provided will save open ends to JSON file that can be
                    imported as assigmnents.
    snap :          bool, optional
                    If True, will make sure that tips positions are within the
                    mesh.
    server :        str, optional
                    If not provided, will try reading from global.
    node :          str, optional
                    If not provided, will try reading from global.

    Return
    ------
    pandas.DataFrame
                    List of potential open ends.

    Examples
    --------
    >>> import dvidtools as dt
    >>> dt.set_param('http://your.server.com:8000', 'node', 'user')
    >>> # Generate list of tips and save to json file
    >>> tips = dt.detect_tips(883338122, save_to='~/Documents/883338122.json')
    """

    # TODOs:
    # - add some sort of confidence based on (a) distance to next PSD and
    #   radius --> use our ground truth to calibrate
    # - add examples
    # - use tortuosity?

    # Get the skeleton
    n = fetch.get_skeleton(x, save_to=None, server=server, node=node)

    # Turn into DataFrame
    n, header = utils.parse_swc_str(n)

    # Find leaf and root nodes
    leafs = n[(~n.node_id.isin(n.parent_id.values)) | (n.parent_id <= 0)]

    # Remove potential duplicated leafs
    if tip_dist:
        # Get all by all distance
        dist = squareform(pdist(leafs[['x', 'y', 'z']].values))
        # Set upper triangle (including self dist) to infinite so that we only
        # get (A->B) and not (B->A) distances
        dist[np.triu_indices(dist.shape[0])] = float('inf')
        # Extract those that are too close
        too_close = list(set(np.where(dist < tip_dist)[0]))
        # Drop 'em
        leafs = leafs.reset_index().drop(too_close, axis=0).reset_index()

    # Skeletons can end up outside the body's voxels - let's snap 'em back
    if snap:
        leafs.loc[:, ['x', 'y', 'z']] = fetch.snap_to_body(x,
                                                           leafs[['x', 'y', 'z']].values,
                                                           server=server,
                                                           node=node)

    if pos_filter:
        # Get filter
        filtered = pos_filter(leafs[['x','y','z']].values)

        if not any(filtered):
            raise ValueError('No tips left after filtering!')

        leafs = leafs.loc[filtered, :]

    n_leafs = leafs.shape[0]

    if psd_dist:
        # Get synapses
        syn = fetch.get_synapses(x, pos_filter=None, with_details=False,
                                 server=server, node=node)
        post = syn[syn.Kind=='PostSyn']

        # Get distances
        dist = cdist(leafs[['x', 'y', 'z']].values,
                     np.vstack(post.Pos.values))

        # Is tip close to PSD?
        at_psd = np.min(dist, axis=1) < psd_dist

        leafs = leafs[~at_psd]

    psd_filtered = n_leafs - leafs.shape[0]

    if done_dist:
        # Check for DONE tags in vicinity
        at_done = []
        for pos in tqdm(leafs[['x', 'y', 'z']].values,
                        desc='Check DONE', leave=False):
            # We are cheating here b/c we don't actually calculate the
            # distance!
            labels = fetch.get_labels_in_area(pos - done_dist/2,
                                              [done_dist] * 3,
                                              server=server, node=node)

            if isinstance(labels, type(None)):
                at_done.append(False)
                continue

            # DONE tags have no "action" and "checked" = 1
            if any([p.get('checked', False) and not p.get('action', False) for p in labels.Prop.values]):
                at_done.append(True)
            else:
                at_done.append(False)

        leafs = leafs[~np.array(at_done, dtype=bool)]

    done_filtered = n_leafs - leafs.shape[0]

    if checked_dist:
        # Check if position has been "Set Checked" in the past
        checked = []
        for pos in tqdm(leafs[['x', 'y', 'z']].values,
                        desc='Test Checked', leave=False):
            # We will look for the assigment in a small window in case the
            # tip has moved slightly between iterations
            ass = fetch.get_assignment_status(pos, window=[checked_dist]*3,
                                              bodyid=x, server=server,
                                              node=node)

            if any([l.get('checked', False) for l in ass]):
                checked.append(True)
            else:
                checked.append(False)

        leafs = leafs[~np.array(checked, dtype=bool)]

    checked_filtered = n_leafs - leafs.shape[0]

    # Make a copy before we wrap up to prevent any data-on-copy warning
    leafs = leafs.copy()

    # Assuming larger radii indicate more likely continuations
    leafs.sort_values('radius', ascending=False, inplace=True)

    if verbose:
        d = OrderedDict({
                        'Total tips': n_leafs,
                        'PSD filtered': psd_filtered,
                        'Done tag filtered': done_filtered,
                        'Checked assignment filtered': checked_filtered,
                        'Tips left': leafs.shape[0],
                       })
        print(pd.DataFrame.from_dict(d, orient='index', columns=[x]))

    if save_to:
        leafs['body ID'] = x
        leafs['text'] = ''
        meta = {'description': 'Generated by dvidtools.detect_tips',
                'date': dt.date.today().isoformat(),
                'url': 'https://github.com/flyconnectome/dvid_tools',
                'parameters' : {'psd_dist': psd_dist,
                                'done_dost': done_dist,
                                'checked_dist': checked_dist,
                                'snap': snap,
                                'tip_dist': tip_dist,
                                'node': node,
                                'server': server}}
        _ = utils.gen_assignments(leafs, save_to=save_to, meta=meta)

    return leafs.reset_index(drop=True)





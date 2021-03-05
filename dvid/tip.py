# This code is part of dvid-tools (https://github.com/flyconnectome/dvid_tools)
# and is released under GNU GPL3

from . import utils
from . import fetch

import datetime as dt
import networkx as nx
import numpy as np
import os
import pandas as pd
import warnings

from collections import OrderedDict
from scipy.spatial.distance import cdist, pdist, squareform
from tqdm import tqdm


__all__ = ['detect_tips']


def detect_tips(x, use_clf=False, psd_dist=False, done_dist=False,
                checked_dist=50, tip_dist=False, pos_filter=None,
                save_to=None, verbose=True, snap=True, server=None,
                node=None):
    """Detect potential open ends on a given neuron.

    In brief, the workflow is as follows:
      1. Extract tips from the neuron's skeleton
      2. Snap tip positions back to mesh (see ``snap``)
      3. Apply filters (see ``_dist`` parameters)
      4. Remove tips close to a previously checked assignment
         (see ``checked_dist``)
      5. (Optionally) Try to prioritize tips (``use_clf``)
      6. Save to a json file (see ``save_to``) that can be opened in neuTu

    Parameters
    ----------
    x :             single body ID
    use_clf :       bool, optional
                    If True, will use a pre-trained classifier to try to
                    predict whether a tip needs human interaction or not. The
                    returned list of tips will contain and be ordered by
                    "confidence" values from -1 (unimportant) to +1
                    (important). THIS IS NOT A MAGIC BULLET! If you need to be
                    certain you have completed a neuron, you will actually
                    have to look at all tips regardless of confidence!
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
                    Function to fitler tips by position. Must accept
                    numpy array (N, 3) and return array of [True, False, ...].
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
    The usual setting up:


    >>> import dvidtools as dt
    >>> dt.set_param('https://your.server.com:8000', 'node', 'user')

    By default, ``detect_tips`` will simply return all end nodes of the
    skeleton that have not previously been "Set checked" in NeuTu:

    >>> # Generate list of tips and save to json file
    >>> tips = dt.detect_tips(883338122, save_to='~/Documents/883338122.json')

    A more advanced example is using a pre-trained classifier to include
    "confidence" values. These confidence range from -1 to +1 and give some
    indication whether a tip needs to be extended or not. This requires
    `sciki-learn <https://scikit-learn.org>`_ to be installed. In a terminal
    run::

        pip install scikit-learn

    Once scikit-learn is installed, you can run the tip detector with
    classifier confidences:

    >>> tips = dt.detect_tips(883338122, use_clf=True,
    ...                      save_to='~/Documents/883338122.json')

    """
    server, node, user = fetch.eval_param(server, node)

    # Get the skeleton
    n = fetch.get_skeletons(x, save_to=None, server=server, node=node)[0]

    if isinstance(n, type(None)):
        raise ValueError('{} appears to not have a skeleton. Please double'
                         ' check the body ID.')

    # Find leaf and root nodes
    leafs = n[(~n.node_id.isin(n.parent_id.values)) | (n.parent_id <= 0)].copy()

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
        filtered = pos_filter(leafs[['x', 'y', 'z']].values)

        if not any(filtered):
            raise ValueError('No tips left after filtering!')

        leafs = leafs.loc[filtered, :]

    n_leafs = leafs.shape[0]

    if psd_dist:
        # Get synapses
        syn = fetch.get_synapses(x, pos_filter=None, with_details=False,
                                 server=server, node=node)
        post = syn[syn.Kind == 'PostSyn']

        # Get distances
        dist = cdist(leafs[['x', 'y', 'z']].values,
                     np.vstack(post.Pos.values))

        # Is tip close to PSD?
        at_psd = np.min(dist, axis=1) < psd_dist

        leafs = leafs.loc[~at_psd]

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

        leafs = leafs.loc[~np.array(at_done, dtype=bool)]

    done_filtered = n_leafs - leafs.shape[0]

    if checked_dist:
        # Check if position has been "Set Checked" in the past
        checked = []
        for pos in tqdm(leafs[['x', 'y', 'z']].values,
                        desc='Test Checked', leave=False):
            # We will look for the assigment in a small window in case the
            # tip has moved slightly between iterations
            ass = fetch.get_assignment_status(pos, window=[checked_dist] * 3,
                                              bodyid=x if snap else None,
                                              server=server,
                                              node=node)

            if any([l.get('checked', False) for l in ass]):
                checked.append(True)
            else:
                checked.append(False)

        leafs = leafs.loc[~np.array(checked, dtype=bool)]

    checked_filtered = n_leafs - leafs.shape[0]

    # Make a copy before we wrap up to prevent any data-on-copy warning
    leafs = leafs.copy()

    # Assuming larger radii indicate more likely continuations
    leafs.sort_values('radius', ascending=False, inplace=True)

    if use_clf:
        try:
            import sklearn
            from sklearn.externals import joblib
        except ImportError:
            print('Must have scikit-learn (https://scikit-learn.org) library '
                  'installed. Skipping classification.')
            sklearn = None
        except BaseException:
            raise

    if use_clf and sklearn:
        fp = os.path.dirname(__file__)
        model_file = os.path.join(fp, 'model/model.pkl')
        scaler_file = os.path.join(fp, 'model/scaler.pkl')

        try:
            clf = joblib.load(model_file)
            scaler = joblib.load(scaler_file)
        except BaseException:
            print('Unable to load classifier model. Skipping classification.')
            clf = None

    if use_clf and clf:
        # Get synapse data
        syn = fetch.get_synapses(x, pos_filter=None, with_details=False)

        features = _generate_features(n, leafs, syn)

        # Fit Data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            features = scaler.fit_transform(features.values)

        # Predict
        probability = clf.predict_proba(features)
        confidence = np.round((probability[:, 1] - probability[:, 0]) /
                              (probability[:, 1] + probability[:, 0]), 3)

        # Add confidence
        leafs['confidence'] = confidence
        leafs['text'] = leafs.confidence.astype(str)
        leafs['comment'] = leafs['text']

        # Sort by confidence
        leafs.sort_values('confidence', inplace=True, ascending=False),

    else:
        leafs['text'] = ''

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
        meta = {'description': 'Generated by dvidtools.detect_tips',
                'date': dt.date.today().isoformat(),
                'url': 'https://github.com/flyconnectome/dvid_tools',
                'parameters': {'psd_dist': psd_dist,
                               'done_dist': done_dist,
                               'checked_dist': checked_dist,
                               'snap': snap,
                               'tip_dist': tip_dist,
                               'node': node,
                               'use_clf': use_clf,
                               'pos_filter': not isinstance(pos_filter, (type(None), bool)),
                               'swc_mutation_id':  getattr(n, 'mutation_id', 'NA'),
                               'server': server}}
        _ = utils.gen_assignments(leafs, save_to=save_to, meta=meta)

    return leafs.reset_index(drop=True)


def _generate_features(swc, tips, syn):
    """Generate features."""
    # Generate graph
    g = utils.swc_to_graph(swc)

    # Collect features
    f = pd.DataFrame([])
    f['surround_radius'] = tips.node_id.map(lambda x: _surround_radius(x, g))
    f['parent_radius'] = tips.node_id.map(lambda x: g.nodes[_first_branch_point(x, g)]['radius'])
    f['parent_vector'] = tips.node_id.map(lambda x: _dot_product(x, g))
    f['hk_dist'] = [_dist_to_hk(x) for x in tips[['x', 'y', 'z']].values]
    f['branch_length'] = tips.node_id.map(lambda x: _branch_length(x, g))
    f['tortuosity'] = tips.node_id.map(lambda x: _tortuosity(x, g))
    f['psd_dist'] = _dist_to_psd(tips, syn)
    f['tbar_dist'] = _dist_to_tbar(tips, syn)
    f['tip_dist'] = _dist_to_tip(tips)
    f['io_ratio'] = _in_out_ratio(tips, syn)

    # Fill NaNs
    f.fillna(0, inplace=True)

    return f


def _first_branch_point(x, g):
    """Return first branch point upstream of x."""
    while True:
        if g.in_degree(x) > 1 or g.out_degree(x) == 0:
            return x
        x = next(g.successors(x))


def _branch_length(x, g):
    """Return length of path from node x to the first branch point."""
    return nx.shortest_path_length(g, x, _first_branch_point(x, g), weight='weight')


def _tortuosity(x, g):
    """Return tortuosity for path from node x to the first branch point."""
    L = _branch_length(x, g)
    R = np.sqrt(np.sum((g.nodes[x]['location'] - g.nodes[_first_branch_point(x, g)]['location'])**2))

    # If no subsequent prior branch point return 0
    if R:
        return L / R
    else:
        return 0


def _dist_to_hk(x):
    """Return distance to closest hot knife section. """
    # x positions of hot knife sections
    hk = np.array([5249, 7920, 10599, 13229, 15894, 18489, 21204, 24040, 26854, 29743, 32359])
    hk = hk.reshape((len(hk), 1))

    d = cdist(np.reshape(x[0], (1, 1)), hk)

    return np.min(d)


def _dot_product(x, g, normal=True):
    """Return dot product of (normalised) vector of x and the first downstream branch point. """
    if g.out_degree(x) == 0:
        return None

    v1 = np.array(g.nodes[x]['location']) - np.array(g.nodes[next(g.successors(x))]['location'])

    bp = _first_branch_point(x, g)

    if g.out_degree(bp) == 0:
        bpp = next(g.predecessors(bp))
    else:
        bpp = next(g.successors(bp))

    v2 = np.array(g.nodes[bp]['location']) - np.array(g.nodes[bpp]['location'])

    if normal:
        v1 /= np.linalg.norm(v1)
        v2 /= np.linalg.norm(v2)

    return np.dot(v1, v2)


def _surround_radius(x, g, n=5):
    """Return mean radius in the surrounding of a node."""
    r = []
    while g.out_degree(x) and len(r) < n:
        r.append(g.nodes[x]['radius'])
        x = next(g.successors(x))

    if r:
        return np.mean(r)
    else:
        return g.nodes[x]['radius']


def _dist_to_psd(swc, syn):
    """Return distance to next PSD for all rows in a SWC table."""
    post = syn[syn.Kind == 'PostSyn']

    dist = cdist(swc[['x', 'y', 'z']].values,
                 np.vstack(post.Pos.values))

    return np.min(dist, axis=1)


def _dist_to_tbar(swc, syn):
    """Return distance to next presynapse for all rows in a SWC table."""
    tbar = syn[syn.Kind == 'PreSyn']

    dist = cdist(swc[['x', 'y', 'z']].values,
                 np.vstack(tbar.Pos.values))

    return np.min(dist, axis=1)


def _dist_to_tip(tips):
    """Return distance to next closest tip for all rows in tips table."""
    dist = squareform(pdist(tips[['x', 'y', 'z']].values))
    dist[dist == 0] = float('inf')

    return np.min(dist, axis=1)


def _in_out_ratio(swc, syn, r=1000):
    """Return ratio of PSDs vs presynapses within given radius."""
    post = syn[syn.Kind == 'PostSyn']
    pre = syn[syn.Kind == 'PreSyn']

    post_dist = cdist(swc[['x', 'y', 'z']].values,
                      np.vstack(post.Pos.values))
    pre_dist = cdist(swc[['x', 'y', 'z']].values,
                     np.vstack(pre.Pos.values))

    n_post = np.sum(post_dist <= r, axis=1)
    n_pre = np.sum(pre_dist <= r, axis=1)
    # Make sure there are no zeros
    total = n_pre + n_post
    total[total == 0] = 1

    # Return ratio
    return (n_post-n_pre) / total

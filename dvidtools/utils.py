# This code is part of dvid-tools (https://github.com/flyconnectome/dvid_tools)
# and is released under GNU GPL3

import json
import pandas as pd

from io import StringIO


def verify_payload(data, required, required_only=True):
    """ Verifies payload.

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
    """ Parse SWC string into a pandas DataFrame.

    Parameters
    ----------
    x :     str

    Returns
    -------
    pandas.DataFrame, header

    """

    if not isinstance(x, str):
        raise TypeError('x must be str, got "{}"'.format(type(x)))

    # Extract header
    header = [l for l in x.split('\n') if l.startswith('#')]

    # Turn header back into string
    header = '\n'.join(header)

    # Turn SWC into a DataFrame
    f = StringIO(x)
    df = pd.read_csv(f, delim_whitespace=True, header=None, comment='#')

    df.columns = ['node_id', 'label', 'x', 'y', 'z', 'radius', 'parent_id']

    return df, header


def gen_assignments(x, save_to=None, meta={}):
    """ Generates JSON file that can be imported into neutu as assignments.

    Parameters
    ----------
    x :         pandas.DataFrame
                Must contain columns ``x``, ``y``, ``z`` or ``location``.
                Optional columns: ``body ID`` and ``text``.
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
        if any([c not in  x.columns for c in ['x', 'y', 'z']]):
            raise ValueError('x must have "location" column or "x", "y" '
                             'and "z" columns')
        x['location'] = x[['x','y','z']].astype(int).apply(list, axis=1)

    for c in ['body ID', 'text']:
        if c not in x.columns:
            x[c] = ''

    x = x[['location', 'text', 'body ID']]

    j = {'metadata': meta,
         'data': x.to_dict(orient='records')}

    if save_to:
        with open(save_to, 'w') as f:
            json.dump(j, f, indent=2)
    else:
        return j


def parse_bid(x):
    try:
        return int(x)
    except:
        raise ValueError('Unable to coerce "{}" into numeric body ID'.format(x))


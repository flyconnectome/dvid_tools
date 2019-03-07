# This code is part of dvid-tools (https://github.com/flyconnectome/dvid_tools)
# and is released under GNU GPL3

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

    try:
        return int(x)
    except:
        raise ValueError('Unable to coerce "{}" into numeric body ID'.format(x))


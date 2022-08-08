import os
import urllib
import functools

SERVER=os.environ['DVID_TEST_SERVER']
NODE=os.environ['DVID_TEST_NODE']

SERVER = urllib.parse.urlparse(SERVER).netloc


def redact_server(func):
    """Redact server and node from exceptions."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except BaseException as e:
            args = list(e.args)
            args[0] = args[0].replace(SERVER, 'DVID_TEST_SERVER')
            args[0] = args[0].replace(NODE, 'DVID_TEST_NODE')
            e.args = args
            raise e

    return wrapper

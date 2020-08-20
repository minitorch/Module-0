import numpy as np


def wrap_tuple(x):
    if isinstance(x, tuple):
        return x
    return (x,)


def unwrap_tuple(x):
    if len(x) == 1:
        return x[0]
    return x


def assert_close(a, b):
    np.testing.assert_allclose(a, b, 1e-2, 1e-2)


class IndexingError(Exception):
    "Exception raised for indexing errors."
    pass

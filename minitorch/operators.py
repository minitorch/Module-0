"""
Collection of the core mathematical operators used throughout the code base.
"""


import math
import numpy as np

# ## Task 0.1

# Implementation of a prelude of elementary functions.


def mul(x, y):
    ":math:`f(x, y) = x * y`"
    # TODO: Implement for Task 0.1.
    return np.multiply(x, y)


def id(x):
    ":math:`f(x) = x`"
    # TODO: Implement for Task 0.1.
    return x


def add(x, y):
    ":math:`f(x, y) = x + y`"
    # TODO: Implement for Task 0.1.
    return np.add(x, y)


def neg(x):
    ":math:`f(x) = -x`"
    # TODO: Implement for Task 0.1.
    return -x


def lt(x, y):
    ":math:`f(x) =` 1.0 if x is less than y else 0.0"
    # TODO: Implement for Task 0.1.
    return np.multiply(1.0, np.less(x, y))


def eq(x, y):
    ":math:`f(x) =` 1.0 if x is equal to y else 0.0"
    # TODO: Implement for Task 0.1.
    return np.multiply(1.0, np.equal(x, y))


def max(x, y):
    ":math:`f(x) =` x if x is greater than y else y"
    # TODO: Implement for Task 0.1.
    return np.maximum(x, y)


def is_close(x, y):
    ":math:`f(x) = |x - y| < 1e-2` "
    # TODO: Implement for Task 0.1.
    return np.less(np.abs(np.subtract(x, y)), 1e-2)


def sigmoid(x):
    r"""
    :math:`f(x) =  \frac{1.0}{(1.0 + e^{-x})}`

    (See `<https://en.wikipedia.org/wiki/Sigmoid_function>`_ .)

    Calculate as

    :math:`f(x) =  \frac{1.0}{(1.0 + e^{-x})}` if x >=0 else :math:`\frac{e^x}{(1.0 + e^{x})}`

    for stability.

    Args:
        x (float): input

    Returns:
        float : sigmoid value
    """
    # TODO: Implement for Task 0.1.
    return np.divide(1.0, 1.0 + np.exp(np.negative(x))) if x >= 0 else np.divide(np.exp(x), 1.0 + np.exp(x))


def relu(x):
    """
    :math:`f(x) =` x if x is greater than 0, else 0

    (See `<https://en.wikipedia.org/wiki/Rectifier_(neural_networks)>`_ .)

    Args:
        x (float): input

    Returns:
        float : relu value
    """
    # TODO: Implement for Task 0.1.
    return np.maximum(0, x)


EPS = 1e-6


def log(x):
    ":math:`f(x) = log(x)`"
    return np.log(x + EPS)


def exp(x):
    ":math:`f(x) = e^{x}`"
    return np.exp(x)


def log_back(x, d):
    r"If :math:`f = log` as above, compute d :math:`d \times f'(x)`"
    # TODO: Implement for Task 0.1.
    return np.divide(d, x + EPS)


def inv(x):
    ":math:`f(x) = 1/x`"
    # TODO: Implement for Task 0.1.
    return np.divide(1.0, x + EPS)


def inv_back(x, d):
    r"If :math:`f(x) = 1/x` compute d :math:`d \times f'(x)`"
    # TODO: Implement for Task 0.1.
    return np.divide(np.negative(d), np.multiply(x + EPS, x + EPS))


def relu_back(x, d):
    r"If :math:`f = relu` compute d :math:`d \times f'(x)`"
    # TODO: Implement for Task 0.1.
    return np.multiply(d, np.greater(x, 0.0))


# ## Task 0.3

# Small library of elementary higher-order functions for practice.


def map(fn):
    """
    Higher-order map.

    .. image:: figs/Ops/maplist.png


    See `<https://en.wikipedia.org/wiki/Map_(higher-order_function)>`_

    Args:
        fn (one-arg function): Function from one value to one value.

    Returns:
        function : A function that takes a list, applies `fn` to each element, and returns a
        new list
    """
    return lambda ls : [fn(x) for x in ls]


def negList(ls):
    "Use :func:`map` and :func:`neg` to negate each element in `ls`"
    return map(neg)(ls)


def zipWith(fn):
    """
    Higher-order zipwith (or map2).

    .. image:: figs/Ops/ziplist.png

    See `<https://en.wikipedia.org/wiki/Map_(higher-order_function)>`_

    Args:
        fn (two-arg function): combine two values

    Returns:
        function : takes two equally sized lists `ls1` and `ls2`, produce a new list by
        applying fn(x, y) on each pair of elements.

    """

    return lambda ls1, ls2 : [fn(ls1[i], ls2[i]) for i in range(len(ls1))]


def addLists(ls1, ls2):
    "Add the elements of `ls1` and `ls2` using :func:`zipWith` and :func:`add`"
    return zipWith(add)(ls1, ls2)


def reduce(fn, start):
    r"""
    Higher-order reduce.

    .. image:: figs/Ops/reducelist.png


    Args:
        fn (two-arg function): combine two values
        start (float): start value :math:`x_0`

    Returns:
        function : function that takes a list `ls` of elements
        :math:`x_1 \ldots x_n` and computes the reduction :math:`fn(x_3, fn(x_2,
        fn(x_1, x_0)))`
    """
    def _reduce(ls):
        x_prev = fn(start, ls[0])
        for i in range(1, len(ls)):
            x_prev = fn(x_prev, ls[i])
        return x_prev
    return _reduce


def sum(ls):
    "Sum up a list using :func:`reduce` and :func:`add`."
    # TODO: Implement for Task 0.3.
    return reduce(add, 0)(ls) if len(ls) else 0


def prod(ls):
    "Product of a list using :func:`reduce` and :func:`mul`."
    return reduce(mul, 1.0)(ls)

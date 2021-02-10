import math

## Task 0.1
## Mathematical operators


def mul(x, y):
    ":math:`f(x, y) = x * y`"
    # TODO: Implement for Task 0.1.
    return x * y
    raise NotImplementedError("Need to implement for Task 0.1")


def id(x):
    ":math:`f(x) = x`"
    # TODO: Implement for Task 0.1.
    return x
    raise NotImplementedError("Need to implement for Task 0.1")


def add(x, y):
    ":math:`f(x, y) = x + y`"
    # TODO: Implement for Task 0.1.
    return x + y
    raise NotImplementedError("Need to implement for Task 0.1")


def neg(x):
    ":math:`f(x) = -x`"
    # TODO: Implement for Task 0.1.
    return -x
    raise NotImplementedError("Need to implement for Task 0.1")


def lt(x, y):
    ":math:`f(x) = 1.0` if x is less than y else 0.0"
    # TODO: Implement for Task 0.1.
    if x < y:
        return 1.0
    else:
        return 0.0
    raise NotImplementedError("Need to implement for Task 0.1")


def eq(x, y):
    ":math:`f(x) =` 1.0 if x is equal to y else 0.0"
    # TODO: Implement for Task 0.1.
    if x == y:
        return 1.0
    else:
        return 0.0
    raise NotImplementedError("Need to implement for Task 0.1")


def max(x, y):
    ":math:`f(x) =` x if x is greater than y else y"
    # TODO: Implement for Task 0.1.
    if x > y:
        return x
    else:
        return y
    raise NotImplementedError("Need to implement for Task 0.1")


def sigmoid(x):
    """
    :math:`f(x) =  \frac{1.0}{(1.0 + e^{-x})}`

    (See `<https://en.wikipedia.org/wiki/Sigmoid_function>`_ .)

    Calculate as

    :math:`f(x) =  \frac{1.0}{(1.0 + e^{-x})}` if x >=0 else :math:`\frac{e^x}{(1.0 + e^{x})}`

    for stability.

    """
    # TODO: Implement for Task 0.1.
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(-x))
    raise NotImplementedError("Need to implement for Task 0.1")


def relu(x):
    """
    :math:`f(x) =` x if x is greater than 0, else 0

    (See `<https://en.wikipedia.org/wiki/Rectifier_(neural_networks)>`_ .)
    """
    # TODO: Implement for Task 0.1.
    if x > 0:
        return x
    else:
        return 0.0
    raise NotImplementedError("Need to implement for Task 0.1")


def relu_back(x, y):
    ":math:`f(x) =` y if x is greater than 0 else 0"
    # TODO: Implement for Task 0.1.
    if x > 0:
        return y
    else:
        return 0
    raise NotImplementedError("Need to implement for Task 0.1")


EPS = 1e-6


def log(x):
    ":math:`f(x) = log(x)`"
    return math.log(x + EPS)


def exp(x):
    ":math:`f(x) = e^{x}`"
    return math.exp(x)


def log_back(a, b):
    return b / (a + EPS)


def inv(x):
    ":math:`f(x) = 1/x`"
    return 1.0 / x


def inv_back(a, b):
    return -(1.0 / a ** 2) * b


## Task 0.3
## Higher-order functions.


def map(fn):
    """
    Higher-order map.

    .. image:: figs/Ops/maplist.png


    See `<https://en.wikipedia.org/wiki/Map_(higher-order_function)>`_

    Args:
        fn (one-arg function): process one value

    Returns:
        function : a function that takes a list and applies `fn` to each element
    """
    # TODO: Implement for Task 0.3.
    def inner_fun(ls):
        if len(ls) > 0:
            for i in range(len(ls)):
                ls[i] = fn(ls[i])
        return ls
    return inner_fun
    raise NotImplementedError("Need to implement for Task 0.3")


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
        applying fn(x, y) one each pair of elements.

    """
    # TODO: Implement for Task 0.3.
    def inner_fun(ls1, ls2):
        ls3 = []
        if len(ls1) == len(ls2) and len(ls1) > 0:
            for i in range(len(ls1)):
                ls3.append(fn(ls1[i], ls2[i]))
        return ls3
    return inner_fun
    raise NotImplementedError("Need to implement for Task 0.3")


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
    # TODO: Implement for Task 0.3.
    def inner_fun(ls):
        reduced_val = start
        if len(ls) > 0:
            for i in ls:
                reduced_val = fn(reduced_val, i)
        return reduced_val
    return inner_fun
    raise NotImplementedError("Need to implement for Task 0.3")


def sum(ls):
    """
    Sum up a list using :func:`reduce` and :func:`add`.
    """
    # TODO: Implement for Task 0.3.
    return reduce(add, 0)(ls)
    raise NotImplementedError("Need to implement for Task 0.3")


def prod(ls):
    """
    Product of a list using :func:`reduce` and :func:`mul`.
    """
    # TODO: Implement for Task 0.3.
    return reduce(mul, 1)(ls)
    raise NotImplementedError("Need to implement for Task 0.3")








"""
#import visdom
#import matplotlib.pyplot as plt
#vis = visdom.Visdom()
from project.datasets import Simple, Split, Xor
N = 100
def classify(pt):
    "Classify based on x position"
    if pt[0] <= 0.2 or pt[0] >= .8:
        return 1.0
    else:
        return 0.0
def classify2(pt):
    "Classify based on x position"
    if pt[0] <= 0.5  :
        return 1.0
    else:
        return 0.0
def classify3(pt):
    "Classify based on x position"
    if (pt[0] <= 0.55 and pt[1] >=.53) or (pt[0] >= 0.55 and pt[1] <=.53) :
        return 1.0
    if (pt[0] <= 0.55 and pt[1] <=.53) or (pt[0] >= 0.55 and pt[1] >=.53) :
        return 0.0
#Split(N, vis=True).graph("Split", model=classify)
#Simple(N, vis=True).graph("Simple",model=classify2)
Xor(N, vis=True).graph("XOR",model=classify3)
"""
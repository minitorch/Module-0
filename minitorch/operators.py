"""
Collection of the core mathematical operators used throughout the code base.
"""

import math
from typing import Callable, Iterable

# ## Task 0.1
#
# Implementation of a prelude of elementary functions.


def mul(x: float, y: float) -> float:
    """
    Multiplication.

    Args:
        x: A float value
        y: A float value

    Returns:
        The value x multiplied by y.
    """

    return x * y


def id(x: float) -> float:
    """
    Identity function.

    Args:
        x: A float value

    Returns:
        The value x.
    """
    return x


def add(x: float, y: float) -> float:
    """
    Addition.

    Args:
        x: A float value
        y: A float value

    Returns:
        The value x plus y.
    """

    return x + y


def neg(x: float) -> float:
    """
    Negation.

    Args:
        x: A float value

    Returns:
        The negative value of x.
    """

    return -x


def lt(x: float, y: float) -> float:
    """
    Less than.

    Args:
        x: A float value
        y: A float value

    Returns:
        1.0 if x is less than y else 0.0
    """
    if x < y:
        return 1.0
    else:
        return 0.0


def eq(x: float, y: float) -> float:
    "$f(x) =$ 1.0 if x is equal to y else 0.0"
    """
    Equal to.

    Args:
        x: A float value
        y: A float value

    Returns:
        1.0 if x is equal to y else 0.0
    """

    if x == y:
        return 1.0
    else:
        return 0.0


def max(x: float, y: float) -> float:
    """
    Maximum of two values.

    Args:
        x: A float value
        y: A float value

    Returns:
        The maximum of x and y.
    """

    if x > y:
        return x
    else:
        return y


def is_close(x: float, y: float) -> float:
    """
    Check if two values are close.

    Args:
        x: A float value
        y: A float value

    Returns:
        1.0 if |x - y| < 1e-2 else 0.0
    """

    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """
    Sigmoid function.

    Args:
        x: A float value

    Returns:
        The value of the sigmoid function at x.
    """

    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """
    Rectified Linear Unit function.

    Args:
        x: A float value

    Returns:
        The value of the ReLU function at x.
    """

    if x > 0:
        return x
    else:
        return 0.0


EPS = 1e-6


def log(x: float) -> float:
    "$f(x) = log(x)$"
    return math.log(x + EPS)


def exp(x: float) -> float:
    "$f(x) = e^{x}$"
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    """
    Logorithm backprop.

    Args:
        x: A float value
        d: A float value

    Returns:
        The value of the derivative of the log function at x times d.
    """

    return d / (x + EPS)


def inv(x: float) -> float:
    """
    Inverse function.

    Args:
        x: A float value

    Returns:
        The value of the inverse function at x.
    """

    return 1.0 / x


def inv_back(x: float, d: float) -> float:
    """
    Inverse backprop.

    Args:
        x: A float value
        d: A float value

    Returns:
        The value of the derivative of the inverse function at x times d.
    """

    return -d / (x * x)


def relu_back(x: float, d: float) -> float:
    """
    ReLU backprop.

    Args:
        x: A float value
        d: A float value

    Returns:
        The value of the derivative of the ReLU function at x times d.
    """

    if x > 0:
        return d
    else:
        return 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """
    Higher-order map.

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: Function from one value to one value.

    Returns:
        A function that takes a list, applies `fn` to each element, and returns a
         new list
    """
    # TODO: Implement for Task 0.3.
    raise NotImplementedError("Need to implement for Task 0.3")


def negList(ls: Iterable[float]) -> Iterable[float]:
    "Use `map` and `neg` to negate each element in `ls`"
    # TODO: Implement for Task 0.3.
    raise NotImplementedError("Need to implement for Task 0.3")


def zipWith(
    fn: Callable[[float, float], float]
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """
    Higher-order zipwith (or map2).

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: combine two values

    Returns:
        Function that takes two equally sized lists `ls1` and `ls2`, produce a new list by
         applying fn(x, y) on each pair of elements.

    """
    # TODO: Implement for Task 0.3.
    raise NotImplementedError("Need to implement for Task 0.3")


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    "Add the elements of `ls1` and `ls2` using `zipWith` and `add`"
    # TODO: Implement for Task 0.3.
    raise NotImplementedError("Need to implement for Task 0.3")


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    r"""
    Higher-order reduce.

    Args:
        fn: combine two values
        start: start value $x_0$

    Returns:
        Function that takes a list `ls` of elements
         $x_1 \ldots x_n$ and computes the reduction :math:`fn(x_3, fn(x_2,
         fn(x_1, x_0)))`
    """
    # TODO: Implement for Task 0.3.
    raise NotImplementedError("Need to implement for Task 0.3")


def sum(ls: Iterable[float]) -> float:
    "Sum up a list using `reduce` and `add`."
    # TODO: Implement for Task 0.3.
    raise NotImplementedError("Need to implement for Task 0.3")


def prod(ls: Iterable[float]) -> float:
    "Product of a list using `reduce` and `mul`."
    # TODO: Implement for Task 0.3.
    raise NotImplementedError("Need to implement for Task 0.3")

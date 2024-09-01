"""
Collection of the core mathematical operators used throughout the code base.
"""

import math

# ## Task 0.1
from typing import Callable, Iterable, Optional

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.
def mul(x: float, y: float) -> float:
    """
    Scalar multiplication.

    Args:
        x: A float value
        y: A float value

    Returns:
        A float value x multiplied by y
    """
    return x * y


def id(x: float) -> float:
    """
    Identity function.

    Args:
        x: A float value.

    Returns:
        The float input unchanged.
    """
    return x


def add(x: float, y: float) -> float:
    """
    Addition function.

    Args:
        x: A float value.
        y: A float value.

    Returns:
        A float value x added to y
    """
    return x + y


def neg(x: float) -> float:
    """
    Negation function.

    Args:
        x: A float value.

    Returns:
        A float value x multiplied by -1.0
    """
    return -x


def lt(x: float, y: float) -> bool:
    """
    Compares 2 float values.

    Args:
        x: A float value.
        y: A float value.

    Returns:
        A boolean value. True if x is less than y.
    """
    return x < y


def eq(x: float, y: float) -> bool:
    """
    Equality function.

    Args:
        x: A float value.
        y: A float value.

    Returns:
        A boolean value. True if x is equal to y.
    """
    return x == y


def max(x: float, y: float) -> float:
    """
    Max function.

    Args:
        x: A float value.
        y: A float value.

    Returns:
        The larger value between x and y.
    """
    return x if x > y else y


def is_close(
    x: float, y: float, atol: Optional[float] = 1e-8, rtol: Optional[float] = 1e-5
) -> bool:
    """
    Checks if x is close to y. Obtained equation from https://pytorch.org/docs/stable/generated/torch.isclose.html.

    Args:
        x: A float value.
        y: A float value.
        atol: Absolute tolerance. Default: 1e-8
        rtol: Relative tolerance. Default: 1e-5

    Returns:
        A boolean value indicating if x is close to y
    """
    return math.fabs(x - y) <= (atol + rtol * math.fabs(y))


def sigmoid(x: float) -> float:
    """
    Sigmoid function.

    Args:
        x: A float value

    Returns:
        A float value 1 / (1 + e^-x)
    """
    return 1.0 / (1.0 + math.exp(-x))


def relu(x: float) -> float:
    return 0.0 if x <= 0.0 else x


def log(x: float) -> float:
    return math.log(x)


def exp(x: float) -> float:
    return math.exp(x)


def inv(x: float) -> float:
    return 1.0 / x


def log_back(x: float, y: float) -> float:
    return y / x


def inv_back(x: float, y: float) -> float:
    return -y / x**2


def relu_back(x: float, y: float) -> float:
    return 0.0 if x <= 0 else y


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.
def map(func: Callable[[float], float], xs: Iterable[float]) -> Iterable[float]:
    return [func(x) for x in xs]


def zipWith(
    func: Callable[[float, float], float], xs: Iterable[float], ys: Iterable[float]
) -> Iterable[float]:
    return [func(x, y) for x, y in zip(xs, ys)]


def reduce(func: Callable[[float, float], float], xs: Iterable[float]) -> float:
    if len(xs) == 0:
        return 0.0

    acc = xs[0]
    for x in xs[1:]:
        acc = func(acc, x)
    return acc


def negList(xs: Iterable[float]) -> Iterable[float]:
    return map(neg, xs)


def addLists(xs: Iterable[float], ys: Iterable[float]) -> Iterable[float]:
    return zipWith(add, xs, ys)


def sum(xs: Iterable[float]) -> float:
    return reduce(add, xs)


def prod(xs: Iterable[float]) -> float:
    return reduce(mul, xs)

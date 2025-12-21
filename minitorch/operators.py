"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

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
    """Multiplies two numbers"""
    return x * y


def id(x: float) -> float:
    """Returns the input unchanged"""
    return x


def add(x: float, y: float) -> float:
    """Adds two numbers"""
    return x + y


def neg(x: float) -> float:
    """Negates a number"""
    return -x


def lt(x: float, y: float) -> bool:
    """Checks if one number is less than another"""
    return x < y


def eq(x: float, y: float) -> bool:
    """Checks if two numbers are equal"""
    return x == y


def max(x: float, y: float) -> float:
    """Returns the larger of two numbers"""
    if x >= y:
        return x
    else:
        return y


def is_close(x: float, y: float) -> bool:
    """Checks if two numbers are close in value"""
    return abs(x - y) < 0.01


def sigmoid(x: float) -> float:
    """Calculates the sigmoid function"""
    return 1 / (1 + math.exp(-x))


def relu(x: float) -> float:
    """Applies the ReLU activation function"""
    return max(0, x)


def log(x: float) -> float:
    """Calculates the natural logarithm"""
    return math.log(x)


def exp(x: float) -> float:
    """Calculates the exponential function"""
    return math.exp(x)


def inv(x: float) -> float:
    """Calculates the reciprocal"""
    return 1 / x


def log_back(x: float, y: float) -> float:
    """Computes the derivative of log times a second arg"""
    return y / x


def inv_back(x: float, y: float) -> float:
    """Computes the derivative of reciprocal times a second arg"""
    return -y / x**2


def relu_back(x: float, y: float) -> float:
    """Computes the derivative of ReLU times a second arg"""
    if x > 0:
        return y
    else:
        return 0


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
def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], list[float]]:
    """Higher-order function that applies a given function to each element of an iterable"""

    def apply(xx: Iterable[float]) -> list[float]:
        cur = []
        for x in xx:
            cur.append(fn(x))
        return cur

    return apply


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], list[float]]:
    """Higher-order function that combines elements from two iterables using a given function"""

    def apply(xx: Iterable[float], yy: Iterable[float]) -> list[float]:
        cur = []
        for x, y in zip(xx, yy):
            cur.append(fn(x, y))
        return cur

    return apply


def reduce(
    fn: Callable[[float, float], float], s: float
) -> Callable[[Iterable[float]], float]:
    """Higher-order function that reduces an iterable to a single value using a given function"""

    def apply(xx: Iterable[float]) -> float:
        out = s
        for x in xx:
            out = fn(out, x)
        return out

    return apply


def negList(xx: Iterable[float]) -> list[float]:
    """Negate all elements in a list using map"""
    return map(neg)(xx)


def addLists(xx: Iterable[float], yy: Iterable[float]) -> list[float]:
    """Add corresponding elements from two lists using zipWith"""
    return zipWith(add)(xx, yy)


def sum(xx: Iterable[float]) -> float:
    """Sum all elements in a list using reduce"""
    return reduce(add, 0.0)(xx)


def prod(xx: Iterable[float]) -> float:
    """Calculate the product of all elements in a list using reduce"""
    return reduce(mul, 1.0)(xx)

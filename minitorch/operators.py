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
    """Calculates $x*y$.

    Args:
    ----
        x: First value x.
        y: Second value y.

    Returns:
    -------
        $x * y$.

    """
    return x * y


def id(x: float) -> float:
    """Preserve input value unchanged.

    Args:
    ----
        x: Input value x.

    Returns:
    -------
        return Input value x out.

    """
    return x


def add(x: float, y: float) -> float:
    """Add two floats $x + y$.

    Args:
    ----
        x: First value x.
        y: Second value y.

    Returns:
    -------
        return $x + y$.

    """
    return x + y


def neg(x: float) -> float:
    """Make input value change sign.

    Args:
    ----
        x: input value x.

    Returns:
    -------
        return $-x$.

    """
    return -1.0 * x


def lt(x: float, y: float) -> float:
    """Checks if one number is less than another.

    Args:
    ----
        x: First value x.
        y: Second value y.

    Returns:
    -------
        1.0 if x is less than y else 0.0.

    """
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Checks if two numbers are equal.

    Args:
    ----
        x: First value x.
        y: Second value y.

    Returns:
    -------
        1.0 if x is equal to y else 0.0.

    """
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Returns the larger of two numbers.

    Args:
    ----
        x: First value x.
        y: Second value y.

    Returns:
    -------
        x if $x > y$ else y.

    """
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """Check if two numbers are close in value.

    Args:
    ----
        x: First value x.
        y: Second value y.

    Returns:
    -------
        1.0 if $|x - y| < 1e-2$, else 0.0.

    """
    return 1.0 if abs(x - y) < 1e-2 else 0.0


def sigmoid(x: float) -> float:
    """Calculates the sigmoid function.

    The sigmoid function is defined as $f(x) = 1 / (1 + e^{-x})$.
    It maps any real number to a value between 0.0 and 1.0.

    Args:
    ----
        x: Input value.

    Returns:
    -------
        The sigmoid of x.

    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        # enhance numerical stability.
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    r"""Applies the ReLU activation function.

    The Rectified Linear Unit (ReLU) is defined as $f(x) = \max(0, x)$.
    It returns the input value if it is positive, and 0.0 otherwise.

    Args:
    ----
        x: Input value x.

    Returns:
    -------
        The ReLU of x.

    """
    return x if x > 0.0 else 0.0


def log(x: float) -> float:
    """Calculates the natural logarithm.

    The natural logarithm is the inverse of the exponential function,
    defined for $x > 0$.

    Args:
    ----
        x: Input value x.

    Returns:
    -------
        The natural logarithm of x, clamped to a minimum positive epsilon.

    """
    return math.log(max(x, 1e-9))


def exp(x: float) -> float:
    """Calculates the exponential function.

    The exponential function is defined as $f(x) = e^x$.

    Args:
    ----
        x: Input value.

    Returns:
    -------
        The exponential of x.

    """
    return math.exp(x)


def inv(x: float) -> float:
    """Calculates the reciprocal.

    The reciprocal is defined as $f(x) = 1 / x$.

    Args:
    ----
        x: Input value.

    Returns:
    -------
        The reciprocal of x ($1.0 / x$).

    """
    return 1.0 / x


def log_back(x: float, d: float) -> float:
    r"""Computes the derivative of log times a second arg.

    $f(x) = \log(x)$, so $f'(x) = 1/x$.
    The backward pass computes $d \times f'(x)$.

    Args:
    ----
        x: The input value to the forward function.
        d: The upstream gradient (gradient flowing back).

    Returns:
    -------
        The gradient with respect to x.

    """
    return d / x


def inv_back(x: float, d: float) -> float:
    r"""Computes the derivative of reciprocal times a second arg.

    $f(x) = 1/x \implies f'(x) = -1/x^2$.
    The backward pass computes $d \times f'(x)$.

    Args:
    ----
        x: The input value to the reciprocal function.
        d: The upstream gradient (gradient flowing back).

    Returns:
    -------
        The derivative $-d / x^2$.

    """
    return -(d / (x**2))


def relu_back(x: float, d: float) -> float:
    """Computes the derivative of ReLU times a second arg.

    Args:
    ----
        x: The input value to the ReLU function during forward pass.
        d: The gradient flowing back from the upper layer.

    Returns:
    -------
        d if x is greater than 0, else 0.0.

    """
    return d if x > 0.0 else 0.0


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


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Higher-order map that applies a function to a list."""

    def _map(ls: Iterable[float]) -> Iterable[float]:
        return [fn(x) for x in ls]

    return _map


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Higher-order zipWith that combines two lists using a function."""

    def _zipWith(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        return [fn(x, y) for x, y in zip(ls1, ls2)]

    return _zipWith


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    """Higher-order reduce that reduces a list to a single value."""

    def _reduce(ls: Iterable[float]) -> float:
        val = start
        for x in ls:
            val = fn(val, x)
        return val

    return _reduce


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Negate each element in a list using map."""
    return map(neg)(ls)


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements from two lists using zipWith."""
    return zipWith(add)(ls1, ls2)


def sum(ls: Iterable[float]) -> float:
    """Sum a list of numbers using reduce."""
    return reduce(add, 0.0)(ls)


def prod(ls: Iterable[float]) -> float:
    """Product of a list of numbers using reduce."""
    return reduce(mul, 1.0)(ls)

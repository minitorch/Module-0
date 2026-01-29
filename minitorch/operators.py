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
def mul(a: float, b: float) -> float:
    """_summary_

    Args:
    ----
        a (float): first parameter
        b (float): second parameter

    Returns:
    -------
        float: the product of a and b

    """
    return a * b


def id(a: float) -> float:
    """Identity function.

    Args:
    ----
        a (float): input value

    Returns:
    -------
        float: the same input value

    """
    return a


def add(a: float, b: float) -> float:
    """Addition function.

    Args:
    ----
        a (float): first parameter
        b (float): second parameter

    Returns:
    -------
        float: the sum of a and b

    """
    return a + b


def neg(a: float) -> float:
    """Negation function.

    Args:
    ----
        a (float): input value

    Returns:
    -------
        float: the negated value of a

    """
    return -a


def lt(a: float, b: float) -> bool:
    """Less-than comparison function.

    Args:
    ----
        a (float): first parameter
        b (float): second parameter

    Returns:
    -------
        bool: True if a is less than b, False otherwise

    """
    return a < b


def eq(a: float, b: float) -> bool:
    """Equality comparison function.

    Args:
    ----
        a (float): first parameter
        b (float): second parameter

    Returns:
    -------
        bool: True if a is equal to b, False otherwise

    """
    return a == b


def max(a: float, b: float) -> float:
    """Maximum function.

    Args:
    ----
        a (float): first parameter
        b (float): second parameter

    Returns:
    -------
        float: the maximum of a and b

    """
    return a if a > b else b


def is_close(a: float, b: float) -> bool:
    """Check if two floats are close to each other within a tolerance.

    Args:
    ----
        a (float): first parameter
        b (float): second parameter

    Returns:
    -------
        bool: True if the absolute difference between a and b is less than 1e-2, False otherwise

    """
    return abs(a - b) < 1e-2


def sigmoid(a: float) -> float:
    """Sigmoid activation function.

    Args:
    ----
        a (float): input value

    Returns:
    -------
        float: the sigmoid of a

    """
    if a >= 0:
        return 1.0 / (1.0 + math.exp(-a))
    else:
        exp_a = math.exp(a)
        return exp_a / (1.0 + exp_a)


def relu(a: float) -> float:
    """ReLU activation function.

    Args:
    ----
        a (float): input value

    Returns:
    -------
        float: the ReLU of a

    """
    return a if a > 0 else 0.0


def log(a: float) -> float:
    """Natural logarithm function.

    Args:
    ----
        a (float): input value

    Returns:
    -------
        float: the natural logarithm of a

    """
    return math.log(a)


def exp(a: float) -> float:
    """Exponential function.

    Args:
    ----
        a (float): input value

    Returns:
    -------
        float: the exponential of a

    """
    return math.exp(a)


def log_back(a: float, b: float) -> float:
    """Backward pass for the logarithm function.

    Args:
    ----
        a (float): input value to the log function
        b (float): gradient from the subsequent layer

    Returns:
    -------
        float: the gradient of the log function

    """
    return b / a


def inv(a: float) -> float:
    """Inverse function.

    Args:
    ----
        a (float): input value

    Returns:
    -------
        float: the inverse of a

    """
    return 1.0 / a


def inv_back(a: float, b: float) -> float:
    """Backward pass for the inverse function.

    Args:
    ----
        a (float): input value to the inverse function
        b (float): gradient from the subsequent layer

    Returns:
    -------
        float: the gradient of the inverse function

    """
    return -b / (a * a)


def relu_back(a: float, b: float) -> float:
    """Backward pass for the ReLU function.

    Args:
    ----
        a (float): input value to the ReLU function
        b (float): gradient from the subsequent layer

    Returns:
    -------
        float: the gradient of the ReLU function

    """
    return b if a > 0 else 0.0


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


def map(fn: Callable[[float], float], ls: Iterable[float]) -> Iterable[float]:
    """Applies a function to each element in a list.

    Args:
    ----
        fn (Callable[[float], float]): function to apply
        ls (Iterable[float]): input list

    Returns:
    -------
        Iterable[float]: list with function applied to each element

    """
    return [fn(x) for x in ls]


def zipWith(
    fn: Callable[[float, float], float], ls1: Iterable[float], ls2: Iterable[float]
) -> Iterable[float]:
    """Combines two lists element-wise using a binary function.

    Args:
    ----
        fn (Callable[[float, float], float]): binary function to apply
        ls1 (Iterable[float]): first input list
        ls2 (Iterable[float]): second input list

    Returns:
    -------
        Iterable[float]: list with function applied element-wise

    """
    return [fn(x, y) for x, y in zip(ls1, ls2)]


def reduce(
    fn: Callable[[float, float], float], ls: Iterable[float], initial: float
) -> float:
    """Reduces a list to a single value using a binary function.

    Args:
    ----
        fn (Callable[[float, float], float]): binary function to apply
        ls (Iterable[float]): input list
        initial (float): initial value for reduction

    Returns:
    -------
        float: reduced value

    """
    result = initial
    for x in ls:
        result = fn(result, x)
    return result


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Negates each element in a list.

    Args:
    ----
        ls (Iterable[float]): input list

    Returns:
    -------
        Iterable[float]: list with each element negated

    """
    return map(neg, ls)


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Adds two lists element-wise.

    Args:
    ----
        ls1 (Iterable[float]): first input list
        ls2 (Iterable[float]): second input list

    Returns:
    -------
        Iterable[float]: list with elements added element-wise

    """
    return zipWith(add, ls1, ls2)


def sum(ls: Iterable[float]) -> float:
    """Sums all elements in a list.

    Args:
    ----
        ls (Iterable[float]): input list

    Returns:
    -------
        float: sum of all elements

    """
    return reduce(add, ls, 0.0)


def prod(ls: Iterable[float]) -> float:
    """Calculates the product of all elements in a list.

    Args:
    ----
        ls (Iterable[float]): input list

    Returns:
    -------
        float: product of all elements

    """
    return reduce(mul, ls, 1.0)

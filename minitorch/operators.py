"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:

# mul - Multiplies two numbers
# id - Returns the input unchanged
# add - Adds two numbers
# neg - Negates a number
# lt - Checks if one number is less than another
# eq - Checks if two numbers are equal
# max - Returns the larger of two numbers
# is_close - Checks if two numbers are close in value
# sigmoid - Calculates the sigmoid function
# relu - Applies the ReLU activation function
# log - Calculates the natural logarithm
# exp - Calculates the exponential function
# inv - Calculates the reciprocal
# log_back - Computes the derivative of log times a second arg
# inv_back - Computes the derivative of reciprocal times a second arg
# relu_back - Computes the derivative of ReLU times a second arg
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.


def mul(a, b):
    return a * b


def id(a):
    return a


def add(a, b):
    return a + b


def neg(a):
    return -1 * a


def lt(a, b):
    return a < b


def eq(a, b):
    return a == b


def max(a, b):
    return a if (a > b) else b


def is_close(a, b):
    return abs(a - b) < 1e-2


def sigmoid(a):
    if a >= 0:
        return 1 / (1 + math.exp(-a))
    else:
        return math.exp(a) / (1 + math.exp(a))


def relu(a):
    return max(0, a)


def log(a):
    return math.log(a)


def exp(a):
    return math.exp(a)


def inv(a):
    return 1 / a


def log_back(a, b):
    return inv(a) * b


def inv_back(a, b):
    return (-1) / (a**2) * b


def relu_back(a, b):
    if a <= 0:
        return 0
    else:
        return b


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions

# map - Higher-order function that applies a given function to each element of an iterable
# zipWith - Higher-order function that combines elements from two iterables using a given function
# reduce - Higher-order function that reduces an iterable to a single value using a given function

# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.


def map(fn, iter):
    return [fn(item) for item in iter]


def zipWith(fn, iter1, iter2):
    return [fn(x, y) for x, y in zip(iter1, iter2)]


def reduce(fn, iter):
    if len(iter) == 0:
        return 0
    result = iter[0]
    for item in iter[1:]:
        result = fn(result, item)
    return result


def negList(ls):
    return map(neg, ls)


def addLists(ls1, ls2):
    return zipWith(add, ls1, ls2)


def sum(ls):
    return reduce(add, ls)


def prod(ls):
    return reduce(mul, ls)

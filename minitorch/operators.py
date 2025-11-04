"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable, Iterator, List, TypeVar, Sequence

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


def mul(x: float, y: float) -> float:
    return x * y


def id(x: float) -> float:
    return x


def add(x: float, y: float) -> float:
    return x + y


def neg(x: float) -> float:
    return -x


def lt(x: float, y: float) -> float:
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def relu(x: float) -> float:
    return x if x > 0.0 else 0.0


def log(x: float) -> float:
    return math.log(x)


def exp(x: float) -> float:
    return math.exp(x)


def inv(x: float) -> float:
    return 1.0 / x


def log_back(a: float, b: float) -> float:
    return b / a


def inv_back(a: float, b: float) -> float:
    return -b / (a * a)


def relu_back(a: float, b: float) -> float:
    return b if a > 0.0 else 0.0


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


T = TypeVar("T")
U = TypeVar("U")


def map(fn: Callable[[T], U], it: Iterable[T]) -> List[U]:
    out: List[U] = []
    for v in it:
        out.append(fn(v))
    return out


def zipWith(fn: Callable[[T, U], T], a: Iterable[T], b: Iterable[U]) -> List[T]:
    out: List[T] = []
    ia = iter(a)
    ib = iter(b)
    while True:
        try:
            va = next(ia)
            vb = next(ib)
        except StopIteration:
            break
        out.append(fn(va, vb))
    return out


def reduce(fn: Callable[[T, T], T], it: Iterable[T], start: T) -> T:
    acc: T = start
    for v in it:
        acc = fn(acc, v)
    return acc


def negList(ls: Iterable[float]) -> List[float]:
    return map(neg, ls)


def addLists(a: Iterable[float], b: Iterable[float]) -> List[float]:
    return zipWith(add, a, b)


def sum(ls: Iterable[float]) -> float:
    return reduce(add, ls, 0.0)


def prod(ls: Iterable[float]) -> float:
    return reduce(mul, ls, 1.0)

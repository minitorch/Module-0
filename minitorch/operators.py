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


def mul(x: float, y: float) -> float:
    return x * y

def id(x: float) -> float:
    return x

def add(x: float,y: float)-> float:
    return x + y

def neg(x: float) -> float:
    return -x

def lt(x:float,y:float):
    return 1.0 if x<y else 0.0

def eq(x: float,y: float):
    return 1.0 if x == y else 0.0

def max(x: float,y: float):
    return x if x > y else y

def is_close(x: float,y: float) -> float:
    return abs(x - y) < 1e-2

def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))
 
def relu(x: float) -> float:
    return x if x > 0 else 0.0

def log(x: float) -> float:
    return math.log(x)

def exp(x: float) -> float:
    return math.exp(x)

def log_back(x: float,d: float) -> float:
    return d / x

def inv(x: float) -> float:
    return 1.0 / x

def inv_back(x: float, d: float) -> float:
    return -d / (x * x)

def relu_back(x: float, d: float)-> float:
    return d if x > 0.0 else 0.0

def map(fn: Callable[[float], float], lst: Iterable[float]) -> Iterable[float]:
    return [fn(x) for x in lst]

def zipWith(fn: Callable[[float], float], lst1: Iterable[float], lst2:Iterable[float]) -> Iterable[float]:
    return [fn(x1, x2) for x1, x2 in zip(lst1, lst2)]

def reduce(fn: Callable[[float, float], float], lst: Iterable[float]) -> float:
    lst = list(lst)  # 转为 list 以便取第一个
    if len(lst) == 0:
        return 0.0
    if len(lst) == 1:
        return lst[0]
    result = lst[0]
    for i in range(1, len(lst)):
        result = fn(result, lst[i])
    return result

def negList(lst: Iterable[float]) -> Iterable[float]:
    return map(neg, lst)

def addLists(lst1: Iterable[float], lst2: Iterable[float]) -> Iterable[float]:
    return zipWith(add, lst1, lst2)


def sum(lst: Iterable[float]) -> float:
    return reduce(add, lst)

def prod(lst: Iterable[float]) -> float:
    return reduce(mul, lst)
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

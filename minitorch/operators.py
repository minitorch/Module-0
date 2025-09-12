"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable, Iterable, Iterator


def mul(x: float, y: float) -> float:
    return x * y


def id(x: float) -> float:
    return x


def add(x: float, y: float) -> float:
    return x + y


def neg(x: float) -> float:
    return -x


def lt(x: float, y: float) -> bool:
    if x < y:
        return True
    else:
        return False


def max(x: float, y: float) -> float:
    if x > y:
        return x
    else:
        return y


def eq(x: float, y: float) -> bool:
    return x == y


def is_close(x: float, y: float) -> bool:
    epsilon = abs(x - y)
    if epsilon > 1e-2:
        return False
    else:
        return True


def sigmoid(x: float) -> float:
    return 1.0 / (1 + math.exp(-x))


def relu(x: float) -> float:
    return max(0, x)


def log(x: float) -> float:
    return math.log(x)


def exp(x: float) -> float:
    return math.exp(x)


def log_back(x: float, b: float) -> float:
    return b / x


def inv(x: float) -> float:
    return 1.0 / x


def inv_back(x: float, b: float) -> float:
    return -b / (x**2)


def relu_back(x: float, b: float) -> float:
    if x < 0:
        return 0
    else:
        return b


def map(xs: Iterable[float], fn: Callable[[float], float]) -> Iterator[float]:
    for x in xs:
        yield fn(x)


def zipWith(
    xs: Iterable[float], ys: Iterable[float], fn: Callable[[float, float], float]
) -> Iterable[float]:
    iter_ys = iter(ys)
    for x in xs:
        y = next(iter_ys)
        yield fn(x, y)


def reduce(xs: Iterable[float], fn: Callable[[float, float], float]) -> float:
    iter_xs = iter(xs)

    try:
        result = next(iter_xs)
    except StopIteration:
        return 0

    for x in iter_xs:
        result = fn(result, x)
    return result


def negList(xs: list[float]) -> list[float]:
    return list(map(xs, lambda x: -x))


def addLists(xs: list[float], ys: list[float]) -> list[float]:
    return list(zipWith(xs, ys, lambda x, y: x + y))


def sum(xs: list[float]) -> float:
    return reduce(xs, lambda x, y: x + y)


def prod(xs: list[float]) -> float:
    return reduce(xs, lambda x, y: x * y)

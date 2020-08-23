import minitorch
from hypothesis import settings
from hypothesis.strategies import composite, floats, integers, lists
import numpy as np

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")


small_ints = integers(min_value=1, max_value=3)
small_floats = floats(min_value=-100, max_value=100)


@composite
def vals(draw, size, number):
    pts = draw(lists(number, min_size=size, max_size=size,))
    return minitorch.tensor(pts)


@composite
def scalars(draw, min_value=-100000, max_value=100000):
    val = draw(floats(min_value=min_value, max_value=max_value))
    return minitorch.Scalar(val)


@composite
def shapes(draw):
    lsize = draw(lists(small_ints, min_size=1, max_size=4))
    return tuple(lsize)


@composite
def tensor_data(draw, numbers=floats(), shape=None):
    if shape is None:
        shape = draw(shapes())
    size = int(minitorch.prod(shape))
    data = draw(lists(numbers, min_size=size, max_size=size))
    return minitorch.TensorData(data, shape)


@composite
def indices(draw, layout):
    return tuple((draw(integers(min_value=0, max_value=s - 1)) for s in layout.shape))


@composite
def tensors(
    draw,
    numbers=floats(allow_nan=False, min_value=-100, max_value=100),
    backend=None,
    shape=None,
):
    td = draw(tensor_data(numbers, shape=shape))
    return minitorch.Tensor(td, backend=backend)


@composite
def shaped_tensors(
    draw,
    n,
    numbers=floats(allow_nan=False, min_value=-100, max_value=100),
    backend=None,
):
    td = draw(tensor_data(numbers))
    values = []
    for i in range(n):
        data = draw(lists(numbers, min_size=td.size, max_size=td.size))
        values.append(
            minitorch.Tensor(minitorch.TensorData(data, td.shape), backend=backend)
        )
    return values


@composite
def matmul_tensors(
    draw, numbers=floats(allow_nan=False, min_value=-100, max_value=100)
):

    i, j, k = [draw(integers(min_value=1, max_value=10)) for _ in range(3)]

    l1 = (i, j)
    l2 = (j, k)
    values = []
    for shape in [l1, l2]:
        size = int(minitorch.prod(shape))
        data = draw(lists(numbers, min_size=size, max_size=size))
        values.append(minitorch.Tensor(minitorch.TensorData(data, shape)))
    return values


def assert_close(a, b):
    np.testing.assert_allclose(a, b, 1e-2, 1e-2)

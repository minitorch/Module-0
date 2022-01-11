from attr.validators import le
from minitorch.operators import (
    mul,
    add,
    neg,
    relu,
    addLists,
    prod,
    negList,
    id,
    inv,
    lt,
    eq,
    is_close,
    max,
    sigmoid,
    relu_back,
    log_back,
    inv_back,
    sum,
)
from hypothesis import given
from hypothesis.strategies import lists
from .strategies import small_floats, assert_close
import pytest
from minitorch import MathTest


# ## Task 0.1 Basic hypothesis tests.


@pytest.mark.task0_1
@given(small_floats, small_floats)
def test_same_as_python(x, y):
    "Check that the main operators all return the same value of the python version"
    assert_close(mul(x, y), x * y)
    assert_close(add(x, y), x + y)
    assert_close(neg(x), -x)
    assert_close(max(x, y), x if x > y else y)
    if x != 0.0:
        assert_close(inv(x), 1.0 / x)


@pytest.mark.task0_1
@given(small_floats)
def test_relu(a):
    if a > 0:
        assert relu(a) == a
    if a < 0:
        assert relu(a) == 0.0


@pytest.mark.task0_1
@given(small_floats, small_floats)
def test_relu_back(a, b):
    if a > 0:
        assert relu_back(a, b) == b
    if a < 0:
        assert relu_back(a, b) == 0.0


@pytest.mark.task0_1
@given(small_floats)
def test_id(a):
    assert id(a) == a


@pytest.mark.task0_1
@given(small_floats)
def test_lt(a):
    "Check that a - 1.0 is always less than a"
    assert lt(a - 1.0, a) == 1.0
    assert lt(a, a - 1.0) == 0.0


@pytest.mark.task0_1
@given(small_floats)
def test_max(a):
    assert max(a - 1.0, a) == a
    assert max(a, a - 1.0) == a
    assert max(a + 1.0, a) == a + 1.0
    assert max(a, a + 1.0) == a + 1.0


@pytest.mark.task0_1
@given(small_floats)
def test_eq(a):
    assert eq(a, a) == 1.0
    assert eq(a, a - 1.0) == 0.0
    assert eq(a, a + 1.0) == 0.0


# ## Task 0.2 - Property Testing

# Implement the following property checks
# that ensure that your operators obey basic
# mathematical rules.


@pytest.mark.task0_2
@given(small_floats)
def test_sigmoid(a):
    """Check properties of the sigmoid function, specifically
    * It is always between 0.0 and 1.0.
    * one minus sigmoid is the same as negative sigmoid
    * It crosses 0 at 0.5
    * it is  strictly increasing.
    """

    assert sigmoid(a) >= 0.0
    assert sigmoid(a) <= 1.0
    assert_close(1.0 - sigmoid(a), sigmoid(-a))
    assert sigmoid(0.0) == 0.5
    if abs(a) > 10:
        assert sigmoid(a + abs(a) + 1e-2) >= sigmoid(a)
    else:
        assert sigmoid(a + abs(a) + 1e-2) > sigmoid(a)


@pytest.mark.task0_2
@given(small_floats, small_floats, small_floats)
def test_transitive(a, b, c):
    "Test the transitive property of less-than (a < b and b < c implies a < c)"
    # TODO: Implement for Task 0.2.
    sorted_abc = sorted([a, b, c])

    if lt(sorted_abc[0], sorted_abc[1]) == 1.0 and lt(sorted_abc[1], sorted_abc[2]) == 1.0: 
        assert lt(sorted_abc[0], sorted_abc[2]) == 1.0
    else:
        assert sorted_abc[0] == sorted_abc[1] or sorted_abc[1] == sorted_abc[2]


@pytest.mark.task0_2
@given(small_floats, small_floats)
def test_symmetric(a, b):
    """
    Write a test that ensures that :func:`minitorch.operators.mul` is symmetric, i.e.
    gives the same value regardless of the order of its input.
    """
    assert mul(a, b) == mul(b, a)


@pytest.mark.task0_2
@given(small_floats, small_floats, small_floats)
def test_distribute(a, b, c):
    r"""
    Write a test that ensures that your operators distribute, i.e.
    :math:`z \times (x + y) = z \times x + z \times y`
    """
    assert_close(mul(c, add(a, b)), add(mul(c, a), mul(c, b)))


@pytest.mark.task0_2
def test_other():
    """
    Write a test that ensures some other property holds for your functions.
    """
    None
    # TODO: Implement for Task 0.2.


# ## Task 0.3  - Higher-order functions

# These tests check that your higher-order functions obey basic
# properties.


@pytest.mark.task0_3
@given(small_floats, small_floats, small_floats, small_floats)
def test_zip_with(a, b, c, d):
    x1, x2 = addLists([a, b], [c, d])
    y1, y2 = a + c, b + d
    assert_close(x1, y1)
    assert_close(x2, y2)


@pytest.mark.task0_3
@given(
    lists(small_floats, min_size=5, max_size=5),
    lists(small_floats, min_size=5, max_size=5),
)
def test_sum_distribute(ls1, ls2):
    """
    Write a test that ensures that the sum of `ls1` plus the sum of `ls2`
    is the same as the sum of each element of `ls1` plus each element of `ls2`.
    """
    assert_close(sum(ls1) + sum(ls2), sum(addLists(ls1, ls2)))


@pytest.mark.task0_3
@given(lists(small_floats))
def test_sum(ls):
    assert_close(sum(ls), sum(ls))


@pytest.mark.task0_3
@given(small_floats, small_floats, small_floats)
def test_prod(x, y, z):
    assert_close(prod([x, y, z]), x * y * z)


@pytest.mark.task0_3
@given(lists(small_floats))
def test_negList(ls):
    check = negList(ls)
    for i in range(len(ls)):
        assert_close(check[i], -ls[i])


# ## Generic mathematical tests

# For each unit this generic set of mathematical tests will run.


one_arg, two_arg, _ = MathTest._tests()


@given(small_floats)
@pytest.mark.parametrize("fn", one_arg)
def test_one_args(fn, t1):
    name, base_fn, _ = fn
    base_fn(t1)


@given(small_floats, small_floats)
@pytest.mark.parametrize("fn", two_arg)
def test_two_args(fn, t1, t2):
    name, base_fn, _ = fn
    base_fn(t1, t2)


@given(small_floats, small_floats)
def test_backs(a, b):
    relu_back(a, b)
    inv_back(a + 2.4, b)
    log_back(abs(a) + 4, b)

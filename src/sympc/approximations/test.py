"""Test."""

from .utils import _as_f64_array


def test_approx(value):
    """Bla bla.

    Args:
        value: a tensor

    Returns:
        Nothing
    """
    value = _as_f64_array(value.numpy())
    return value

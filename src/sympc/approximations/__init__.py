"""Approximation Functions."""
from .exponential import exp
from .log import log
from .reciprocal import reciprocal
from .sigmoid import sigmoid
from .test import test_approx
from .utils import sign

APPROXIMATIONS = {
    "sigmoid": sigmoid,
    "log": log,
    "exp": exp,
    "reciprocal": reciprocal,
    "sign": sign,
    "test_approx": test_approx,
}

__all__ = ["APPROXIMATIONS"]

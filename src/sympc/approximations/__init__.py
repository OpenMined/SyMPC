"""Approximation Functions."""
from .exponential import exp
from .log import log
from .reciprocal import reciprocal
from .sigmoid import sigmoid
from .utils import sign

APPROXIMATIONS = {
    "sigmoid": sigmoid,
    "log": log,
    "exp": exp,
    "reciprocal": reciprocal,
    "sign": sign,
}

__all__ = ["APPROXIMATIONS"]

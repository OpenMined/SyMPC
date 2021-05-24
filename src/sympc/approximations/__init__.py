"""Approximation Functions."""
from .exponential import exp
from .log import log
from .reciprocal import reciprocal
from .sigmoid import sigmoid

APPROXIMATIONS = {"sigmoid": sigmoid, "log": log, "exp": exp, "reciprocal": reciprocal}

__all__ = ["APPROXIMATIONS"]

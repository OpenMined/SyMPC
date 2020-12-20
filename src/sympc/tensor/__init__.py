"""
Custom MPC Tensors
"""

from .share_tensor import ShareTensor  # isort:skip
from .mpc_tensor import MPCTensor

__all__ = [
    "ShareTensor",
    "MPCTensor",
]

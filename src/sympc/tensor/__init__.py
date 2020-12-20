"""
Custom MPC Tensors
"""

from .mpc_tensor import MPCTensor
from .share_tensor import ShareTensor

__all__ = [
    "ShareTensor",
    "MPCTensor",
]

"""
Custom MPC Tensors
"""

from .share import ShareTensor
from .share_control import ShareTensorCC

__all__ = [
    "ShareTensor",
    "ShareTensorCC",
]

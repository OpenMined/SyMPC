"""Custom MPC Tensors."""


from .share_tensor import ShareTensor  # isort:skip
from . import static
from .mpc_tensor import METHODS_TO_ADD
from .mpc_tensor import MPCTensor
from .register_approximation import RegisterApproximation
from .replicatedshare_tensor import PRIME_NUMBER
from .replicatedshare_tensor import ReplicatedSharedTensor

__all__ = [
    "ShareTensor",
    "ReplicatedSharedTensor",
    "MPCTensor",
    "RegisterApproximation",
    "METHODS_TO_ADD",
    "static",
    "PRIME_NUMBER",
]

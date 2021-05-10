# -*- coding: utf-8 -*-
"""This package represents the MPC component for Syft.

For the moment it has some basic functionality, but more would come in the following weeks.
"""

# third party
from pkg_resources import DistributionNotFound
from pkg_resources import get_distribution

from . import approximations  # noqa: 401
from . import config  # noqa: 401
from . import encoder  # noqa: 401
from . import protocol  # noqa: 401
from . import session  # noqa: 401
from . import store  # noqa: 401
from . import tensor  # noqa: 401

from . import module  # noqa: 401 isort: skip

try:
    # third party
    import syft

    syft.load("sympc")
except ImportError as e:
    print("PySyft is needed to be able to use SyMPC")
    raise e


def add_methods_tensor_syft() -> None:
    """Add SyMPC methods (only "share") to the torch.Tensor.

    Raises:
        ValueError: if torch.Tensor has already the method added.
    """
    # third party
    import torch

    for method in tensor.METHODS_TO_ADD:
        if getattr(torch.Tensor, method.__name__, None) is not None:
            raise ValueError(f"Method {method.__name__} already exists in tensor!")
        setattr(torch.Tensor, method.__name__, method)


try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = "unknown"
finally:
    del get_distribution, DistributionNotFound

add_methods_tensor_syft()

allowed_operations_on_share_tensor = [
    ("sympc.store.CryptoStore.get_primitives_from_store", "syft.lib.python.List"),
    ("sympc.store.CryptoStore.store", "syft.lib.python.Dict"),
    ("sympc.session.Session.crypto_store", "sympc.store.CryptoStore"),
    ("sympc.protocol.fss.fss.mask_builder", "sympc.tensor.ShareTensor"),
    ("sympc.protocol.fss.fss.evaluate", "sympc.tensor.ShareTensor"),
    ("sympc.protocol.spdz.spdz.mul_parties", "sympc.tensor.ShareTensor"),
    ("sympc.protocol.spdz.spdz.spdz_mask", "syft.lib.python.Tuple"),
    ("sympc.protocol.spdz.spdz.div_wraps", "sympc.tensor.ShareTensor"),
    (
        "sympc.session.Session.przs_generate_random_share",
        "sympc.tensor.ShareTensor",
    ),
    (
        "sympc.store.CryptoStore.populate_store",
        "syft.lib.python._SyNone",
    ),
    (
        "sympc.utils.get_new_generator",
        "torch.Generator",
    ),
    (
        "sympc.tensor.ShareTensor.__add__",
        "sympc.tensor.ShareTensor",
    ),
    (
        "sympc.tensor.ShareTensor.__sub__",
        "sympc.tensor.ShareTensor",
    ),
    (
        "sympc.tensor.ShareTensor.__rmul__",
        "sympc.tensor.ShareTensor",
    ),
    (
        "sympc.tensor.ShareTensor.__mul__",
        "sympc.tensor.ShareTensor",
    ),
    (
        "sympc.tensor.ShareTensor.__matmul__",
        "sympc.tensor.ShareTensor",
    ),
    (
        "sympc.tensor.ShareTensor.__truediv__",
        "sympc.tensor.ShareTensor",
    ),
    (
        "sympc.tensor.ShareTensor.__rmatmul__",
        "sympc.tensor.ShareTensor",
    ),
    (
        "sympc.tensor.ShareTensor.numel",
        "syft.lib.python.Int",  # FIXME: Can't we just return an int??
    ),
    (
        "sympc.tensor.ShareTensor.T",
        "sympc.tensor.ShareTensor",
    ),
    ("sympc.tensor.ShareTensor.unsqueeze", "sympc.tensor.ShareTensor"),
    ("sympc.tensor.ShareTensor.view", "sympc.tensor.ShareTensor"),
]

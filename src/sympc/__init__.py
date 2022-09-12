# -*- coding: utf-8 -*-
"""This package represents the MPC component for Syft.

For the moment it has some basic functionality, but more would come in the following weeks.
"""

# third party
from pkg_resources import DistributionNotFound
from pkg_resources import get_distribution

from . import api  # noqa: 401
from . import approximations  # noqa: 401
from . import config  # noqa: 401
from . import encoder  # noqa: 401
from . import protocol  # noqa: 401
from . import rust_test  # noqa: 401
from . import session  # noqa: 401
from . import store  # noqa: 401
from . import tensor  # noqa: 401

from . import grads  # noqa: 401 isort: skip
from . import module  # noqa: 401 isort: skip
from . import optim  # noqa: 401 isort: skip


try:
    # third party
    import syft

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

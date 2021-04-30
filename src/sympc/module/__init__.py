"""Implementations of the different neural network layers.

Add the share/reconstruct method to the Syft Module defined here:
https://github.com/OpenMined/PySyft/blob/dev/src/syft/lib/torch/module.py
"""


# stdlib
from collections import OrderedDict
import copy

# third party
import syft as sy
import torch

import sympc.module as sympc_module
from sympc.session import Session

from .nn import Conv2d
from .nn import Linear

MAP_TORCH_TO_SYMPC = {
    "Linear": Linear,
    "Conv2d": Conv2d,
}

MAP_TORCH_TO_SYMPC.update({f"{k}Pointer": v for k, v in MAP_TORCH_TO_SYMPC.items()})


def share(_self, session: Session) -> sy.Module:
    """Share remote state dictionary between the parties of the session.

    Arguments:
        session: Session holding different information like the parties and the computed data.

    Returns:
        Neural network module with the possibility to share remote state dictionary
    """
    mpc_module = copy.copy(_self)

    mpc_module._modules = OrderedDict()
    mpc_module.torch_ref = sympc_module

    for name, module in _self.modules.items():
        state_dict = module.state_dict()
        name_layer = type(module).__name__
        sympc_type_layer = MAP_TORCH_TO_SYMPC[name_layer]
        sympc_layer = sympc_type_layer(session=session)
        sympc_layer.share_state_dict(state_dict)
        mpc_module._modules[name] = sympc_layer

    return mpc_module


def reconstruct(_self) -> sy.Module:
    """Get back the shares from all the parties and reconstruct the underlying value.

    Returns:
        Neural network module with the secret reconstructed
    """
    syft_module = copy.copy(_self)

    # For this we will need to fetch the syft_module locally
    # even if it was created at another party
    syft_module.real_module = None
    syft_module.torch_ref = torch

    for name, module in _self.modules.items():
        state_dict = module.reconstruct_state_dict()
        torch_module = type(module).get_torch_module(module)
        torch_module.load_state_dict(state_dict)

        setattr(syft_module, name, torch_module)

    return syft_module


get = reconstruct

for method in {share, reconstruct, get}:
    if getattr(sy.Module, method.__name__, None) is not None:
        raise ValueError(f"Method {method.__name__} already exists in the sy.Module")

    setattr(sy.Module, method.__name__, method)

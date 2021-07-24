"""Implementations of the different neural network layers.

Add the share/reconstruct method to the Syft Module defined here:
https://github.com/OpenMined/PySyft/blob/dev/src/syft/lib/torch/module.py
"""


# stdlib
from collections import OrderedDict
import copy
from typing import Dict

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

ADDITIONAL_ATTRIBUTES = {
    "Conv2d": ["padding", "dilation", "groups", "stride", "in_channels", "out_channels"]
}

SKIP_LAYERS_NAME = {"Flatten"}


def copy_additional_attributes(layer: torch.nn.modules, layer_name: str) -> Dict:
    """Copy attributes from torch layer to SyMPC layer.

    Args:
        layer (torch.nn.modules): The torch layer
        layer_name (str): layer name

    Returns:
        additional_attributes (Dict): sympc layer with additional attributes
    """
    additional_attributes = {}

    if layer_name in ADDITIONAL_ATTRIBUTES:

        for arg in ADDITIONAL_ATTRIBUTES[layer_name]:
            additional_attributes[arg] = getattr(layer, arg)

    return additional_attributes


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
        if name_layer in SKIP_LAYERS_NAME:
            mpc_module._modules[name] = module
        else:
            sympc_type_layer = MAP_TORCH_TO_SYMPC[name_layer]
            sympc_layer = sympc_type_layer(session=session)
            additional_attributes = copy_additional_attributes(module, name_layer)
            sympc_layer.share_state_dict(state_dict, additional_attributes)
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

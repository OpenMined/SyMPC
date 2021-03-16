# stdlib
from collections import OrderedDict
import copy
from typing import Any
from typing import Dict

# third party
import syft as sy
import torch

import sympc.module as sympc_module

from .nn import Conv2d
from .nn import Linear

MAP_TORCH_TO_SYMPC = {torch.nn.Linear: Linear, torch.nn.Conv2d: Conv2d}


def share(_self, **kwargs: Dict[Any, Any]) -> sy.Module:
    session = kwargs["session"]
    parties = session.parties
    nr_parties = session.nr_parties

    mpc_module = copy.copy(_self)

    mpc_module._modules = OrderedDict()
    mpc_module.torch_ref = sympc_module

    for name, module in _self.modules.items():
        local_state_dict = module.state_dict()
        sympc_type_layer = MAP_TORCH_TO_SYMPC[type(module)]
        sympc_layer = sympc_type_layer(session=session)
        sympc_layer.share_state_dict(local_state_dict)
        mpc_module._modules[name] = sympc_layer

    return mpc_module


def reconstruct(_self):
    syft_module = copy.copy(_self)
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

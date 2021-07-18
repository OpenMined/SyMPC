"""The MPC Linear Layer."""

# stdlib
from collections import OrderedDict
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

# third party
import torch

from sympc.tensor import MPCTensor
from sympc.utils import ispointer

from .smpc_module import SMPCModule


class Linear(SMPCModule):
    """A Linear SMPC Layer."""

    __slots__ = [
        "weight",
        "bias",
        "session",
        "in_features",
        "out_features",
        "_parameters",
    ]

    in_features: Tuple[int]
    out_features: Tuple[int]
    weight: MPCTensor
    bias: Optional[MPCTensor]
    _parameters: Dict[str, MPCTensor]

    def __init__(self, session) -> None:
        """The initializer for the Linear layer.

        Args:
            session (Session): the session used to identify the layer
        """
        self.bias = None
        self._parameters = None
        self.session = session

    def forward(self, x: MPCTensor) -> MPCTensor:
        """Do a feedforward through the layer.

        Args:
            x (MPCTensor): the input

        Returns:
            An MPCTensor that results by applying the layer specific operation on the input
        """
        res = x @ self.weight.t()

        if self.bias is not None:
            res = res + self.bias

        return res

    __call__ = forward

    def parameters(self, recurse: bool = False) -> MPCTensor:
        """Get the parameters of the Linear module.

        Args:
            recurse (bool): For the moment not used. TODO

        Yields:
            Each parameter of the module
        """
        for param in self._parameters.values():
            yield param

    def share_state_dict(
        self,
        state_dict: Dict[str, Any],
        additional_attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Share the parameters of the normal Linear layer.

        Args:
            state_dict (Dict[str, Any]): the state dict that would be shared
            additional_attributes (Dict[str, Any]): Attributes apart from weights.
        """
        bias = None
        if ispointer(state_dict):
            weight = state_dict["weight"].resolve_pointer_type()
            if "bias" in weight.client.python.List(state_dict).get():
                bias = state_dict["bias"].resolve_pointer_type()
            shape = weight.client.python.Tuple(weight.shape)
            shape = shape.get()
        else:
            weight = state_dict["weight"]
            bias = state_dict.get("bias")
            shape = state_dict["weight"].shape

        self.out_features, self.in_features = shape
        self.weight = MPCTensor(
            secret=weight, session=self.session, shape=shape, requires_grad=True
        )
        self._parameters = OrderedDict({"weight": self.weight})

        if bias is not None:
            self.bias = MPCTensor(
                secret=bias,
                session=self.session,
                shape=(self.out_features,),
                requires_grad=True,
            )
            self._parameters["bias"] = self.bias

    def reconstruct_state_dict(self) -> Dict[str, Any]:
        """Reconstruct the shared state dict.

        Returns:
            The reconstructed state dict (Dict[str, Any])
        """
        state_dict = OrderedDict()
        state_dict["weight"] = self.weight.reconstruct()

        if self.bias is not None:
            state_dict["bias"] = self.bias.reconstruct()

        return state_dict

    @staticmethod
    def get_torch_module(linear_module: "Linear") -> torch.nn.Module:
        """Get a torch module from a given MPC Layer module.

        The parameters of the models are not set.

        Args:
            linear_module (Linear): the MPC Linear layer

        Returns:
            A torch Linear module
        """
        bias = linear_module.bias is not None
        module = torch.nn.Linear(
            in_features=linear_module.in_features,
            out_features=linear_module.out_features,
            bias=bias,
        )
        return module

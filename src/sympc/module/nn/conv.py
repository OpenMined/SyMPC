"""MPC Conv2d Layer."""

# stdlib
from collections import OrderedDict
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

# third party
import torch

from sympc.session import Session
from sympc.tensor import MPCTensor
from sympc.utils import ispointer

from .smpc_module import SMPCModule


class Conv2d(SMPCModule):
    """Convolutional 2D."""

    __slots__ = (
        "session",
        "weight",
        "bias",
        "kernel_size",
        "stride",
        "padding",
        "dilation",
        "groups",
        "_parameters",
    )

    in_channels: int
    out_channels: int
    kernel_size: Tuple[int, ...]
    stride: Tuple[int, ...]
    padding: Tuple[int, ...]
    dilation: Tuple[int, ...]
    groups: int
    weight: List[MPCTensor]
    bias: Optional[MPCTensor]
    _parameters: OrderedDict

    def __init__(self, session: Session) -> None:
        """Initialize Conv2d layer.

        The stride, padding, dilation and groups are hardcoded for the moment.

        Args:
            session (Session): the session used to identify the layer
        """
        self.session = session
        self.in_channels = None
        self.out_channels = None
        self.stride = 1
        self.padding = 0
        self.dilation = 1
        self.groups = 1
        self._parameters = None

    def forward(self, x: MPCTensor) -> MPCTensor:
        """Do a feedforward through the layer.

        Args:
            x (MPCTensor): the input

        Returns:
            An MPCTensor representing the layer specific operation applied on the input
        """
        res = x.conv2d(
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

        return res

    __call__ = forward

    def set_additional_attributes(self, attributes: Dict) -> None:
        """Sets attributes of conv apart from weights.

        Args:
            attributes (Dict): Attributes with their values.

        Raises:
            ValueError: If the attribute does not exist.
        """
        for attr in attributes.keys():
            if hasattr(self, attr):
                setattr(self, attr, attributes[attr])
            else:
                raise ValueError(f"Attribute {attr} does not exist in SyMPC module.")

    def share_state_dict(
        self,
        state_dict: Dict[str, Any],
        additional_attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Share the parameters of the normal Conv2d layer.

        Args:
            state_dict (Dict[str, Any]): the state dict that would be shared.
            additional_attributes (Dict[str, Any]): Attributes of conv apart from weights.

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

        if ispointer(additional_attributes):
            self.set_additional_attributes(
                additional_attributes.get().resolve_pointer_type()
            )
        else:
            self.set_additional_attributes(additional_attributes)

        # Weight shape (out_channel, in_channels/groups, kernel_size_w, kernel_size_h)
        # we have groups == 1

        (
            self.out_channels,
            self.in_channels,
            kernel_size_w,
            kernel_size_h,
        ) = shape

        self.kernel_size = (kernel_size_w, kernel_size_h)
        self.weight = MPCTensor(secret=weight, session=self.session, shape=shape)
        self._parameters = OrderedDict({"weight": self.weight})

        if bias is not None:
            self.bias = MPCTensor(
                secret=bias, session=self.session, shape=(self.out_channels,)
            )
            self._parameters["bias"] = self.bias

    def parameters(self, recurse: bool = False) -> MPCTensor:
        """Get the parameters of the Linear module.

        Args:
            recurse (bool): For the moment not used. TODO

        Yields:
            Each parameter of the module
        """
        for param in self._parameters.values():
            yield param

    def reconstruct_state_dict(self) -> Dict[str, Any]:
        """Reconstruct the shared state dict.

        Returns:
            Dict[str, Any]: The reconstructed state dict.
        """
        state_dict = OrderedDict()
        state_dict["weight"] = self.weight.reconstruct()

        if self.bias is not None:
            state_dict["bias"] = self.bias.reconstruct()

        return state_dict

    @staticmethod
    def get_torch_module(conv_module: "Conv2d") -> torch.nn.Module:
        """Get a torch module from a given MPC Conv2d module.

        The parameters of the models are not set.

        Args:
            conv_module (Conv2d): the MPC Conv2d layer

        Returns:
            torch.nn.Module: A torch Conv2d module.
        """
        bias = conv_module.bias is not None
        module = torch.nn.Conv2d(
            in_channels=conv_module.in_channels,
            out_channels=conv_module.out_channels,
            kernel_size=conv_module.kernel_size,
            stride=conv_module.stride,
            padding=conv_module.padding,
            dilation=conv_module.dilation,
            groups=conv_module.groups,
            bias=bias,
        )
        return module

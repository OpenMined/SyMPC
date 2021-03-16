# stdlib
from collections import OrderedDict
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

# third party
import torch

from sympc.tensor import MPCTensor

from .smpc_module import SMPCModule


class Conv2d(SMPCModule):
    __slots__ = [
        "session",
        "weight",
        "bias",
        "kernel_size",
        "stride",
        "padding",
        "dilation",
        "groups",
    ]

    in_channels: int
    out_channels: int
    kernel_size: Tuple[int, ...]
    stride: Tuple[int, ...]
    padding: Tuple[int, ...]
    dilation: Tuple[int, ...]
    groups: int
    weight: List[MPCTensor]
    bias: Optional[MPCTensor]

    def __init__(self, session) -> None:
        self.session = session
        self.stride = 1
        self.padding = 0
        self.dilation = 1
        self.groups = 1

    def forward(self, x: MPCTensor) -> MPCTensor:
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

    def share_state_dict(self, state_dict: Dict[str, Any]) -> None:
        # Weight shape (out_channel, in_channels/groups, kernel_size_w, kernel_size_h)
        # we have groups == 1
        (
            self.out_channels,
            self.in_channels,
            kernel_size_w,
            kernel_size_h,
        ) = state_dict["weight"].shape

        if kernel_size_w != kernel_size_h:
            raise ValueError(
                f"Kernel sizes mismatch {kernel_size_w} and {kernel_size_h}"
            )

        self.kernel_size = kernel_size_w

        self.weight = state_dict["weight"].share(session=self.session)

        if "bias" in state_dict:
            self.bias = state_dict["bias"].share(session=self.session)

    def reconstruct_state_dict(self) -> Dict[str, Any]:
        state_dict = OrderedDict()
        state_dict["weight"] = self.weight.reconstruct()

        if self.bias is not None:
            state_dict["bias"] = self.bias.reconstruct()

        return state_dict

    @staticmethod
    def get_torch_module(conv_module: "Conv2d") -> torch.nn.Module:
        bias = conv_module.bias is not None
        module = torch.nn.Conv2d(
            in_channels=conv_module.in_channels,
            out_channels=conv_module.out_channels,
            kernel_size=conv_module.kernel_size,
            bias=bias,
        )
        return module

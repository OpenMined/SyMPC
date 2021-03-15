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


class Conv2d:
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

        # Weight shape (out_channel, in_channels/groups, kernel_size_w, kernel_size_h)
        # we have groups == 1
        (
            out_channels,
            in_channels,
            kernel_size_w,
            kernel_size_h,
        ) = conv_module.weight.shape

        if kernel_size_w != kernel_size_h:
            raise ValueError(
                f"Kernel sizes mismatch {kernel_size_w} and {kernel_size_h}"
            )

        module = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size_w,
            bias=bias,
        )
        return module

    @staticmethod
    def eq_close(
        conv1: torch.nn.Conv2d,
        conv2: torch.nn.Conv2d,
        rtol: float = 1e-05,
        atol: float = 1e-08,
    ) -> bool:
        if not (
            isinstance(conv1, torch.nn.Conv2d) and isinstance(conv2, torch.nn.Conv2d)
        ):
            raise ValueError("linear1 and linear2 should be Conv2d layers")

        if (conv1.bias is None) != (conv2.bias is None):
            return False

        if not torch.allclose(conv1.weight, conv2.weight, rtol=rtol, atol=atol):
            return False

        if not torch.allclose(conv1.bias, conv2.bias, rtol=rtol, atol=atol):
            return False

        return True

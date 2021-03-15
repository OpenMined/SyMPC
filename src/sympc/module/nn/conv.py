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

RTOL = 10e-3


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
        print(self.weight.shape)
        print(self.bias.shape)
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
        print(state_dict["bias"])

        if "bias" in state_dict:
            print("Share bias", state_dict["bias"].shape)
            self.bias = state_dict["bias"].share(session=self.session)

    def reconstruct_state_dict(self) -> Dict[str, Any]:
        state_dict = OrderedDict()
        state_dict["weight"] = self.weight.reconstruct()

        if self.bias is not None:
            state_dict["bias"] = self.bias.reconstruct()

        return state_dict

    @staticmethod
    def get_torch_module(linear_module: "Conv2d") -> torch.nn.Module:
        bias = linear_module.bias is not None
        module = torch.nn.Linear(
            in_features=linear_module.in_features,
            out_features=linear_module.out_features,
            bias=bias,
        )
        return module

    @staticmethod
    def eq_close(conv1: torch.nn.Conv2d, conv2: torch.nn.Conv2d) -> bool:
        if not (
            isinstance(linear1, torch.nn.Conv2d)
            and isinstance(linear2, torch.nn.Conv2d)
        ):
            raise ValueError("linear1 and linear2 should be Conv2d layers")

        if (conv1.bias is None) != (linear2.bias is None):
            return False

        if not torch.allclose(conv1.weight, linear2.weight, rtol=RTOL):
            return False

        if not torch.allclose(conv1.bias, conv2.bias, rtol=RTOL):
            return False

        return True

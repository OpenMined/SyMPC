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

ATOL = 10e-5


class Linear:
    __slots__ = ["weight", "bias", "session", "in_features", "out_features"]

    def __init__(self, session) -> None:
        self.in_features: Optional[Tuple[int]] = None
        self.out_features: Optional[Tuple[int]] = None
        self.weight: List[MPCTensor] = []
        self.bias: Optional[MPCTensor] = None
        self.session = session

    def forward(self, x: MPCTensor) -> MPCTensor:
        res = x @ self.weight.T

        if self.bias is not None:
            res = res + self.bias

        return res

    __call__ = forward

    def share_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.in_features = state_dict["weight"].shape[1]
        self.out_features = state_dict["weight"].shape[0]

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
    def get_torch_module(linear_module: "Linear") -> torch.nn.Module:
        bias = linear_module.bias is not None
        module = torch.nn.Linear(
            in_features=linear_module.in_features,
            out_features=linear_module.out_features,
            bias=bias,
        )
        return module

    @staticmethod
    def eq_close(linear1: torch.nn.Linear, linear2: torch.nn.Linear) -> bool:
        if not (
            isinstance(linear1, torch.nn.Linear)
            and isinstance(linear2, torch.nn.Linear)
        ):
            raise ValueError("linear1 and linear2 should be Linear layers")

        if (linear1.bias is None) != (linear2.bias is None):
            return False

        if not torch.allclose(linear1.weight, linear2.weight, atol=ATOL):
            return False

        if not torch.allclose(linear1.bias, linear2.bias, atol=ATOL):
            return False

        return True

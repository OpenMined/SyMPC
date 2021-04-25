# stdlib
from typing import Any
from typing import Callable
from typing import List

# third party
import numpy as np
import syft as sy
import torch

from sympc.session import Session
from sympc.session import SessionManager
from sympc.tensor import MPCTensor


class ConvNet(sy.Module):
    def __init__(self, torch_ref):
        super(ConvNet, self).__init__(torch_ref=torch_ref)
        self.conv1 = self.torch_ref.nn.Conv2d(
            in_channels=1, out_channels=5, kernel_size=5
        )
        self.fc1 = self.torch_ref.nn.Linear(2880, 10)
        self.fc2 = self.torch_ref.nn.Linear(10, 5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.torch_ref.nn.functional.relu(x)
        x = x.view(1, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.torch_ref.nn.functional.relu(x)
        return x


def run_conv_model(get_clients: Callable[[int], List[Any]]):
    model = ConvNet(torch)

    clients = get_clients(2)

    session = Session(parties=clients)
    SessionManager.setup_mpc(session)

    mpc_model = model.share(session=session)

    x_secret = torch.randn((1, 1, 28, 28))
    x_mpc = MPCTensor(secret=x_secret, session=session)

    model.eval()

    # For the moment we have only inference
    expected = model(x_secret)

    res_mpc = mpc_model(x_mpc)
    assert isinstance(res_mpc, MPCTensor)

    res = res_mpc.reconstruct()
    expected = expected.detach().numpy()
    assert np.allclose(res, expected, atol=1e-3)


def test_run_conv_model(benchmark, get_clients: Callable[[int], List[Any]]):
    benchmark(run_conv_model, get_clients)

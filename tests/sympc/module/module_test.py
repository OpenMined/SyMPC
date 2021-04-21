# stdlib
from typing import Any
from typing import Callable
from typing import List
from typing import Type

# third party
import numpy as np
import pytest
import syft as sy
import torch

from sympc.module import MAP_TORCH_TO_SYMPC
from sympc.session import Session
from sympc.session import SessionManager
from sympc.tensor import MPCTensor


class LinearNet(sy.Module):
    def __init__(self, torch_ref):
        super(LinearNet, self).__init__(torch_ref=torch_ref)
        self.fc1 = self.torch_ref.nn.Linear(3, 10)
        self.fc2 = self.torch_ref.nn.Linear(10, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.torch_ref.nn.functional.relu(x)
        x = self.fc2(x)
        x = self.torch_ref.nn.functional.relu(x)
        return x


class ConvNet(sy.Module):
    def __init__(self, torch_ref, kernel_size=5):
        super(ConvNet, self).__init__(torch_ref=torch_ref)
        self.conv1 = self.torch_ref.nn.Conv2d(
            in_channels=1, out_channels=5, kernel_size=kernel_size
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


def test_run_linear_model(get_clients: Callable[[int], List[Any]]):
    model = LinearNet(torch)

    clients = get_clients(2)

    session = Session(parties=clients)
    SessionManager.setup_mpc(session)

    mpc_model = model.share(session=session)

    x_secret = torch.randn(2, 3)
    x_mpc = MPCTensor(secret=x_secret, session=session)

    model.eval()

    # For the moment we have only inference
    expected = model(x_secret)

    res_mpc = mpc_model(x_mpc)
    assert isinstance(res_mpc, MPCTensor)

    res = res_mpc.reconstruct()
    expected = expected.detach().numpy()
    assert np.allclose(res, expected, atol=1e-3)


def test_exception_conv2d_kernel_mismatch(get_clients):

    clients = get_clients(2)

    model = ConvNet(torch, kernel_size=(5, 4))
    # Setup the session for the computation
    session = Session(parties=clients)
    SessionManager.setup_mpc(session)

    with pytest.raises(ValueError):
        mpc_model = model.share(session=session)


def test_run_conv_model(get_clients: Callable[[int], List[Any]]):
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


@pytest.mark.parametrize("is_remote", [False, True])
@pytest.mark.parametrize("model_type", [LinearNet, ConvNet])
def test_reconstruct_shared_model(
    is_remote: bool, model_type: Type[sy.Module], get_clients: Callable[[int], Any]
):
    net = model_type(torch)

    clients = get_clients(2)

    session = Session(parties=clients)
    SessionManager.setup_mpc(session)

    if is_remote:
        model = net.send(clients[0])
    else:
        model = net

    mpc_model = model.share(session=session)
    res = mpc_model.reconstruct()

    assert isinstance(res, sy.Module)

    if is_remote:
        # If the model is remote fetch it such that we could compare it
        model = model.get()

    for name_res, name_expected in zip(res.modules, model.modules):
        assert name_res == name_expected

        module_expected = model.modules[name_expected]
        module_res = res.modules[name_res]

        name_module = type(module_expected).__name__
        assert MAP_TORCH_TO_SYMPC[name_module].eq_close(
            module_expected, module_res, atol=1e-4
        )

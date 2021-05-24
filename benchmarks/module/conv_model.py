"""Simple convolutional network and helper functions for benchmarking."""

# stdlib
from typing import Any
from typing import Callable
from typing import List
from typing import Tuple

# third party
import syft as sy
import torch

from sympc.session import Session
from sympc.session import SessionManager
from sympc.tensor import MPCTensor


class ConvNet(sy.Module):
    """Simple convolutional network."""

    def __init__(self, torch_ref):
        """Initialize convolutional network.

        Arguments:
            torch_ref: Reference to the torch library
        """
        super(ConvNet, self).__init__(torch_ref=torch_ref)
        self.conv1 = self.torch_ref.nn.Conv2d(
            in_channels=1, out_channels=5, kernel_size=5
        )
        self.fc1 = self.torch_ref.nn.Linear(2880, 10)
        self.fc2 = self.torch_ref.nn.Linear(10, 5)

    def forward(self, x):
        """Do a feedforward through the layer.

        Args:
            x (MPCTensor): the input

        Returns:
            An MPCTensor representing the layer specific operation applied on the input
        """
        x = self.conv1(x)
        x = self.torch_ref.nn.functional.relu(x)
        x = x.view(1, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.torch_ref.nn.functional.relu(x)
        return x


def set_up_model(
    get_clients: Callable[[int], List[Any]]
) -> Tuple[sy.Module, sy.Module, MPCTensor]:
    """Create a convolutional network and do a feedforwad.

    Args:
        get_clients: Fixture that returns a list of clients

    Returns:
        A tuple with three items: (origin Model, Syft Model, MPCTensor)
    """
    model = ConvNet(torch)

    clients = get_clients(2)

    session = Session(parties=clients)
    SessionManager.setup_mpc(session)

    mpc_model = model.share(session=session)

    x_secret = torch.randn((1, 1, 28, 28))
    x_mpc = MPCTensor(secret=x_secret, session=session)

    model.eval()

    expected = model(x_secret)

    return (expected, mpc_model, x_mpc)


def run_inference_conv_model(
    expected: sy.Module, mpc_model: sy.Module, x_mpc: MPCTensor
):
    """Run the inference on the passed in model.

    Args:
        expected: the origin model
        mpc_model: the Syft model
        x_mpc: the MPCTensor
    """
    """Run the inference on the passed in model"""
    res_mpc = mpc_model(x_mpc)
    res_mpc.reconstruct()

    expected = expected.detach().numpy()

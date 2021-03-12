# third party
import syft as sy
import torch

from sympc.session import Session
from sympc.session import SessionManager
from sympc.tensor import MPCTensor


class SyNet(sy.Module):
    def __init__(self, torch_ref):
        super(SyNet, self).__init__(torch_ref=torch_ref)
        self.fc1 = self.torch_ref.nn.Linear(3, 10)
        self.fc2 = self.torch_ref.nn.Linear(10, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.torch_ref.nn.functional.relu(x)
        x = self.fc2(x)
        x = self.torch_ref.nn.functional.relu(x)
        return x


def test_run_simple_model(get_clients):
    module = SyNet(torch)

    clients = get_clients(2)

    session = Session(parties=clients)
    SessionManager.setup_mpc(session)

    mpc_module = module.share(session=session)

    x = torch.randn(2, 3)
    x_secret = MPCTensor(secret=x, session=session)

    module.eval()

    # For the moment we have only inference
    expected = module(x)

    res_mpc = mpc_module(x_secret)
    assert isinstance(res_mpc, MPCTensor)

    res = res_mpc.reconstruct()
    assert torch.allclose(res, expected, atol=10e-5)


def test_reconstruct_shared_model(get_clients):
    module = SyNet(torch)

    clients = get_clients(2)

    session = Session(parties=clients)
    SessionManager.setup_mpc(session)

    mpc_module = module.share(session=session)

    res = mpc_module.reconstruct()

    assert isinstance(res, sy.Module)

    for name_res, name_expected in zip(res.modules, module.modules):
        module_expected = module.modules[name_expected]

        assert name_res == name_expected

        module_res = res.modules[name_res]

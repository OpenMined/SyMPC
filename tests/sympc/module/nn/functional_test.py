# stdlib

# third party
import numpy as np
import pytest
import torch

import sympc
from sympc.module.nn import mse_loss
from sympc.module.nn import relu
from sympc.module.nn import sigmoid
from sympc.session import Session
from sympc.session import SessionManager
from sympc.tensor import MPCTensor


def test_sigmoid(get_clients) -> None:
    clients = get_clients(2)
    session = Session(parties=clients)
    SessionManager.setup_mpc(session)

    secret = torch.Tensor([-2, -1.5, 0, 1, 1.5, 2])
    mpc_tensor = MPCTensor(secret=secret, session=session)

    res = sigmoid(mpc_tensor)
    res_expected = torch.sigmoid(secret)

    assert np.allclose(res.reconstruct(), res_expected, atol=1e-1)


def test_relu(get_clients) -> None:
    clients = get_clients(2)
    session = Session(parties=clients)
    SessionManager.setup_mpc(session)

    secret = torch.Tensor([-2, -1.5, 0, 1, 1.5, 2])
    mpc_tensor = MPCTensor(secret=secret, session=session)

    res = relu(mpc_tensor)
    res_expected = torch.nn.functional.relu(secret)

    assert all(res.reconstruct() == res_expected)


@pytest.mark.xfail  # flaky test
@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_mse_loss(get_clients, reduction) -> None:
    clients = get_clients(4)
    session = Session(parties=clients)
    SessionManager.setup_mpc(session)

    y_secret = torch.Tensor([0.23, 0.32, 0.2, 0.3])
    y_mpc = MPCTensor(secret=y_secret, session=session)

    y_pred = torch.Tensor([0.1, 0.3, 0.4, 0.2])
    y_pred_mpc = MPCTensor(secret=y_pred, session=session)

    res = mse_loss(y_mpc, y_pred_mpc, reduction)
    res_expected = torch.nn.functional.mse_loss(y_secret, y_pred, reduction=reduction)

    assert np.allclose(res.reconstruct(), res_expected, atol=1e-4)


POSSIBLE_CONFIGS_MAXPOOL_2D = [
    (1, 1, 0),
    (2, 1, 0),
    (2, 1, 1),
    (2, 2, 0),
    (2, 2, 1),
    (3, 1, 0),
    (3, 1, 1),
    (3, 2, 0),
    (3, 2, 1),
    (3, 3, 0),
    (3, 3, 1),
    ((5, 3), (1, 2), (2, 1)),
]


@pytest.mark.xfail  # flaky test
@pytest.mark.parametrize("kernel_size, stride, padding", POSSIBLE_CONFIGS_MAXPOOL_2D)
def test_max_pool2d(get_clients, kernel_size, stride, padding) -> None:
    clients = get_clients(2)
    session = Session(parties=clients)
    SessionManager.setup_mpc(session)

    secret = torch.Tensor(
        [
            [
                [0.23, 0.32, 0.62, 2.23, 5.32],
                [0.2, -0.3, -0.53, -15, 0.32],
                [0.22, 0.42, -10, -0.55, 2.32],
                [0.12, 0.22, -10, -0.35, -3.2],
                [23.12, -4.22, 5.3, -0.12, 6.0],
            ]
        ]
    )
    mpc = MPCTensor(secret=secret, session=session)

    res = sympc.module.nn.max_pool2d(
        mpc, kernel_size=kernel_size, stride=stride, padding=padding
    )
    res_expected = torch.max_pool2d(
        secret, kernel_size=kernel_size, stride=stride, padding=padding
    )

    assert np.allclose(res.reconstruct(), res_expected, atol=1e-4)


@pytest.mark.xfail  # flaky test
def test_max_pool2d_raises_value_error_kernel_gt_input(get_clients) -> None:
    clients = get_clients(2)
    session = Session(parties=clients)
    SessionManager.setup_mpc(session)

    secret = torch.Tensor([[[0.23]]])
    mpc = MPCTensor(secret=secret, session=session)

    with pytest.raises(ValueError):
        sympc.module.nn.max_pool2d(mpc, kernel_size=5, stride=1, padding=1)

    with pytest.raises(ValueError):
        sympc.module.nn.max_pool2d(
            mpc, kernel_size=2, stride=1, padding=0, return_indices=True
        )

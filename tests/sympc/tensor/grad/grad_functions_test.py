# stdlib
from typing import List

# third party
import numpy as np
import pytest
import torch

from sympc.session import Session
from sympc.session import SessionManager
from sympc.tensor import MPCTensor
from sympc.tensor.grads.grad_functions import GradAdd
from sympc.tensor.grads.grad_functions import GradConv2d
from sympc.tensor.grads.grad_functions import GradFlatten
from sympc.tensor.grads.grad_functions import GradFunc
from sympc.tensor.grads.grad_functions import GradMul
from sympc.tensor.grads.grad_functions import GradReshape
from sympc.tensor.grads.grad_functions import GradSigmoid
from sympc.tensor.grads.grad_functions import GradSub
from sympc.tensor.grads.grad_functions import GradSum
from sympc.tensor.grads.grad_functions import GradT
from sympc.utils import parallel_execution


def test_grad_func_abstract_forward_exception() -> None:
    with pytest.raises(NotImplementedError):
        GradFunc.forward({})


def test_grad_func_abstract_backward_exception() -> None:
    with pytest.raises(NotImplementedError):
        GradFunc.backward({})


def test_grad_transpose_forward(get_clients) -> None:
    secret = torch.Tensor([[1, 2, 3], [4, 5, 6]])
    mpc_tensor = secret.share(parties=get_clients(4))

    ctx = {}
    res_mpc = GradT.forward(ctx, mpc_tensor)

    res = res_mpc.reconstruct()
    expected = secret.t()

    assert (res == expected).all()


def test_grad_transpose_backward(get_clients) -> None:
    parties = get_clients(4)
    grad = torch.Tensor([[1, 2, 3], [4, 5, 6]])
    grad_mpc = grad.t().share(parties=parties)

    ctx = {}
    res_mpc = GradT.backward(ctx, grad_mpc)

    res = res_mpc.reconstruct()
    expected = grad

    assert (res == expected).all()


def test_grad_add_different_dims_forward(get_clients) -> None:
    parties = get_clients(4)
    x = torch.Tensor([[1, 2, 3], [4, 5, 6]])
    y = torch.Tensor([1, 2, 3])

    x_mpc = x.share(parties=parties)
    y_mpc = y.share(parties=parties)

    ctx = {}
    res_mpc = GradAdd.forward(ctx, x_mpc, y_mpc)

    res = res_mpc.reconstruct()
    expected = x + y

    assert (res == expected).all()


def test_grad_add_different_dims_backward(get_clients) -> None:
    parties = get_clients(4)

    grad = torch.Tensor([[[2, 4, 6], [5, 7, 9]]])
    grad_x = grad
    grad_y = torch.Tensor([[7, 11, 15]])
    grad_mpc = grad.share(parties=parties)

    ctx = {"x_shape": (2, 3), "y_shape": (1, 3)}
    res_mpc_x, res_mpc_y = GradAdd.backward(ctx, grad_mpc)

    assert (res_mpc_x.reconstruct() == grad_x).all()
    assert (res_mpc_y.reconstruct() == grad_y).all()


def test_grad_add_forward(get_clients) -> None:
    parties = get_clients(4)
    x = torch.Tensor([[1, 2, 3], [4, 5, 6]])
    y = torch.Tensor([[1, 4, 6], [8, 10, 12]])

    x_mpc = x.share(parties=parties)
    y_mpc = y.share(parties=parties)

    ctx = {}
    res_mpc = GradAdd.forward(ctx, x_mpc, y_mpc)

    res = res_mpc.reconstruct()
    expected = x + y

    assert (res == expected).all()


def test_grad_add_backward(get_clients) -> None:
    parties = get_clients(4)

    grad = torch.Tensor([1, 2, 3, 4])
    grad_mpc = grad.share(parties=parties)

    ctx = {"x_shape": (4,), "y_shape": (4,)}
    res_mpc_x, res_mpc_y = GradAdd.backward(ctx, grad_mpc)

    assert (res_mpc_x.reconstruct() == grad).all()
    assert (res_mpc_y.reconstruct() == grad).all()


def test_grad_sum_forward(get_clients) -> None:
    parties = get_clients(4)
    x = torch.Tensor([[1, 2, 3], [4, 5, 6]])

    x_mpc = x.share(parties=parties)

    ctx = {}
    res_mpc = GradSum.forward(ctx, x_mpc)

    assert ctx["x_shape"] == (2, 3)

    res = res_mpc.reconstruct()
    expected = x.sum()

    assert (res == expected).all()


def test_grad_sum_backward(get_clients) -> None:
    parties = get_clients(4)
    grad = torch.tensor(420)

    grad_mpc = grad.share(parties=parties)

    shape = (2, 3)
    ctx = {"x_shape": shape}

    res_mpc = GradSum.backward(ctx, grad_mpc)

    res = res_mpc.reconstruct()
    expected = torch.ones(size=shape) * grad

    assert (res == expected).all()


def test_grad_sub_forward(get_clients) -> None:
    parties = get_clients(4)
    x = torch.Tensor([[1, 2, 3], [4, 5, 6]])
    y = torch.Tensor([[1, 4, 6], [8, 10, 12]])

    x_mpc = x.share(parties=parties)
    y_mpc = y.share(parties=parties)

    ctx = {}
    res_mpc = GradSub.forward(ctx, x_mpc, y_mpc)

    res = res_mpc.reconstruct()
    expected = x - y

    assert (res == expected).all()


def test_grad_sub_backward(get_clients) -> None:
    parties = get_clients(4)

    grad = torch.Tensor([1, 2, 3, 4])
    grad_mpc = grad.share(parties=parties)

    ctx = {"x_shape": (4,), "y_shape": (4,)}
    res_mpc_x, res_mpc_y = GradSub.backward(ctx, grad_mpc)

    assert (res_mpc_x.reconstruct() == grad).all()
    assert (res_mpc_y.reconstruct() == -grad).all()


def test_grad_sub_different_dims_forward(get_clients) -> None:
    parties = get_clients(4)
    x = torch.Tensor([[1, 2, 3], [4, 5, 6]])
    y = torch.Tensor([1, 2, 3])

    x_mpc = x.share(parties=parties)
    y_mpc = y.share(parties=parties)

    ctx = {}
    res_mpc = GradSub.forward(ctx, x_mpc, y_mpc)

    res = res_mpc.reconstruct()
    expected = x - y

    assert (res == expected).all()


def test_grad_sub_different_dims_backward(get_clients) -> None:
    parties = get_clients(4)

    grad = torch.Tensor([[[2, 4, 6], [5, 7, 9]]])
    grad_x = grad
    grad_y = -torch.Tensor([[7, 11, 15]])
    grad_mpc = grad.share(parties=parties)

    ctx = {"x_shape": (2, 3), "y_shape": (1, 3)}
    res_mpc_x, res_mpc_y = GradSub.backward(ctx, grad_mpc)

    assert (res_mpc_x.reconstruct() == grad_x).all()
    assert (res_mpc_y.reconstruct() == grad_y).all()


def test_grad_sigmoid_forward(get_clients) -> None:
    # We need Function Secret Sharing (only for 2 parties) for
    # comparing
    parties = get_clients(2)
    x = torch.Tensor([7, 10, 12])

    x_mpc = x.share(parties=parties)

    ctx = {}
    res_mpc = GradSigmoid.forward(ctx, x_mpc)

    assert "probabilities" in ctx

    res = res_mpc.reconstruct()
    expected = x.sigmoid()

    assert np.allclose(res, expected, rtol=1e-2)


def test_grad_sigmoid_backward(get_clients) -> None:
    parties = get_clients(4)
    grad = torch.tensor([0.3, 0.4, 0.7])

    grad_mpc = grad.share(parties=parties)

    ctx = {"probabilities": grad}

    res_mpc = GradSigmoid.backward(ctx, grad_mpc)

    res = res_mpc.reconstruct()
    expected = grad * grad * (1 - grad)

    assert np.allclose(res, expected, rtol=1e-2)


def test_grad_mul_forward(get_clients) -> None:
    parties = get_clients(4)
    x = torch.Tensor([[1, 2], [3, -4]])
    y = torch.Tensor([[1, -4], [8, 9]])

    x_mpc = x.share(parties=parties)
    y_mpc = y.share(parties=parties)

    ctx = {}
    res_mpc = GradMul.forward(ctx, x_mpc, y_mpc)

    assert "x" in ctx
    assert "y" in ctx

    res = res_mpc.reconstruct()
    expected = x * y

    assert np.allclose(res, expected, rtol=1e-3)


def test_grad_mul_backward(get_clients) -> None:
    parties = get_clients(4)

    grad = torch.Tensor([[1, 2], [3, 4]])
    x = torch.Tensor([[1, 2], [3, -4]])
    y = torch.Tensor([[1, -4], [8, 9]])

    x_mpc = x.share(parties=parties)
    y_mpc = y.share(parties=parties)
    grad_mpc = grad.share(parties=parties)

    ctx = {"x": x_mpc, "y": y_mpc}

    res_mpc_x, res_mpc_y = GradMul.backward(ctx, grad_mpc)

    assert np.allclose(res_mpc_x.reconstruct(), y * grad, rtol=1e-3)
    assert np.allclose(res_mpc_y.reconstruct(), x * grad, rtol=1e-3)


def test_grad_conv2d_forward(get_clients) -> None:
    parties = get_clients(4)
    input_secret = torch.ones(1, 1, 4, 4)
    weight_secret = torch.ones(1, 1, 2, 2)
    input = input_secret.share(parties=parties)
    weight = weight_secret.share(parties=parties)

    kwargs = {"bias": None, "stride": 1, "padding": 0, "dilation": 1, "groups": 1}

    ctx = {}
    res_mpc = GradConv2d.forward(ctx, input, weight, **kwargs)

    assert "input" in ctx
    assert "weight" in ctx
    assert "stride" in ctx
    assert "padding" in ctx
    assert "dilation" in ctx
    assert "groups" in ctx

    res = res_mpc.reconstruct()
    expected = torch.nn.functional.conv2d(input_secret, weight_secret, **kwargs)

    assert np.allclose(res, expected, rtol=1e-3)


def test_grad_conv2d_backward(get_clients) -> None:
    parties = get_clients(4)

    grad = torch.Tensor([[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]])

    input = torch.Tensor(
        [
            [
                [
                    [-2.1805, -1.3338, -0.9718, -0.1335],
                    [-0.5632, 1.2667, 0.9994, -0.0627],
                    [-0.9563, 0.5861, -1.4422, -0.4825],
                    [0.2732, -1.1900, -0.6624, -0.7513],
                ]
            ]
        ]
    )

    weight = torch.Tensor([[[[0.3257, -0.7538], [-0.5773, -0.7619]]]])

    x_mpc = input.share(parties=parties)
    y_mpc = weight.share(parties=parties)
    grad_mpc = grad.share(parties=parties)

    ctx = {
        "input": x_mpc,
        "weight": y_mpc,
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "groups": 1,
    }

    res_mpc_input, res_mpc_weight = GradConv2d.backward(ctx, grad_mpc)
    expected_input = torch.nn.functional.grad.conv2d_input(input.size(), weight, grad)
    expected_weight = torch.nn.functional.grad.conv2d_weight(input, weight.size(), grad)

    assert np.allclose(res_mpc_input.reconstruct(), expected_input, rtol=1e-3)
    assert np.allclose(res_mpc_weight.reconstruct(), expected_weight, rtol=1e-3)


@pytest.mark.parametrize(
    "common_args", [[(6, 6), 2, 1, (3, 3), 1], [(4, 4), 1, 0, (2, 2), 1]]
)
@pytest.mark.parametrize("nr_parties", [2, 3, 4])
def test_get_grad_input_padding(get_clients, common_args: List, nr_parties) -> None:
    clients = get_clients(2)
    session = Session(parties=clients)
    SessionManager.setup_mpc(session)

    grad = torch.Tensor([[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]])
    grad_mpc = MPCTensor(secret=grad, session=session)

    input_size, stride, padding, kernel_size, dilation = common_args

    expected_padding = torch.nn.functional.grad._grad_input_padding(
        grad,
        input_size,
        (stride, stride),
        (padding, padding),
        kernel_size,
        (dilation, dilation),
    )

    args = [[el] + common_args + [session] for el in grad_mpc.share_ptrs]
    shares = parallel_execution(
        GradConv2d.get_grad_input_padding, grad_mpc.session.parties
    )(args)
    grad_input_padding = MPCTensor(shares=shares, session=grad_mpc.session)
    output_padding_tensor = grad_input_padding.reconstruct()
    output_padding_tensor /= grad_mpc.session.nr_parties
    calculated_padding = tuple(output_padding_tensor.to(torch.int).tolist())

    assert calculated_padding == expected_padding


def test_grad_reshape_forward(get_clients) -> None:
    parties = get_clients(4)
    x = torch.Tensor([[1, 2], [3, -4], [-9, 0]])

    x_mpc = x.share(parties=parties)

    ctx = {}
    shape = (3, 2)
    res_mpc = GradReshape.forward(ctx, x_mpc, shape)

    assert "x_shape" in ctx

    res_shape = res_mpc.shape

    assert res_shape == shape
    assert np.allclose(res_mpc.reconstruct(), x.reshape(shape), rtol=1e-3)


def test_grad_reshape_backward(get_clients) -> None:
    parties = get_clients(4)

    grad = torch.Tensor([[1, 2, 3], [3, 4, 7]])
    x = torch.Tensor([[1, 2], [3, -4], [5, 8]])

    x_mpc = x.share(parties=parties)
    grad_mpc = grad.share(parties=parties)

    ctx = {"x_shape": x_mpc.shape}

    res_mpc_grad = GradReshape.backward(ctx, grad_mpc)
    res_mpc_grad_shape = res_mpc_grad.shape

    assert res_mpc_grad_shape == x_mpc.shape
    assert np.allclose(res_mpc_grad.reconstruct(), grad.reshape(x_mpc.shape), rtol=1e-3)


@pytest.mark.parametrize("args", [[0, -1], [1, -1], [0, 1]])
def test_grad_flatten_forward(get_clients, args: list) -> None:
    parties = get_clients(4)
    x = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    x_mpc = x.share(parties=parties)

    ctx = {}
    start_dim, end_dim = args
    res_mpc = GradFlatten.forward(ctx, x_mpc, start=start_dim, end=end_dim)

    assert "x_shape" in ctx

    expected = torch.flatten(x, start_dim=start_dim, end_dim=end_dim)
    assert np.allclose(res_mpc.reconstruct(), expected, rtol=1e-3)


def test_grad_flatten_backward(get_clients) -> None:
    parties = get_clients(4)

    grad = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8])
    x = torch.Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    x_mpc = x.share(parties=parties)
    grad_mpc = grad.share(parties=parties)

    ctx = {"x_shape": x_mpc.shape}

    res_mpc_grad = GradFlatten.backward(ctx, grad_mpc)

    assert np.allclose(res_mpc_grad.reconstruct(), x, rtol=1e-3)

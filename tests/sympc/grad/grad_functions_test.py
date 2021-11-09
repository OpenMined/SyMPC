# stdlib

# third party
import numpy as np
import pytest
import syft as sy
import torch

import sympc
from sympc.grads.grad_functions import GradAdd
from sympc.grads.grad_functions import GradConv2d
from sympc.grads.grad_functions import GradDiv
from sympc.grads.grad_functions import GradFlatten
from sympc.grads.grad_functions import GradFunc
from sympc.grads.grad_functions import GradMatMul
from sympc.grads.grad_functions import GradMaxPool2D
from sympc.grads.grad_functions import GradMul
from sympc.grads.grad_functions import GradPow
from sympc.grads.grad_functions import GradReLU
from sympc.grads.grad_functions import GradReshape
from sympc.grads.grad_functions import GradSigmoid
from sympc.grads.grad_functions import GradSub
from sympc.grads.grad_functions import GradSum
from sympc.grads.grad_functions import GradT
from sympc.session import Session
from sympc.session import SessionManager
from sympc.tensor import MPCTensor


class LinearSyNet(sy.Module):
    def __init__(self, torch_ref):
        super(LinearSyNet, self).__init__(torch_ref=torch_ref)
        self.fc1 = self.torch_ref.nn.Linear(3, 10)
        self.fc2 = self.torch_ref.nn.Linear(10, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.torch_ref.nn.functional.relu(x)
        x = self.fc2(x)
        x = self.torch_ref.nn.functional.relu(x)
        return x


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

    assert "x" in ctx
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
        "x": x_mpc,
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


@pytest.mark.parametrize("power", [2, 4, 5])
def test_grad_pow_forward(get_clients, power) -> None:
    parties = get_clients(4)
    x = torch.Tensor([[1, 2, 3], [4, 5, 6]])

    x_mpc = x.share(parties=parties)

    ctx = {}

    res_mpc = GradPow.forward(ctx, x_mpc, power)

    assert "x" in ctx
    assert "y" in ctx

    res = res_mpc.reconstruct()
    expected = x ** power

    assert np.allclose(res, expected, rtol=1e-3)


@pytest.mark.parametrize("power", [1.0, torch.tensor([1, 3])])
def test_grad_pow_forward_exception(get_clients, power) -> None:
    parties = get_clients(4)
    x = torch.Tensor([[1, 2, 3], [4, 5, 6]])

    x_mpc = x.share(parties=parties)

    ctx = {}

    with pytest.raises(TypeError):
        GradPow.forward(ctx, x_mpc, power)


@pytest.mark.parametrize("power", [2, 4, 5])
def test_grad_pow_backward(get_clients, power) -> None:
    parties = get_clients(4)

    grad = torch.Tensor([1, 2, 3, 4])
    grad_mpc = grad.share(parties=parties)

    x = torch.Tensor([1, 4, 9, 16])
    x_mpc = x.share(parties=parties)

    ctx = {"x": x_mpc, "y": power}
    res_mpc = GradPow.backward(ctx, grad_mpc)
    res = res_mpc.reconstruct()

    expected = power * x ** (power - 1) * grad

    assert np.allclose(res, expected, rtol=1e-3)


def test_grad_matmul_forward(get_clients) -> None:
    parties = get_clients(4)
    x = torch.Tensor([[1, 2], [3, -4]])
    y = torch.Tensor([[1, -4], [8, 9]])

    x_mpc = x.share(parties=parties)
    y_mpc = y.share(parties=parties)

    ctx = {}
    res_mpc = GradMatMul.forward(ctx, x_mpc, y_mpc)

    assert "x" in ctx
    assert "y" in ctx

    res = res_mpc.reconstruct()
    expected = x @ y

    assert np.allclose(res, expected, rtol=1e-3)


def test_grad_matmul_backward(get_clients) -> None:
    parties = get_clients(4)

    grad = torch.Tensor([[1, 2], [3, 4]])
    x = torch.Tensor([[1, 2], [3, -4]])
    y = torch.Tensor([[1, -4], [8, 9]])

    x_mpc = x.share(parties=parties)
    y_mpc = y.share(parties=parties)
    grad_mpc = grad.share(parties=parties)

    ctx = {"x": x_mpc, "y": y_mpc}

    res_mpc_x, res_mpc_y = GradMatMul.backward(ctx, grad_mpc)

    assert np.allclose(res_mpc_x.reconstruct(), grad @ y.T, rtol=1e-3)
    assert np.allclose(res_mpc_y.reconstruct(), x.T @ grad, rtol=1e-3)


def test_grad_matmul_raise_value_error_mismatch_shape(get_clients) -> None:
    parties = get_clients(4)

    x = torch.Tensor([[1, 2], [3, -4]])
    x_mpc = x.share(parties=parties)

    grad = torch.Tensor([[1, 2], [3, 4]])
    grad_mpc = grad.share(parties=parties)

    y = torch.Tensor([[1, -4], [8, 9], [10, 11]])
    y_mpc = y.share(parties=parties)

    ctx = {"x": x_mpc, "y": y_mpc}

    with pytest.raises(ValueError):
        GradMatMul.backward(ctx, grad_mpc)


def test_grad_relu_forward(get_clients) -> None:
    # We need Function Secret Sharing (only for 2 parties) for
    # comparing
    parties = get_clients(2)
    x = torch.Tensor([-7, 0, 12])

    x_mpc = x.share(parties=parties)

    ctx = {}
    res_mpc = GradReLU.forward(ctx, x_mpc)

    assert "mask" in ctx

    res = res_mpc.reconstruct()
    expected = x.relu()

    assert np.allclose(res, expected, rtol=1e-3)


def test_grad_relu_backward(get_clients) -> None:
    parties = get_clients(2)
    grad = torch.tensor([0, -1.453, 0.574, -0.89])

    grad_mpc = grad.share(parties=parties)
    mask = torch.tensor([0, 0, 1, 0])

    ctx = {"mask": mask}

    res_mpc = GradReLU.backward(ctx, grad_mpc)

    res = res_mpc.reconstruct()
    expected = grad * mask
    assert np.allclose(res, expected, rtol=1e-3)


@pytest.mark.xfail  # flaky test
def test_forward(get_clients) -> None:
    model = LinearSyNet(torch)

    clients = get_clients(2)

    session = Session(parties=clients)
    session.autograd_active = True
    SessionManager.setup_mpc(session)
    mpc_model = model.share(session=session)

    x_secret = torch.tensor(
        [[0.125, -1.25, -4.25], [-3, 3, 8], [-3, 3, 8]], requires_grad=True
    )
    x_mpc = MPCTensor(secret=x_secret, session=session, requires_grad=True)

    out_torch = model(x_secret)
    out_mpc = mpc_model(x_mpc)

    s_torch = torch.sum(out_torch)
    s_mpc = out_mpc.sum()

    s_torch.backward()
    s_mpc.backward()

    assert np.allclose(x_mpc.grad.get(), x_secret.grad, rtol=1e-2)


def test_grad_maxpool_2d_dilation_error(get_clients) -> None:
    parties = get_clients(2)

    secret = torch.Tensor(
        [[[0.23, 0.32, 0.423], [0.2, -0.3, -0.53], [0.32, 0.42, -100]]]
    )

    x = secret.share(parties=parties)

    with pytest.raises(ValueError):
        GradMaxPool2D.forward({}, x, kernel_size=2, dilation=2)

    with pytest.raises(ValueError):
        GradMaxPool2D.forward({}, x, kernel_size=2, dilation=(1, 2))


@pytest.mark.xfail  # flaky test
def test_grad_maxpool_2d_backward_value_error_indices_shape(get_clients) -> None:
    parties = get_clients(2)

    secret = torch.tensor(
        [
            [
                [0.23, 0.32, 0.62, 2.23, 5.32],
                [0.12, 0.22, -10, -0.35, -3.2],
                [3.12, -4.22, 5.3, -0.12, 6.0],
            ]
        ],
        requires_grad=True,
    )

    ctx = {
        "x_shape": secret.shape,
        "kernel_size": (5, 5),
        "stride": 1,
        "padding": (1, 1),
        "dilation": 1,
        "indices": torch.tensor([[1, 2], [3, 4]]),
    }

    with pytest.raises(ValueError):
        GradMaxPool2D.backward(ctx, secret.share(parties=parties))


@pytest.mark.xfail  # flaky test
def test_grad_maxpool_2d_backward_value_error_kernel_gt_input(get_clients) -> None:
    parties = get_clients(2)

    secret = torch.tensor(
        [
            [
                [0.23, 0.32, 0.62, 2.23, 5.32],
                [0.12, 0.22, -10, -0.35, -3.2],
                [3.12, -4.22, 5.3, -0.12, 6.0],
            ]
        ],
        requires_grad=True,
    )

    ctx = {
        "x_shape": secret.shape,
        "kernel_size": (5, 5),
        "stride": 1,
        "padding": (1, 1),
        "dilation": 1,
        "indices": torch.tensor([[[[1]]]]),
    }

    with pytest.raises(ValueError):
        GradMaxPool2D.backward(ctx, secret.share(parties=parties))


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


def test_grad_maxpool_2d_forward_value_error_kernel_gt_input(get_clients) -> None:
    parties = get_clients(2)

    secret = torch.Tensor(
        [
            [
                [0.23, 0.32],
                [0.2, -0.3],
                [0.22, 0.42],
            ]
        ]
    )

    x = secret.share(parties=parties)

    with pytest.raises(ValueError):
        GradMaxPool2D.forward(
            {}, x, kernel_size=(6, 3), stride=1, padding=1, dilation=1
        )


@pytest.mark.xfail  # flaky test
@pytest.mark.parametrize("kernel_size, stride, padding", POSSIBLE_CONFIGS_MAXPOOL_2D)
def test_grad_maxpool_2d_forward(get_clients, kernel_size, stride, padding) -> None:
    parties = get_clients(2)

    secret = torch.Tensor(
        [
            [
                [0.23, 0.32, 0.62, 2.23, 5.32],
                [0.2, -0.3, -0.53, -15, 0.32],
                [0.22, 0.42, -10, -0.55, 2.32],
                [0.12, 0.22, -10, -0.35, -3.2],
                [3.12, -4.22, 5.3, -0.12, 6.0],
            ]
        ]
    )

    x = secret.share(parties=parties)

    ctx = {}
    res_mpc = GradMaxPool2D.forward(
        ctx, x, kernel_size=kernel_size, stride=stride, padding=padding, dilation=1
    )

    assert "x_shape" in ctx
    assert "kernel_size" in ctx
    assert "stride" in ctx
    assert "padding" in ctx
    assert "dilation" in ctx
    assert "indices" in ctx

    res = res_mpc.reconstruct()
    expected = torch.max_pool2d(
        secret, kernel_size=kernel_size, stride=stride, padding=padding, dilation=1
    )

    assert np.allclose(res, expected, rtol=1e-3)


@pytest.mark.xfail  # flaky test
@pytest.mark.parametrize("kernel_size, stride, padding", POSSIBLE_CONFIGS_MAXPOOL_2D)
def test_grad_maxpool_2d_backward(get_clients, kernel_size, stride, padding) -> None:
    parties = get_clients(2)

    secret = torch.tensor(
        [
            [
                [0.23, 0.32, 0.62, 2.23, 5.32],
                [0.2, -0.3, -0.53, -15, 0.32],
                [0.22, 0.42, -10, -0.55, 2.32],
                [0.12, 0.22, -10, -0.35, -3.2],
                [3.12, -4.22, 5.3, -0.12, 6.0],
            ]
        ],
        requires_grad=True,
    )

    output = torch.max_pool2d(
        secret, kernel_size=kernel_size, stride=stride, padding=padding, dilation=1
    )
    grad_mpc = output.share(parties=parties)

    x = secret.share(parties=parties)

    grad, indices = sympc.module.nn.functional.max_pool2d(
        x,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=1,
        return_indices=True,
    )

    ctx = {
        "x_shape": secret.shape,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": 1,
        "indices": indices,
    }

    output.backward(gradient=output)
    expected_grad = secret.grad

    res_mpc = GradMaxPool2D.backward(ctx, grad_mpc)

    res = res_mpc.reconstruct()
    assert np.allclose(res, expected_grad, rtol=1e-3)


def test_grad_div_forward(get_clients) -> None:
    # We need Function Secret Sharing (only for 2 parties) for
    # comparing
    parties = get_clients(2)
    x = torch.tensor([2.1, 3.2])
    y = torch.tensor([6.03, 4.1])

    x_mpc = x.share(parties=parties)
    y_mpc = y.share(parties=parties)

    ctx = {}
    res_mpc = GradDiv.forward(ctx, x_mpc, y_mpc)

    res = res_mpc.reconstruct()
    expected = x / y

    assert np.allclose(res, expected, rtol=1e-3)


def test_grad_div_backward(get_clients) -> None:
    parties = get_clients(2)

    session = Session(parties=parties)
    session.autograd_active = True
    SessionManager.setup_mpc(session)

    x_secret = torch.tensor([1.0, 2.1, 3.0, -4.13], requires_grad=True)
    x = MPCTensor(secret=x_secret, session=session, requires_grad=True)

    y_secret = torch.tensor([-2.0, 3.0, 4.39, 5.0], requires_grad=True)
    y = MPCTensor(secret=y_secret, session=session, requires_grad=True)

    z = x_secret / y_secret
    z.backward(torch.tensor([1, 1, 1, 1]))

    grad = torch.tensor([1, 1, 1, 1])
    grad_mpc = MPCTensor(secret=grad, session=session, requires_grad=True)

    ctx = {"x": x, "y": y, "result": x / y}

    grad_x, grad_y = GradDiv.backward(ctx, grad_mpc)

    expected_grad_x = x_secret.grad
    expected_grad_y = y_secret.grad

    res_x = grad_x.reconstruct()
    res_y = grad_y.reconstruct()

    assert np.allclose(res_x, expected_grad_x, rtol=1e-2)
    assert np.allclose(res_y, expected_grad_y, rtol=1e-2)

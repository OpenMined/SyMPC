# third party
import numpy as np
import pytest
import torch

from sympc.tensor.grads.grad_functions import GradAdd
from sympc.tensor.grads.grad_functions import GradFunc
from sympc.tensor.grads.grad_functions import GradMul
from sympc.tensor.grads.grad_functions import GradSigmoid
from sympc.tensor.grads.grad_functions import GradSub
from sympc.tensor.grads.grad_functions import GradSum
from sympc.tensor.grads.grad_functions import GradT


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


def test_grad_add_forward_value_exception(get_clients) -> None:
    parties = get_clients(4)
    x = torch.Tensor([[1, 2, 3], [4, 5, 6]])
    y = torch.Tensor([[1, 2, 3]])

    x_mpc = x.share(parties=parties)
    y_mpc = y.share(parties=parties)

    with pytest.raises(ValueError):
        GradAdd.forward({}, x_mpc, y_mpc)


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

    ctx = {}
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


def test_grad_sub_forward_value_exception(get_clients) -> None:
    parties = get_clients(4)
    x = torch.Tensor([[1, 2, 3], [4, 5, 6]])
    y = torch.Tensor([[1, 2, 3]])

    x_mpc = x.share(parties=parties)
    y_mpc = y.share(parties=parties)

    with pytest.raises(ValueError):
        GradSub.forward({}, x_mpc, y_mpc)


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

    ctx = {}
    res_mpc_x, res_mpc_y = GradSub.backward(ctx, grad_mpc)

    assert (res_mpc_x.reconstruct() == grad).all()
    assert (res_mpc_y.reconstruct() == grad).all()


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


def test_grad_sigmoid_bacward(get_clients) -> None:
    parties = get_clients(4)
    grad = torch.tensor([0.3, 0.4, 0.7])

    grad_mpc = grad.share(parties=parties)

    ctx = {"probabilities": grad}

    res_mpc = GradSigmoid.backward(ctx, grad_mpc)

    res = res_mpc.reconstruct()
    expected = grad * grad * (1 - grad)

    assert np.allclose(res, expected, rtol=1e-2)


def test_grad_mul_forward_value_exception(get_clients) -> None:
    parties = get_clients(4)
    x = torch.Tensor([[1, 2], [3, 4]])
    y = torch.Tensor([-1, -2])

    x_mpc = x.share(parties=parties)
    y_mpc = y.share(parties=parties)

    with pytest.raises(ValueError):
        GradMul.forward({}, x_mpc, y_mpc)


def test_grad_mul_backward_value_exception(get_clients) -> None:
    parties = get_clients(4)

    grad = torch.Tensor([1, -2, -3, 4])
    x = torch.Tensor([[1, 2], [3, -4]])
    y = torch.Tensor([[1, -4], [8, 9]])

    x_mpc = x.share(parties=parties)
    y_mpc = y.share(parties=parties)

    grad_mpc = grad.share(parties=parties)
    ctx = {"x": x_mpc, "y": y_mpc}

    with pytest.raises(ValueError):
        GradMul.backward(ctx, grad_mpc)


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

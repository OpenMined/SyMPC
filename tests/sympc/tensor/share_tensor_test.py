# stdlib
import operator
from uuid import uuid4

# third party
import numpy as np
import pytest
import torch

from sympc.config import Config
from sympc.tensor import ShareTensor


@pytest.mark.parametrize("precision", [12, 3])
@pytest.mark.parametrize("base", [4, 6])
def test_send_get(get_clients, precision, base) -> None:
    x = torch.Tensor([0.122, 1.342, 4.67])
    x_share = ShareTensor(
        data=x, config=Config(encoder_precision=precision, encoder_base=base)
    )
    client = get_clients(1)[0]
    x_ptr = x_share.send(client)

    assert x_share == x_ptr.get()


def test_different_session_ids() -> None:
    x_share = ShareTensor(data=5, session_uuid=uuid4())
    y_share = ShareTensor(data=5, session_uuid=uuid4())

    # Different session ids
    assert x_share != y_share


def test_same_session_id_and_data() -> None:

    session_id = uuid4()
    x_share = ShareTensor(data=5, session_uuid=session_id)
    y_share = ShareTensor(data=6, session_uuid=session_id)

    # Different session ids
    assert x_share != y_share


@pytest.mark.parametrize("op_str", ["add", "sub", "mul", "matmul"])
@pytest.mark.parametrize("base, precision", [(2, 16), (2, 17), (10, 3), (10, 4)])
def test_ops_share_share_local(op_str, precision, base) -> None:
    op = getattr(operator, op_str)

    x = torch.Tensor([[0.125, -1.25], [-4.25, 4]])
    y = torch.Tensor([[4.5, -2.5], [5, 2.25]])

    x_share = ShareTensor(
        data=x, config=Config(encoder_base=base, encoder_precision=precision)
    )
    y_share = ShareTensor(
        data=y, config=Config(encoder_base=base, encoder_precision=precision)
    )

    expected_res = op(x, y)
    res = op(x_share, y_share)
    tensor_decoded = res.fp_encoder.decode(res.tensor)

    assert np.allclose(tensor_decoded, expected_res, rtol=base ** -precision)


@pytest.mark.parametrize("op_str", ["add", "sub", "mul", "matmul"])
@pytest.mark.parametrize("base, precision", [(2, 16), (2, 17), (10, 3), (10, 4)])
def test_ops_share_tensor_local(op_str, precision, base) -> None:
    op = getattr(operator, op_str)

    x = torch.Tensor([[0.125, -1.25], [-4.25, 4]])
    y = torch.Tensor([[4.5, -2.5], [5, 2.25]])

    x_share = ShareTensor(
        data=x, config=Config(encoder_base=base, encoder_precision=precision)
    )

    expected_res = op(x, y)
    res = op(x_share, y)
    tensor_decoded = res.fp_encoder.decode(res.tensor)

    assert np.allclose(tensor_decoded, expected_res, rtol=base ** -precision)


@pytest.mark.parametrize("op_str", ["add", "sub", "mul", "matmul"])
@pytest.mark.parametrize("base, precision", [(2, 16), (2, 17), (10, 3), (10, 4)])
def test_reverse_ops_share_tensor_local(op_str, precision, base) -> None:
    op = getattr(operator, op_str)

    x = torch.Tensor([[0.125, -1.25], [-4.25, 4]])
    y = torch.Tensor([[4.5, -2.5], [5, 2.25]])

    x_share = ShareTensor(
        data=x, config=Config(encoder_base=base, encoder_precision=precision)
    )

    expected_res = op(y, x)
    res = op(y, x_share)
    tensor_decoded = res.fp_encoder.decode(res.tensor)

    assert np.allclose(tensor_decoded, expected_res, rtol=base ** -precision)


def test_invalid_op_exception() -> None:

    op = getattr(operator, "truediv")

    x = torch.Tensor([[0.125, -1.25], [-4.25, 4]])
    y = torch.Tensor([[4.5, -2.5], [5, 2.25]])

    x_share = ShareTensor(data=x)

    with pytest.raises(TypeError):
        op(y, x_share)


def test_div_with_float_exception() -> None:

    x = torch.Tensor([[0.125, -1.25], [-4.25, 4]])

    x_share = ShareTensor(data=x)

    with pytest.raises(ValueError):
        x_share / 5.3


@pytest.mark.parametrize("op_str", ["add", "sub", "mul"])
@pytest.mark.parametrize("base, precision", [(2, 16), (2, 17), (10, 3), (10, 4)])
def test_ops_share_integer_local(op_str, precision, base) -> None:
    op = getattr(operator, op_str)

    x = torch.Tensor([0.125, -1.25, 4.25, -4.25, 4])
    y = 4

    x_share = ShareTensor(
        data=x, config=Config(encoder_base=base, encoder_precision=precision)
    )

    expected_res = op(x, y)
    res = op(x_share, y)
    tensor_decoded = res.fp_encoder.decode(res.tensor)

    assert np.allclose(tensor_decoded, expected_res, rtol=base ** -precision)


@pytest.mark.parametrize("op_str", ["lt", "gt"])
@pytest.mark.parametrize("base, precision", [(2, 16), (2, 17), (10, 3), (10, 4)])
def test_ineq_share_share_local(op_str, precision, base) -> None:
    op = getattr(operator, op_str)

    x = torch.Tensor([[0.125, -1.25], [-4.25, 4]])
    y = torch.Tensor([[4.5, -2.5], [5, 2.25]])

    x_share = ShareTensor(
        data=x, config=Config(encoder_base=base, encoder_precision=precision)
    )
    y_share = ShareTensor(
        data=y, config=Config(encoder_base=base, encoder_precision=precision)
    )

    expected_res = op(x, y)
    res = op(x_share, y_share)

    assert (res == expected_res).all()


@pytest.mark.parametrize("op_str", ["lt", "gt"])
@pytest.mark.parametrize("base, precision", [(2, 16), (2, 17), (10, 3), (10, 4)])
def test_ineq_share_tensor_local(op_str, precision, base) -> None:
    op = getattr(operator, op_str)

    x = torch.Tensor([[0.125, -1.25], [-4.25, 4]])
    y = torch.Tensor([[4.5, -2.5], [5, 2.25]])

    x_share = ShareTensor(
        data=x, config=Config(encoder_base=base, encoder_precision=precision)
    )

    expected_res = op(x, y)
    res = op(x_share, y)

    assert (res == expected_res).all()


def test_share_print() -> None:

    x = torch.Tensor([5.0])
    x_share = ShareTensor(data=x)

    encoded_x = x_share.fp_encoder.encode(x)

    expected = "[ShareTensor]"
    expected = f"{expected}\n\t| Session UUID: None"
    expected = f"{expected}\n\t| {x_share.fp_encoder}"
    expected = f"{expected}\n\t| Data: {encoded_x}"

    assert expected == x_share.__str__()


def test_share_repr() -> None:

    x = torch.Tensor([5.0])
    x_share = ShareTensor(data=x)

    encoded_x = x_share.fp_encoder.encode(x)

    expected = "[ShareTensor]"
    expected = f"{expected}\n\t| Session UUID: None"
    expected = f"{expected}\n\t| {x_share.fp_encoder}"
    expected = f"{expected}\n\t| Data: {encoded_x}"

    assert expected == x_share.__str__() == x_share.__repr__()


def test_share_decode() -> None:

    x = torch.Tensor([5.0])
    x_share = ShareTensor(data=x)

    assert x == x_share.decode()

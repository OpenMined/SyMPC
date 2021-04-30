# stdlib
import operator

# third party
import numpy as np
import pytest
import torch

from sympc.session import Session
from sympc.session import SessionManager
from sympc.tensor import MPCTensor
from sympc.tensor import ShareTensor


def test_mpc_tensor_exception(get_clients) -> None:
    alice_client, bob_client = get_clients(2)
    session = Session(parties=[alice_client, bob_client])

    with pytest.raises(ValueError):
        MPCTensor(secret=42, session=session)

    with pytest.raises(ValueError):
        _ = MPCTensor(secret=torch.Tensor([1, -2]), session=session)


def test_reconstruct(get_clients) -> None:
    clients = get_clients(2)
    session = Session(parties=clients)
    SessionManager.setup_mpc(session)

    a_rand = 3
    a = ShareTensor(data=a_rand, encoder_precision=0)
    _ = MPCTensor.generate_shares(secret=a, nr_parties=2, tensor_type=torch.long)

    _ = MPCTensor.generate_shares(secret=a_rand, nr_parties=2, tensor_type=torch.long)

    x_secret = torch.Tensor([1, -2, 3.0907, -4.870])
    x = MPCTensor(secret=x_secret, session=session)
    x = x.reconstruct()

    assert np.allclose(x_secret, x)


def test_op_mpc_different_sessions(get_clients) -> None:
    clients = get_clients(2)
    session_one = Session(parties=clients)
    session_two = Session(parties=clients)
    SessionManager.setup_mpc(session_one)
    SessionManager.setup_mpc(session_two)

    x = MPCTensor(secret=torch.Tensor([1, -2]), session=session_one)
    y = MPCTensor(secret=torch.Tensor([1, -2]), session=session_two)

    with pytest.raises(ValueError):
        _ = x + y


def test_remote_mpc_no_shape(get_clients) -> None:
    alice_client, bob_client = get_clients(2)
    session = Session(parties=[alice_client, bob_client])
    SessionManager.setup_mpc(session)

    x_remote = alice_client.torch.Tensor([1, -2, 0.3])

    with pytest.raises(ValueError):
        _ = MPCTensor(secret=x_remote, session=session)


def test_remote_mpc_with_shape(get_clients) -> None:
    alice_client, bob_client = get_clients(2)
    session = Session(parties=[alice_client, bob_client])
    SessionManager.setup_mpc(session)

    x_remote = alice_client.torch.Tensor([1, -2, 0.3])
    x = MPCTensor(secret=x_remote, shape=(1, 3), session=session)
    result = x.reconstruct()

    assert np.allclose(x_remote.get(), result, atol=1e-5)


def test_remote_not_tensor(get_clients) -> None:
    alice_client, bob_client = get_clients(2)
    session = Session(parties=[alice_client, bob_client])
    SessionManager.setup_mpc(session)

    x_remote_int = bob_client.python.Int(5)
    x = MPCTensor(secret=x_remote_int, shape=(1,), session=session)
    result = x.reconstruct()

    assert x_remote_int == result

    x_remote_int = bob_client.python.Float(5.4)
    x = MPCTensor(secret=x_remote_int, shape=(1,), session=session)
    result = x.reconstruct()

    assert np.allclose(x_remote_int.get(), result, atol=1e-5)


def test_local_secret_not_tensor(get_clients) -> None:
    alice_client, bob_client = get_clients(2)
    session = Session(parties=[alice_client, bob_client])
    SessionManager.setup_mpc(session)

    x_int = 5
    x = MPCTensor(secret=x_int, session=session)
    result = x.reconstruct()

    assert x_int == result

    x_float = 5.987
    x = MPCTensor(secret=x_float, session=session)
    result = x.reconstruct()

    assert np.allclose(torch.tensor(x_float), result)


@pytest.mark.parametrize("nr_clients", [2, 3, 4, 5])
@pytest.mark.parametrize("op_str", ["mul", "matmul"])
def test_ops_mpc_mpc(get_clients, nr_clients, op_str) -> None:
    clients = get_clients(nr_clients)
    session = Session(parties=clients)
    SessionManager.setup_mpc(session)

    op = getattr(operator, op_str)

    x_secret = torch.Tensor([[0.125, -1.25], [-4.25, 4]])
    y_secret = torch.Tensor([[4.5, -2.5], [5, 2.25]])
    x = MPCTensor(secret=x_secret, session=session)
    y = MPCTensor(secret=y_secret, session=session)
    result = op(x, y).reconstruct()
    expected_result = op(x_secret, y_secret)

    assert np.allclose(result, expected_result, rtol=10e-4)


@pytest.mark.parametrize("nr_clients", [2])
@pytest.mark.parametrize("bias", [None, torch.ones(1)])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0, 1])
def test_conv2d_mpc_mpc(get_clients, nr_clients, bias, stride, padding) -> None:
    clients = get_clients(nr_clients)
    session = Session(parties=clients)
    SessionManager.setup_mpc(session)

    input_secret = torch.ones(1, 1, 4, 4)
    weight_secret = torch.ones(1, 1, 2, 2)
    input = MPCTensor(secret=input_secret, session=session)
    weight = MPCTensor(secret=weight_secret, session=session)

    kwargs = {"bias": bias, "stride": stride, "padding": padding}

    result = input.conv2d(weight, **kwargs).reconstruct()
    expected_result = torch.nn.functional.conv2d(input_secret, weight_secret, **kwargs)

    assert np.allclose(result, expected_result, rtol=10e-4)


@pytest.mark.parametrize("nr_clients", [2, 3, 4, 5])
@pytest.mark.parametrize("op_str", ["mul", "matmul", "truediv"])
def test_ops_mpc_public(get_clients, nr_clients, op_str) -> None:
    clients = get_clients(nr_clients)
    session = Session(parties=clients)
    SessionManager.setup_mpc(session)

    x_secret = torch.Tensor([[0.125, -1.25], [-4.25, 4]])

    if op_str == "truediv":
        y_secret = torch.Tensor([[2, 3], [4, 5]]).long()
    else:
        y_secret = torch.Tensor([[4.5, -2.5], [5, 2.25]])
    x = MPCTensor(secret=x_secret, session=session)

    op = getattr(operator, op_str)
    expected_result = op(x_secret, y_secret)
    result = op(x, y_secret).reconstruct()
    assert np.allclose(result, expected_result, atol=10e-4)


@pytest.mark.parametrize("nr_clients", [2, 3, 4, 5])
@pytest.mark.parametrize("op_str", ["add", "sub", "mul", "matmul"])
def test_ops_public_mpc(get_clients, nr_clients, op_str) -> None:
    clients = get_clients(nr_clients)
    session = Session(parties=clients)
    SessionManager.setup_mpc(session)

    op = getattr(operator, op_str)

    x_secret = torch.Tensor([[0.125, -1.25], [-4.25, 4]])
    y_secret = torch.Tensor([[4.5, -2.5], [5, 2.25]])
    x = MPCTensor(secret=x_secret, session=session)

    expected_result = op(y_secret, x_secret)
    result = op(y_secret, x).reconstruct()

    assert np.allclose(result, expected_result, atol=10e-4)


@pytest.mark.parametrize("nr_clients", [2, 3, 4, 5])
@pytest.mark.parametrize("op_str", ["add", "sub", "mul", "truediv"])
def test_ops_integer(get_clients, nr_clients, op_str) -> None:
    clients = get_clients(nr_clients)
    session = Session(parties=clients)
    SessionManager.setup_mpc(session)

    op = getattr(operator, op_str)

    x_secret = torch.Tensor([0.125, -1.25, -4.25, 4])
    y = 4

    x = MPCTensor(secret=x_secret, session=session)

    expected_result = op(x_secret, y)
    result = op(x, y).reconstruct()

    assert np.allclose(result, expected_result, atol=10e-3)


def test_mpc_print(get_clients) -> None:
    clients = get_clients(2)
    session = Session(parties=clients)
    SessionManager.setup_mpc(session)

    x_secret = torch.Tensor([5.0])

    x = MPCTensor(secret=x_secret, session=session)

    expected = f"[MPCTensor]\nShape: {x_secret.shape}\n\t|"
    expected = (
        f"{expected} <VirtualMachineClient: P_0 Client> -> ShareTensorPointer\n\t|"
    )
    expected = f"{expected} <VirtualMachineClient: P_1 Client> -> ShareTensorPointer"

    assert expected == x.__str__()
    assert x.__str__() == x.__str__()


def test_generate_shares() -> None:

    precision = 12
    base = 4

    x_secret = torch.Tensor([5.0])

    # test with default values
    x_share = ShareTensor(data=x_secret)

    shares_from_share_tensor = MPCTensor.generate_shares(x_share, 2)
    shares_from_secret = MPCTensor.generate_shares(x_secret, 2)

    assert sum(shares_from_share_tensor).tensor == sum(shares_from_secret).tensor

    x_share = ShareTensor(data=x_secret, encoder_precision=precision, encoder_base=base)

    shares_from_share_tensor = MPCTensor.generate_shares(x_share, 2)
    shares_from_secret = MPCTensor.generate_shares(
        x_secret, 2, encoder_precision=precision, encoder_base=base
    )

    assert sum(shares_from_share_tensor).tensor == sum(shares_from_secret).tensor


def test_generate_shares_session(get_clients) -> None:
    clients = get_clients(2)
    session = Session(parties=clients)
    SessionManager.setup_mpc(session)

    x_secret = torch.Tensor([5.0])
    x_share = ShareTensor(data=x_secret, session=session)

    shares_from_share_tensor = MPCTensor.generate_shares(x_share, 2)
    shares_from_secret = MPCTensor.generate_shares(x_secret, 2, session=session)

    assert sum(shares_from_share_tensor) == sum(shares_from_secret)


@pytest.mark.parametrize("protocol", ["FSS"])
@pytest.mark.parametrize("op_str", ["le", "lt", "ge", "gt"])
def test_comparison_mpc_mpc(get_clients, protocol, op_str) -> None:
    clients = get_clients(2)
    session = Session(parties=clients, protocol=protocol)
    SessionManager.setup_mpc(session)

    op = getattr(operator, op_str)

    x_secret = torch.Tensor([[0.125, -1.25], [-4.25, 4], [-3, 3]])
    y_secret = torch.Tensor([[4.5, -2.5], [5, 2.25], [-3, 3]])
    x = MPCTensor(secret=x_secret, session=session)
    y = MPCTensor(secret=y_secret, session=session)
    result = op(x, y).reconstruct()
    expected_result = op(x_secret, y_secret)

    assert (result == expected_result).all()


@pytest.mark.parametrize("protocol", ["FSS"])
@pytest.mark.parametrize("op_str", ["eq", "ne"])
def test_equality_mpc_mpc(get_clients, protocol, op_str) -> None:
    clients = get_clients(2)
    session = Session(parties=clients, protocol=protocol)
    SessionManager.setup_mpc(session)

    op = getattr(operator, op_str)

    x_secret = torch.Tensor([[0.125, -2.5], [-4.25, 2.25]])
    y_secret = torch.Tensor([[4.5, -2.5], [5, 2.25]])
    x = MPCTensor(secret=x_secret, session=session)
    y = MPCTensor(secret=y_secret, session=session)
    result = op(x, y).reconstruct()
    expected_result = op(x_secret, y_secret)

    assert (result == expected_result).all()


@pytest.mark.parametrize("protocol", ["FSS"])
@pytest.mark.parametrize("op_str", ["le", "lt", "ge", "gt", "eq", "ne"])
def test_comp_mpc_public(get_clients, protocol, op_str) -> None:
    clients = get_clients(2)
    session = Session(parties=clients, protocol=protocol)
    SessionManager.setup_mpc(session)

    op = getattr(operator, op_str)

    x_secret = torch.Tensor([[0.125, -2.5], [-4.25, 2.25]])
    y_secret = 2.5
    x = MPCTensor(secret=x_secret, session=session)
    result = op(x, y_secret).reconstruct()
    expected_result = op(x_secret, y_secret)

    assert (result == expected_result).all()


@pytest.mark.parametrize("protocol", ["FSS"])
@pytest.mark.parametrize("op_str", ["le", "lt", "ge", "gt", "eq", "ne"])
def test_comp_public_mpc(get_clients, protocol, op_str) -> None:
    clients = get_clients(2)
    session = Session(parties=clients, protocol=protocol)
    SessionManager.setup_mpc(session)

    op = getattr(operator, op_str)

    x_secret = 2.5
    y_secret = torch.Tensor([[0.125, -2.5], [-4.25, 2.25]])
    y = MPCTensor(secret=y_secret, session=session)
    result = op(x_secret, y).reconstruct()
    expected_result = op(x_secret, y_secret)

    assert (result == expected_result).all()


def test_share_get_method_parties(get_clients) -> None:
    clients = get_clients(2)

    x_secret = torch.Tensor([1.0, 2.0, 5.0])
    expected_res = x_secret * x_secret

    mpc_tensor = x_secret.share(parties=clients)
    res = mpc_tensor * mpc_tensor

    assert all(res.get() == expected_res)


def test_share_get_method_parties_exception(get_clients) -> None:
    clients = get_clients(4)

    x_secret = torch.Tensor([1.0, 2.0, 5.0])
    _ = x_secret * x_secret

    mpc_tensor1 = x_secret.share(parties=clients[:2])
    mpc_tensor2 = x_secret.share(parties=clients[2:])

    with pytest.raises(ValueError):
        _ = mpc_tensor1 * mpc_tensor2


def test_share_get_method_session(get_clients) -> None:
    clients = get_clients(2)
    session = Session(parties=clients)
    SessionManager.setup_mpc(session)

    x_secret = torch.Tensor([5.0, 6.0, 7.0])
    expected_res = x_secret * x_secret

    mpc_tensor = x_secret.share(session=session)
    res = mpc_tensor * mpc_tensor

    assert all(res.get() == expected_res)


@pytest.mark.parametrize("power", [4, 7])
def test_pow(get_clients, power) -> None:
    clients = get_clients(2)
    session = Session(parties=clients)
    SessionManager.setup_mpc(session)

    x_secret = torch.Tensor([5.0, -3])
    x = MPCTensor(secret=x_secret, session=session)

    power_secret = x_secret ** power
    power = x ** power

    assert torch.allclose(power_secret, power.reconstruct())

    with pytest.raises(RuntimeError):
        power = x ** -2

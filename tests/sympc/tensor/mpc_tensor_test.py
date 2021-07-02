# stdlib
import operator

# third party
import numpy as np
import pytest
import torch

from sympc.config import Config
from sympc.protocol.protocol import Protocol
from sympc.session import Session
from sympc.session import SessionManager
from sympc.tensor import MPCTensor
from sympc.tensor import ShareTensor


def test_setupmpc_nocall_exception(get_clients) -> None:
    alice_client, bob_client = get_clients(2)
    session = Session(parties=[alice_client, bob_client])

    with pytest.raises(ValueError):
        MPCTensor(secret=42, session=session)

    with pytest.raises(ValueError):
        MPCTensor(secret=torch.Tensor([1, -2]), session=session)


def test_mpc_share_nosession_exception() -> None:
    secret = torch.Tensor([[0.1, -1], [-4, 4]])

    with pytest.raises(ValueError):
        secret.share()


def test_reconstruct(get_clients) -> None:
    clients = get_clients(2)
    session = Session(parties=clients)
    SessionManager.setup_mpc(session)

    a_rand = 3
    a = ShareTensor(data=a_rand, config=Config(encoder_precision=0))
    MPCTensor.generate_shares(secret=a, nr_parties=2, tensor_type=torch.long)

    MPCTensor.generate_shares(
        secret=a_rand, nr_parties=2, config=Config(), tensor_type=torch.long
    )

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
        x + y


def test_remote_mpc_no_shape(get_clients) -> None:
    alice_client, bob_client = get_clients(2)
    session = Session(parties=[alice_client, bob_client])
    SessionManager.setup_mpc(session)

    x_remote = alice_client.torch.Tensor([1, -2, 0.3])

    with pytest.raises(ValueError):
        MPCTensor(secret=x_remote, session=session)


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


@pytest.mark.parametrize("nr_clients", [2, 3, 5])
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
@pytest.mark.parametrize("op_str", ["truediv"])
def test_ops_mpc_mpc_div(get_clients, nr_clients, op_str) -> None:
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


@pytest.mark.parametrize("nr_clients", [2, 3, 5])
@pytest.mark.parametrize("bias", [None, torch.ones(1)])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("op_str", ["conv2d", "conv_transpose2d"])
def test_conv_mpc_mpc(get_clients, nr_clients, bias, stride, padding, op_str) -> None:
    clients = get_clients(nr_clients)
    session = Session(parties=clients)
    SessionManager.setup_mpc(session)

    input_secret = torch.ones(1, 1, 4, 4)
    weight_secret = torch.ones(1, 1, 2, 2)
    input = MPCTensor(secret=input_secret, session=session)
    weight = MPCTensor(secret=weight_secret, session=session)

    kwargs = {"bias": bias, "stride": stride, "padding": padding}

    op = getattr(MPCTensor, op_str)
    result = op(input, weight, **kwargs).reconstruct()
    op = getattr(torch.nn.functional, op_str)
    expected_result = op(input_secret, weight_secret, **kwargs)

    assert np.allclose(result, expected_result, rtol=10e-4)


@pytest.mark.parametrize("nr_clients", [2, 3, 5])
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
    result = op(x, y_secret)
    result = result.reconstruct()
    assert np.allclose(result, expected_result, atol=10e-4)


@pytest.mark.parametrize("nr_parties", [3, 5])
def test_ops_divfloat_exception(get_clients, nr_parties) -> None:
    # Define the virtual machines that would be use in the computation
    parties = get_clients(nr_parties)

    # Setup the session for the computation
    session = Session(parties=parties)
    SessionManager.setup_mpc(session)

    x_secret = torch.Tensor([[0.1, -1], [-4, 4]])
    y_secret = torch.Tensor([[4.0, -2.5], [5, 2]])

    # 3. Share the secret building an MPCTensor
    x = MPCTensor(secret=x_secret, session=session)
    y = MPCTensor(secret=y_secret, session=session)

    with pytest.raises(ValueError):
        x / y


@pytest.mark.parametrize("nr_clients", [2, 3, 5])
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


falcon = Protocol.registered_protocols["Falcon"]()


@pytest.mark.parametrize("nr_clients", [2, 3, 5])
@pytest.mark.parametrize("op_str", ["add", "sub"])
def test_ops_public_tensor_rst(get_clients, nr_clients, op_str) -> None:
    clients = get_clients(nr_clients)
    session = Session(parties=clients, protocol=falcon)
    SessionManager.setup_mpc(session)

    op = getattr(operator, op_str)

    x_secret = torch.Tensor([[0.125, -1.25], [-4.25, 4]])
    y_public = torch.Tensor([[4.5, -2.5], [5, 2.25]])
    x = MPCTensor(secret=x_secret, session=session)

    expected_result = op(x_secret, y_public)
    result = op(x, y_public).reconstruct()

    assert np.allclose(result, expected_result, atol=10e-4)


@pytest.mark.parametrize("op_str", ["mul"])  # matmul to be added
def test_ops_mpc_private_rst_mul(get_clients, op_str) -> None:
    clients = get_clients(3)
    session = Session(parties=clients)
    SessionManager.setup_mpc(session)

    op = getattr(operator, op_str)

    x_secret = torch.Tensor([[0.125, -1.25], [-4.25, 4]])
    y_secret = torch.Tensor([[4.5, -2.5], [5, 2.25]])
    x = MPCTensor(secret=x_secret, session=session)
    y = MPCTensor(secret=y_secret, session=session)
    result = op(x, y).reconstruct()
    expected_result = op(x_secret, y_secret)

    assert np.allclose(result, expected_result, rtol=10e-3)


@pytest.mark.parametrize("nr_clients", [2, 3, 5])
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


@pytest.mark.parametrize("nr_clients", [2, 3, 5])
@pytest.mark.parametrize("op_str", ["add", "sub"])
def test_ops_public_integer_rst(get_clients, nr_clients, op_str) -> None:
    clients = get_clients(nr_clients)
    session = Session(parties=clients, protocol=falcon)
    SessionManager.setup_mpc(session)

    op = getattr(operator, op_str)

    x_secret = torch.Tensor([0.125, -1.25, -4.25, 4])
    y = 4

    x = MPCTensor(secret=x_secret, session=session)

    expected_result = op(x_secret, y)
    result = op(x, y).reconstruct()

    assert np.allclose(result, expected_result, atol=10e-3)


@pytest.mark.parametrize("nr_clients", [2, 3, 5])
@pytest.mark.parametrize("op_str", ["add", "sub"])
def test_ops_mpc_private_rst(get_clients, nr_clients, op_str) -> None:
    clients = get_clients(nr_clients)
    session = Session(parties=clients, protocol=falcon)
    SessionManager.setup_mpc(session)

    op = getattr(operator, op_str)

    x_secret = torch.Tensor([[0.125, -1.25], [-4.25, 4]])
    y_secret = torch.Tensor([[4.5, -2.5], [5, 2.25]])
    x = MPCTensor(secret=x_secret, session=session)
    y = MPCTensor(secret=y_secret, session=session)
    expected_result = op(x_secret, y_secret)
    result = op(x, y).reconstruct()

    assert np.allclose(result, expected_result, atol=10e-4)


def test_mpc_len(get_clients) -> None:
    clients = get_clients(2)
    session = Session(parties=clients)
    SessionManager.setup_mpc(session)

    for dim in range(1, 5):
        x_secret = torch.arange(-6, 6).view(dim, -1)
        x = MPCTensor(secret=x_secret, session=session)
        assert len(x_secret) == len(x)


def test_mpc_print(get_clients) -> None:
    clients = get_clients(2)
    session = Session(parties=clients)
    SessionManager.setup_mpc(session)

    x_secret = torch.Tensor([5.0])

    x = MPCTensor(secret=x_secret, session=session)

    expected = f"[MPCTensor]\nShape: {x_secret.shape}\nRequires Grad: False\n\t|"
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

    shares_from_share_tensor = MPCTensor.generate_shares(x_share, nr_parties=2)
    shares_from_secret = MPCTensor.generate_shares(
        x_secret, nr_parties=2, config=Config()
    )

    assert sum(shares_from_share_tensor).tensor == sum(shares_from_secret).tensor

    x_share = ShareTensor(
        data=x_secret, config=Config(encoder_precision=precision, encoder_base=base)
    )

    shares_from_share_tensor = MPCTensor.generate_shares(x_share, 2)
    shares_from_secret = MPCTensor.generate_shares(
        x_secret, 2, config=Config(encoder_precision=precision, encoder_base=base)
    )

    assert sum(shares_from_share_tensor).tensor == sum(shares_from_secret).tensor


def test_generate_shares_config(get_clients) -> None:
    x_secret = torch.Tensor([5.0])
    x_share = ShareTensor(data=x_secret)

    shares_from_share_tensor = MPCTensor.generate_shares(x_share, 2)
    shares_from_secret = MPCTensor.generate_shares(
        x_secret, 2, config=Config(encoder_base=2, encoder_precision=16)
    )

    assert sum(shares_from_share_tensor) == sum(shares_from_secret)


fss_protocol = Protocol.registered_protocols["FSS"]()


@pytest.mark.parametrize("protocol", [fss_protocol])
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


@pytest.mark.parametrize("protocol", [fss_protocol])
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


@pytest.mark.parametrize("protocol", [fss_protocol])
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


@pytest.mark.parametrize("protocol", [fss_protocol])
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
    x_secret * x_secret

    mpc_tensor1 = x_secret.share(parties=clients[:2])
    mpc_tensor2 = x_secret.share(parties=clients[2:])

    with pytest.raises(ValueError):
        mpc_tensor1 * mpc_tensor2


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


def test_backward(get_clients):
    clients = get_clients(4)
    session = Session(parties=clients)
    session.autograd_active = True
    SessionManager.setup_mpc(session)

    x_secret = torch.tensor([[0.125, -1.25], [-4.25, 4], [-3, 3]], requires_grad=True)
    y_secret = torch.tensor([[4.5, -2.5], [5, 2.25], [-3, 3]], requires_grad=True)
    x = MPCTensor(secret=x_secret, session=session, requires_grad=True)
    y = MPCTensor(secret=y_secret, session=session, requires_grad=True)

    res_mpc = x * y
    res = x_secret * y_secret
    s_mpc = res_mpc.sum()
    s = torch.sum(res)
    s_mpc.backward()
    s.backward()

    assert np.allclose(x.grad.get(), x_secret.grad, rtol=1e-3)
    assert np.allclose(y.grad.get(), y_secret.grad, rtol=1e-3)


def test_backward_without_requires_grad(get_clients):
    clients = get_clients(4)
    session = Session(parties=clients)
    session.autograd_active = True
    SessionManager.setup_mpc(session)

    x_secret = torch.tensor([[0.125, -1.25], [-4.25, 4], [-3, 3]])
    y_secret = torch.tensor([[4.5, -2.5], [5, 2.25], [-3, 3]])
    x = MPCTensor(secret=x_secret, session=session)
    y = MPCTensor(secret=y_secret, session=session)

    res_mpc = x - y
    s_mpc = res_mpc.sum()
    s_mpc.backward()

    assert not res_mpc.requires_grad
    assert res_mpc.grad is None
    assert x.grad is None
    assert y.grad is None


def test_backward_with_one_requires_grad(get_clients):
    clients = get_clients(4)
    session = Session(parties=clients)
    session.autograd_active = True
    SessionManager.setup_mpc(session)

    x_secret = torch.tensor([[0.125, -1.25], [-4.25, 4], [-3, 3]], requires_grad=True)
    y_secret = torch.tensor([[4.5, -2.5], [5, 2.25], [-3, 3]])
    x = MPCTensor(secret=x_secret, session=session, requires_grad=True)
    y = MPCTensor(secret=y_secret, session=session)

    res_mpc = x - y
    res = x_secret - y_secret
    s_mpc = res_mpc.sum()
    s = torch.sum(res)
    s_mpc.backward()
    s.backward()

    # TODO: add assert for res_mpc.grad and res.grad
    assert res_mpc.requires_grad
    assert np.allclose(x.grad.get(), x_secret.grad, rtol=1e-3)
    assert y.grad is None


def test_invalid_share_class(get_clients) -> None:
    clients = get_clients(2)
    session = Session(parties=clients)
    SessionManager.setup_mpc(session)
    x = torch.tensor([1, 2, 3])
    x_s = MPCTensor(secret=x, session=session)
    x_s.session.protocol.share_class = "invalid"
    with pytest.raises(TypeError):
        x_s + x


def test_ops_different_share_class(get_clients) -> None:
    clients = get_clients(2)
    session1 = Session(parties=clients)
    session2 = Session(parties=clients, protocol=falcon)
    SessionManager.setup_mpc(session1)
    SessionManager.setup_mpc(session2)
    x = torch.tensor([1, 2, 3])
    x_share = MPCTensor(secret=x, session=session1)
    x_rst = MPCTensor(secret=x, session=session2)
    with pytest.raises(TypeError):
        x_share + x_rst


def test_get_shape_none() -> None:
    with pytest.raises(ValueError):
        MPCTensor._get_shape("mul", None, None)

# third party
import pytest

from sympc.protocol import Protocol
from sympc.protocol.beaver.beaver import conv2d_store_get
from sympc.protocol.beaver.beaver import matmul_store_get
from sympc.protocol.beaver.beaver import mul_store_get
from sympc.protocol.fss.fss import _generate_primitive
from sympc.protocol.fss.fss import get_primitive
from sympc.store.exceptions import EmptyPrimitiveStore
from sympc.tensor import ShareTensor


class TestProtocol(metaclass=Protocol):
    share_class = ShareTensor
    security_levels = ["semi-honest", "malicious", "covert"]


def test_register_protocol() -> None:
    assert "TestProtocol" in Protocol.registered_protocols
    assert Protocol.registered_protocols["TestProtocol"] == TestProtocol


def test_exception_no_share_class() -> None:
    with pytest.raises(ValueError):

        class TestProtocolShareException(metaclass=Protocol):
            security_levels = ["semi-honest"]


def test_exception_no_security_levels() -> None:
    with pytest.raises(ValueError):

        class TestProtocolSecurityException(metaclass=Protocol):
            share_class = ShareTensor


def test_protocol_class_same_name() -> None:
    class TestName(metaclass=Protocol):
        share_class = ShareTensor
        security_levels = ["semi-honest"]

    with pytest.raises(ValueError):

        class TestName(metaclass=Protocol):
            share_class = ShareTensor
            security_levels = ["semi-honest"]


def test_exception_unsupported_fss_operation():
    with pytest.raises(ValueError):
        _generate_primitive("relu", 10)


def test_exception_insufficient_fss_primitives():
    store = {}
    store["fss_eq"] = [[10]]
    with pytest.raises(ValueError):
        get_primitive(store, 10)


@pytest.mark.parametrize(
    "function", [mul_store_get, matmul_store_get, conv2d_store_get]
)
def test_exception_empty_primitive_store_no_key(function):
    """Exception when there exists no key in store."""
    a_shape = (2, 2)
    b_shape = (2, 2)
    store = {}

    with pytest.raises(EmptyPrimitiveStore):
        function(store, a_shape, b_shape)


@pytest.mark.parametrize(
    "function", [mul_store_get, matmul_store_get, conv2d_store_get]
)
def test_exception_empty_primitive_store_no_primitive(function):
    """Exception when there exists a key in store but has no primitive."""
    a_shape = (2, 2)
    b_shape = (2, 2)

    store = {
        "beaver_mul_(2, 2)_(2, 2)": [],
        "beaver_mat_mul_(2, 2)_(2, 2)": [],
        "beaver_conv2d_(2, 2)_(2, 2)": [],
    }

    with pytest.raises(EmptyPrimitiveStore):
        function(store, a_shape, b_shape)

    # invalid shapes

    with pytest.raises(EmptyPrimitiveStore):
        function(store, (2, 3), b_shape)

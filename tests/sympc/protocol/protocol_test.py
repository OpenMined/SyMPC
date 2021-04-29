# third party
import pytest

from sympc.protocol import Protocol
from sympc.protocol.fss.fss import _generate_primitive
from sympc.protocol.fss.fss import _get_primitive


class TestProtocol(metaclass=Protocol):
    pass


def test_register_protocol() -> None:
    assert "TestProtocol" in Protocol.registered_protocols
    assert Protocol.registered_protocols["TestProtocol"] == TestProtocol


def test_exception_unsupported_fss_operation():
    with pytest.raises(ValueError):
        _generate_primitive("relu", 10)


def test_exception_insufficient_fss_primitives():
    store = {}
    store["fss_eq"] = [[10]]
    with pytest.raises(ValueError):
        _get_primitive("fss_eq", store, 100)

from sympc.protocol import Protocol


class TestProtocol(metaclass=Protocol):
    pass


def test_register_protocol() -> None:
    print(Protocol.registered_protocols)
    assert "TestProtocol" in Protocol.registered_protocols
    assert Protocol.registered_protocols["TestProtocol"] == TestProtocol

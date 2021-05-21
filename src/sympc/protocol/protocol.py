"""Controls existing protocols."""
# stdlib
from typing import Any
from typing import Dict
from typing import List


class Protocol(type):
    """Keep traces of registered protocols.

    Attributes:
        registered_protocols (Dict[Any, Any]): Dictionary with the registered protocols.
    """

    registered_protocols: Dict[Any, Any] = {}

    def __new__(cls, name: str, bases, dct: Dict[Any, Any]):
        """Control creation of new instances.

        Args:
            name (str): Name of the protocol
            bases: asdf.
            dct (Dict[Any, Any]): Dictionary.

        Returns:
            Protocol: Defined protocol.

        Raises:
            ValueError: if the protocol we want to register does not have a 'share_class' attribute
                        if the protocol is already registered with the same name
        """
        new_cls = super().__new__(cls, name, bases, dct)

        if getattr(new_cls, "share_class", None) is None:
            raise ValueError(
                "share_class attribut should be present in the protocol class"
            )

        Protocol.registered_protocols[name] = new_cls

        def __init__(self, *args: List[Any], **kwargs: Dict[Any, Any]) -> None:  # noqa
            raise ValueError("Protocol class should not be instantiated")

        setattr(new_cls, "__init__", __init__)
        return new_cls

"""The Metaclass used for the Tensors from SyMPC.

We use this such that we can have a forward mechanism.
"""
# stdlib
from typing import Any
from typing import Dict
from typing import Tuple
from typing import Type
from typing import cast


class SyMPCTensor(type):
    """Metaclass for the MPCTensor and ShareTensors."""

    def __new__(
        cls: Type["SyMPCTensor"], name: str, bases: Tuple[Any], dic: Dict[str, Any]
    ) -> "SyMPCTensor":
        """Add new methods and properties to the defined tensor classes.

        Args:
            name (str): name of the class which has SyMPCTensor metaclass
            bases (Tuple[Any]): classes from which our class inherits
            dic (Dict[str, Any]): attributes of the class

        Returns:
            The new class with added methods

        Raises:
            ValueError: the method we try to add is already defined in the class
        """
        res = super().__new__(cls, name, bases, dic)

        forward_methods = getattr(res, "METHODS_FORWARD")
        hook_method = getattr(res, "hook_method")
        for method in forward_methods:
            if method in res.__dict__:
                raise ValueError(f"Attribute {method} already exists in {name}")
            setattr(res, method, hook_method(method))

        forward_properties = getattr(res, "PROPERTIES_FORWARD")

        hook_property = getattr(res, "hook_property")
        for prop in forward_properties:
            if prop in res.__dict__:
                raise ValueError(f"Attribute {prop} already exists in {name}")
            setattr(res, prop, hook_property(prop))

        res = cast(SyMPCTensor, res)
        return res

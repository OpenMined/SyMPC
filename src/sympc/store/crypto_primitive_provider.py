"""Crypto Primitives."""

# stdlib
import json
from typing import Any
from typing import Callable
from typing import DefaultDict
from typing import Dict
from typing import List

from sympc.session import Session


class CryptoPrimitiveProvider:
    """A trusted third party should use this class to generate crypto primitives."""

    _func_providers: Dict[str, Callable] = {}
    _logging = False
    _ops_list: DefaultDict[str, List] = DefaultDict(list)

    def __init__(self) -> None:  # noqa
        raise ValueError("This class should not be initialized")

    @staticmethod
    def generate_primitives(
        op_str: str,
        sessions: List[Any],
        g_kwargs: Dict[str, Any] = {},
        p_kwargs: Dict[str, Any] = {},
    ) -> List[Any]:
        """Generate "op_str" primitives.

        Args:
            op_str (str): Operator.
            sessions (Session): Session.
            g_kwargs: Generate kwargs passed to the registered function.
            p_kwargs: Populate kwargs passed to the registered populate function.

        Returns:
            List[Any]: List of primitives.

        Raises:
            ValueError: If op_str is not registered.

        """
        if op_str not in CryptoPrimitiveProvider._func_providers:
            raise ValueError(f"{op_str} not registered")

        generator = CryptoPrimitiveProvider._func_providers[op_str]
        primitives = generator(**g_kwargs)

        if CryptoPrimitiveProvider._logging and (
            p_kwargs.get("a_shape") and p_kwargs.get("b_shape")
        ):
            CryptoPrimitiveProvider._ops_list[op_str].append(
                {"a_shape": p_kwargs.get("a_shape"), "b_shape": p_kwargs.get("b_shape")}
            )

        if p_kwargs is not None:
            """Do not transfer the primitives if there is not specified a
            values for populate kwargs."""
            CryptoPrimitiveProvider._transfer_primitives_to_parties(
                op_str, primitives, sessions, p_kwargs
            )

        # Since we do not have (YET!) the possiblity to return typed tuples from a remote
        # execute function we are using this
        return primitives

    @staticmethod
    def _transfer_primitives_to_parties(
        op_str: str,
        primitives: List[Any],
        sessions: List[Session],
        p_kwargs: Dict[str, Any],
    ) -> None:
        if not isinstance(primitives, list):
            raise ValueError("Primitives should be a List")

        if len(primitives) != len(sessions):
            raise ValueError(
                f"Primitives Len {len(primitives)} != Sessions Len {len(sessions)}"
            )

        for primitives_party, session in zip(primitives, sessions):
            session.crypto_store.populate_store(
                op_str, primitives_party, **p_kwargs  # TODO
            )

    @staticmethod
    def get_state() -> str:
        """Get the state of a CryptoProvider.

        Returns:
            str: CryptoProvider
        """
        res = f"Providers: {list(CryptoPrimitiveProvider._func_providers.keys())}\n"
        return res

    @staticmethod
    def start_logging() -> None:
        """Sets the variable to True to start primitive logging."""
        CryptoPrimitiveProvider._logging = True

    @staticmethod
    def stop_logging() -> json:
        """Sets the varible to False to stop primitive logging.

        Returns:
            json: returns the json object containing ops details.
        """
        CryptoPrimitiveProvider._logging = False
        log_json = json.dumps(CryptoPrimitiveProvider._ops_list)
        CryptoPrimitiveProvider._ops_list.clear()
        return log_json

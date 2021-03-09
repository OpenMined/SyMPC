# stdlib
from typing import Any
from typing import Callable
from typing import Dict
from typing import List

from sympc.session import Session


class CryptoPrimitiveProvider:
    """A trusted third party should use this class to generate crypto
    primitives."""

    _func_providers: Dict[str, Callable] = {}

    def __init__(self) -> None:
        raise ValueError("This class should not be initialized")

    @staticmethod
    def generate_primitives(
        op_str: str,
        sessions: List[Any],
        n_instances: int = 1,
        g_kwargs: Dict[str, Any] = {},
        p_kwargs: Dict[str, Any] = {},
    ) -> List[Any]:
        """Generate "op_str" primitives. The "g_kwargs" (generate kwargs) are
        passed to the registered generator function The "p_kwargs" (populate
        kwargs) are passed to the registered populate function.

        :return: list of primitives
        :rtype: list of Any Type
        """

        if op_str not in CryptoPrimitiveProvider._func_providers:
            raise ValueError(f"{op_str} not registered")

        assert n_instances == 1  # nosec

        generator = CryptoPrimitiveProvider._func_providers[op_str]

        primitives = generator(**g_kwargs)

        if p_kwargs is not None:
            """Do not transfer the primitives if there is not specified a
            values for populate kwargs."""
            CryptoPrimitiveProvider._transfer_primitives_to_parties(
                op_str, primitives, sessions, p_kwargs
            )

        # TODO: "primitives_sequenatial" here represents the pointers to the primitives
        # The function "generate_primitives" should ideally not return anything

        # Since we do not have (YET!) the possiblity to return typed tuples from a remote
        # execute function we are using this
        return primitives

    @staticmethod
    def _transfer_primitives_to_parties(
        op_str: str,
        primitives: List[Any],
        sessions: List["Session"],
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
    def get_state() -> None:
        res = f"Providers: {list(CryptoPrimitiveProvider._func_providers.keys())}\n"
        return res

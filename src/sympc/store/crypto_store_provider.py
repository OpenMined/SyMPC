# stdlib
from typing import Any
from typing import Callable
from typing import Dict
from typing import List

from sympc.session import Session


class CryptoPrimitiveProvider:
    """A trusted third party should use this class to generate crypto primitives """

    _FUNC_PROVIDERS: Dict[str, Callable] = {}
    _DEFAULT_NR_INSTANCES = 10

    def __init__(self) -> None:
        raise ValueError("This class should not be initialized")

    @staticmethod
    def generate_primitives(
        op_str: str,
        sessions: int,
        n_instances: int = 1,
        g_kwargs: Dict[str, Any] = {},
        p_kwargs: Dict[str, Any] = {},
    ) -> List[Any]:
        if op_str not in CryptoPrimitiveProvider._FUNC_PROVIDERS:
            raise ValueError(f"{op_str} not registered")

        generator = CryptoPrimitiveProvider._FUNC_PROVIDERS[op_str]

        res = [generator(**g_kwargs) for _ in range(n_instances)]

        """
        Example -- for n_instances=2 and n_parties=2:
        For Beaver Triples the "res" would look like:
        res = [
            ([a0_sh_p0, a0_sh_p1], [b0_sh_p0, b0_sh_p1], [c0_sh_p0, c0_sh_p1]),
            ([a1_sh_p0, a1_sh_p1], [b1_sh_p0, b1_sh_p1], [c1_sh_p0, c1_sh_p1])
        ]

        We want to send to each party the values they should hold:
        primitives = [
            ((a0_sh_p0, b0_sh_p0, c0_sh_p0), (a1_sh_p0, b1_sh_p0, c1_sh_p0)), # (Row 0)
            ((a0_sh_p1, b0_sh_p1, c0_sh_p1), (a1_sh_p1, b1_sh_p1, c1_sh_p1))  # (Row 1)
        ]

        The first party (party 0) receives Row 0 and the second party (party 1) receives Row 1
        """
        primitives = list(zip(*map(lambda x: zip(*x), res)))

        if p_kwargs is not None:
            """
            Do not transfer the primitives if there is not
            specified a values for populate kwargs
            """
            CryptoPrimitiveProvider._transfer_primitives_to_parties(
                op_str, primitives, sessions, p_kwargs
            )

        # TODO: "res" here represents the pointers to the primitives
        # The function "generate_primitives" should ideally not return anything

        # Since we do not have (YET!) the possiblity to return typed tuples from a remote
        # execute function we are using this
        return res

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
            session.populate_crypto_store(op_str, list(primitives_party), **p_kwargs)

    @staticmethod
    def show() -> None:
        res = f"Providers: {list(CryptoPrimitiveProvider._FUNC_PROVIDERS.keys())}\n"
        res += f"Default_number_instances: {CryptoPrimitiveProvider._DEFAULT_NR_INSTANCES}\n"
        print(res)

"""Crypto Primitives."""

# stdlib
import itertools
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
    _ops_list: DefaultDict[str, List] = DefaultDict(list)
    _LOGGING = False
    _FILENAME = "primitive_log.json"

    def __init__(self) -> None:  # noqa
        raise ValueError("This class should not be initialized")

    @staticmethod
    def generate_primitives(
        op_str: str,
        session: Session,
        g_kwargs: Dict[str, Any] = {},
        p_kwargs: Dict[str, Any] = {},
    ) -> List[Any]:
        """Generate "op_str" primitives.

        Args:
            op_str (str): Operator.
            session (Session): Session.
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

        for remote_session_uuid, primitive in zip(
            session.rank_to_uuid.values(), primitives
        ):
            if not isinstance(primitive, (list, tuple)):
                primitive = (primitive,)

            for share in itertools.chain(*primitive):
                share.session_uuid = remote_session_uuid

        if CryptoPrimitiveProvider._LOGGING:
            CryptoPrimitiveProvider._ops_list[op_str].append((p_kwargs, g_kwargs))

        if p_kwargs is not None:
            """Do not transfer the primitives if there is not specified a
            values for populate kwargs."""
            CryptoPrimitiveProvider._transfer_primitives_to_parties(
                op_str, primitives, session.session_ptrs, p_kwargs
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
        CryptoPrimitiveProvider._LOGGING = True

    @staticmethod
    def stop_logging(generate_file: bool = False):
        """Sets the variable to False to stop primitive logging.

        Args:
            generate_file: when set to True generates a seperate primitive_log.json file

        Returns:
            json: returns the json object containing ops details.
        """
        CryptoPrimitiveProvider._LOGGING = False
        log = CryptoPrimitiveProvider._ops_list.copy()
        CryptoPrimitiveProvider._ops_list.clear()

        if generate_file:
            with open(CryptoPrimitiveProvider._FILENAME, "w") as f:
                f.write(json.dumps(log))
        return dict(log)

    @staticmethod
    def generate_primitive_from_dict(
        primitive_log: Dict[str, Any], session: Session
    ) -> None:
        """Generates primitives from the log provided.

        Args:
            primitive_log (dict): the dict log created with primitive logging.
            session (Session): Session.

        Raises:
            ValueError: If primitive_log is None.
        """
        if primitive_log is None:
            raise ValueError("The provided log is cannot be None")

        for op_str, args in primitive_log.items():
            for (p_kwargs, g_kwargs) in args:
                CryptoPrimitiveProvider.generate_primitives(
                    op_str=op_str,
                    session=session,
                    g_kwargs=g_kwargs,
                    p_kwargs=p_kwargs,
                )

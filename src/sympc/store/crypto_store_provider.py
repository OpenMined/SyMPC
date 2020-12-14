from typing import Callable
from store import CRYPTO_PROVIDERS


class CryptoPrimitiveProvider:

    def generate_primitives(name: str, instances: int) -> List[List[Any]]:
        if name not in CRYPTO_PROVIDER:
            raise ValueError(f"{name} not found in {CRYPTO_PROVIDER.keys()}")





    def transfer_primitives_to_parties(name, primitives: List[List[Any]], session_parties) -> None
        if len(primitives) != len(session_parties):
            raise ValueError(f"Primitives Len {len(primitives)} != Sessions Len {len(session_parties)}")

        for primitive_list, session in zip(primitives, session_parties):
            session.populate_store(primitives,


def register_provider(name):
    """Decorator to register a crypto primitive provider"""



    return f_provider

"""Package where needed crypto material is stored."""

from sympc.store.crypto_primitive_provider import CryptoPrimitiveProvider
from sympc.store.crypto_store import CryptoStore


def register_primitive_generator(name: str):
    """Decorator to register a crypto primitive provider.

    Args:
        name (str): Name of the primitive.

    # noqa: DAR201
    """

    def register_generator(func_generator):
        if name in CryptoPrimitiveProvider._func_providers:
            raise ValueError(f"Provider {name} already in _func_providers")
        CryptoPrimitiveProvider._func_providers[name] = func_generator
        return func_generator

    return register_generator


def register_primitive_store_add(name):
    """Decorator to add primitives to the store.

    Args:
        name (str): Name of the primitive.

    # noqa: DAR201
    """

    def register_add(func_add):
        if name in CryptoStore._func_add_store:
            raise ValueError(f"Crypto Store 'adder' {name} already in _func_add_store")
        CryptoStore._func_add_store[name] = func_add
        return func_add

    return register_add


def register_primitive_store_get(name):
    """Decorator to retrieve primitives from the store.

    Args:
        name (str): Name of the primitive.

    # noqa: DAR201
    """

    def register_get(func_get):
        if name in CryptoStore._func_get_store:
            raise ValueError(f"Crypto Store 'getter' {name} already in _func_get_store")
        CryptoStore._func_get_store[name] = func_get
        return func_get

    return register_get


__all__ = ["CryptoStore", "CryptoPrimitiveProvider"]

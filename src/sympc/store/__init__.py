from sympc.store.crypto_store import CryptoStore
from sympc.store.crypto_store_provider import CryptoPrimitiveProvider


def register_primitive_generator(name):
    """Decorator to register a crypto primitive provider"""

    def register_generator(func_generator):
        if name in CryptoPrimitiveProvider._FUNC_PROVIDERS:
            raise ValueError(f"Provider {name} already in _FUNC_PROVIDERS")
        CryptoPrimitiveProvider._FUNC_PROVIDERS[name] = func_generator
        return func_generator

    return register_generator


def register_primitive_store_add(name):
    """Decorator to add primitives to the store"""

    def register_add(func_add):
        if name in CryptoStore._FUNC_ADD_STORE:
            raise ValueError(f"Crypto Store 'adder' {name} already in _FUNC_ADD_STORE")
        CryptoStore._FUNC_ADD_STORE[name] = func_add
        return func_add

    return register_add


def register_primitive_store_get(name):
    """Decorator to retrieve primitives from the store"""

    def register_get(func_get):
        if name in CryptoStore._FUNC_GET_STORE:
            raise ValueError(f"Crypto Store 'getter' {name} already in _FUNC_GET_STORE")
        CryptoStore._FUNC_GET_STORE[name] = func_get
        return func_get

    return register_get


__all__ = ["CryptoStore", "CryptoPrimitiveProvider", "register_primitive_generator"]

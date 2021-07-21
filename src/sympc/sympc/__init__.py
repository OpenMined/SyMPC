__all__ = ["lib", "ffi"]

# stdlib
import os

from .ffi import ffi

lib = ffi.dlopen(os.path.join(os.path.dirname(__file__), "native.so"))
del os

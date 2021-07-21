"""Test for Maturin"""

from sympc.rust_test import RustTest


def test_rust() -> None:
    rust_test = RustTest()
    assert rust_test.test_rust() is True

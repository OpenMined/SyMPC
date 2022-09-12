"""RustTest impl."""
from sympc.sympc import lib


class RustTest:
    """Implementation of RustTest."""

    def __init__(self) -> None:
        """Init."""

    @staticmethod
    def test_rust() -> bool:
        """Test.

        Returns:
            Success
        """
        point = lib.get_origin()
        point.x = 10
        point.y = 10
        # assert cffi_pure.lib.is_in_range(point, 15)

        print("SUCCESS")
        return True

"""RustTest impl."""
# third party
import cffi_pure


class RustTest:
    """Implementation of RustTest."""

    def __init__(self) -> None:
        """Init."""

    def test_rust():
        """Test."""
        point = cffi_pure.lib.get_origin()
        point.x = 10
        point.y = 10
        # assert cffi_pure.lib.is_in_range(point, 15)

        print("SUCCESS")

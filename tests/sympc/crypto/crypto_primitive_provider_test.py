from sympc.store import CryptoPrimitiveProvider
from sympc.store import register_primitive_generator


@register_primitive_generator("test_generator")
def test_provider(n_instances):
    return 100


def test_func_primitive():

    CryptoPrimitiveProvider.show_state()
    assert False

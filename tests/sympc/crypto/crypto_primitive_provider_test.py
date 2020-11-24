from sympc.store import register_primitive_generator
from sympc.store import CryptoPrimitiveProvider


@register_primitive_generator("test_generator")
def test_provider(n_instances):
    return 100


def test_func_primitive():

    CryptoPrimitiveProvider.show_state()
    print(CryptoPrimitiProvider.generate_primitives("test_generator", 200))
    assert False

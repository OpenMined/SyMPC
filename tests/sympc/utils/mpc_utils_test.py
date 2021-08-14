from sympc.utils import get_nr_bits


def test_get_nr_bits() -> None:
    input_ring = [2 ** 64, 2 ** 32, 2 ** 16, 2 ** 8, 67, 2 ** 1]
    exp_res = [64, 32, 16, 8, 7, 1]
    for x, y in zip(input_ring, exp_res):
        assert get_nr_bits(x) == y

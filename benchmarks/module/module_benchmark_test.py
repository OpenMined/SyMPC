"""Neural networks benchmarks."""
# stdlib
from typing import Any
from typing import Callable
from typing import List

# third party
from conv_model import run_conv_model


def test_run_conv_model(benchmark, get_clients: Callable[[int], List[Any]]):
    """Benchmark simple convolutional network.

    Arguments:
        benchmark: Fixture that benchmarks any function passed
        get_clients: Fixture that returns a list of clients
    """
    benchmark(run_conv_model, get_clients)

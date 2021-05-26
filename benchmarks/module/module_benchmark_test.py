"""Neural networks benchmarks."""
# stdlib
from typing import Any
from typing import Callable
from typing import List

# third party
from conv_model import run_inference_conv_model
from conv_model import set_up_model


def test_run_inference_conv_model(benchmark, get_clients: Callable[[int], List[Any]]):
    """Benchmark inference on simple convolutional network.

    Arguments:
        benchmark: Fixture that benchmarks any function passed
        get_clients: Fixture that returns a list of clients
    """
    expected, mpc_model, x_mpc = set_up_model(get_clients)
    benchmark(run_inference_conv_model, expected, mpc_model, x_mpc)

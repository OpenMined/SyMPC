# stdlib
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Tuple

# third party
import pytest
import syft as sy
import torch

from sympc.session import Session
from sympc.session import SessionManager
from sympc.store import CryptoPrimitiveProvider
from sympc.store import register_primitive_generator
from sympc.store import register_primitive_store_add
from sympc.store import register_primitive_store_get
from sympc.tensor import MPCTensor

PRIMITIVE_NR_ELEMS = 4


class LinearNet(sy.Module):
    def __init__(self, torch_ref):
        super(LinearNet, self).__init__(torch_ref=torch_ref)
        self.fc1 = self.torch_ref.nn.Linear(3, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.torch_ref.nn.functional.relu(x)
        return x


@register_primitive_generator("test")
def provider_test(nr_parties: int, nr_instances: int) -> List[Tuple[int]]:
    """This function will generate the values:

    [((0, 0, 0, 0), (0, 0, 0, 0), ...), ((1, 1, 1, 1), (1, 1, 1, 1)),
    ...]
    """
    primitives = [
        tuple(tuple(i for _ in range(PRIMITIVE_NR_ELEMS)) for _ in range(nr_instances))
        for i in range(nr_parties)
    ]
    return primitives


@register_primitive_store_get("test")
def provider_test_get(
    store: Dict[str, List[Any]], nr_instances: int
) -> List[Tuple[int]]:

    return [store["test_key"][i] for i in range(nr_instances)]


@register_primitive_store_add("test")
def provider_test_add(
    store: Dict[str, List[Any]], primitives: Iterable[Any]
) -> List[Tuple[int]]:
    store["test_key"] = primitives


def test_exception_init() -> None:
    with pytest.raises(ValueError):
        CryptoPrimitiveProvider()


def test_generate_primitive_exception() -> None:
    with pytest.raises(ValueError):
        CryptoPrimitiveProvider.generate_primitives(op_str="SyMPC", sessions=[])


def test_transfer_primitives_type_exception() -> None:
    with pytest.raises(ValueError):
        """Primitives should be a list."""
        CryptoPrimitiveProvider._transfer_primitives_to_parties(
            op_str="test", primitives=50, sessions=[], p_kwargs={}
        )


def test_transfer_primitives_mismatch_len_exception() -> None:
    with pytest.raises(ValueError):
        """Primitives and Sesssions should have the same len."""
        CryptoPrimitiveProvider._transfer_primitives_to_parties(
            op_str="test", primitives=[1], sessions=[], p_kwargs={}
        )


def test_register_primitive() -> None:

    val = CryptoPrimitiveProvider.get_state()
    expected_providers = "test"

    assert expected_providers in val, "Test Provider not registered"


@pytest.mark.parametrize("nr_instances", [1, 5, 100])
@pytest.mark.parametrize("nr_parties", [2, 3, 4])
def test_generate_primitive(
    get_clients: Callable, nr_parties: int, nr_instances: int
) -> None:
    parties = get_clients(nr_parties)
    session = Session(parties=parties)
    SessionManager.setup_mpc(session)

    g_kwargs = {"nr_parties": nr_parties, "nr_instances": nr_instances}
    res = CryptoPrimitiveProvider.generate_primitives(
        "test",
        sessions=session.session_ptrs,
        g_kwargs=g_kwargs,
        p_kwargs=None,
    )

    assert isinstance(res, list)
    assert len(res) == nr_parties

    for i, primitives in enumerate(res):
        for primitive in primitives:
            assert primitive == tuple(i for _ in range(PRIMITIVE_NR_ELEMS))


@pytest.mark.parametrize(
    ("nr_instances", "nr_instances_retrieve"),
    [(1, 1), (5, 4), (5, 5), (100, 25), (100, 100)],
)
@pytest.mark.parametrize("nr_parties", [2, 3, 4])
def test_generate_and_transfer_primitive(
    get_clients: Callable,
    nr_parties: int,
    nr_instances: int,
    nr_instances_retrieve: int,
) -> None:
    parties = get_clients(nr_parties)
    session = Session(parties=parties)
    SessionManager.setup_mpc(session)

    g_kwargs = {"nr_parties": nr_parties, "nr_instances": nr_instances}
    CryptoPrimitiveProvider.generate_primitives(
        "test",
        sessions=session.session_ptrs,
        g_kwargs=g_kwargs,
        p_kwargs={},
    )

    for i in range(nr_parties):
        remote_crypto_store = session.session_ptrs[i].crypto_store
        primitives = remote_crypto_store.get_primitives_from_store(
            op_str="test", nr_instances=nr_instances_retrieve
        ).get()
        assert primitives == [
            tuple(i for _ in range(PRIMITIVE_NR_ELEMS))
            for _ in range(nr_instances_retrieve)
        ]


def test_primitive_logging_model(get_clients) -> None:
    model = LinearNet(torch)

    clients = get_clients(2)

    session = Session(parties=clients)
    SessionManager.setup_mpc(session)

    mpc_model = model.share(session=session)

    x_secret = torch.randn(2, 3)
    x_mpc = MPCTensor(secret=x_secret, session=session)

    model.eval()

    expected_primitive_log = {
        "beaver_matmul": [{"a_shape": (2, 3), "b_shape": (3, 10), "nr_parties": 2}],
        "fss_comp": [{"n_values": 20}],
        "beaver_mul": [{"a_shape": (2, 10), "b_shape": (2, 10), "nr_parties": 2}],
    }

    CryptoPrimitiveProvider.start_logging()
    mpc_model(x_mpc)
    primitive_log = CryptoPrimitiveProvider.stop_logging()

    assert expected_primitive_log == primitive_log


@pytest.mark.parametrize(
    "ops",
    [
        ["beaver_mul", {"a_shape": [1, 5], "b_shape": [1, 5], "nr_parties": 2}],
        [
            "beaver_matmul",
            {"a_shape": [1, 2880], "b_shape": [2880, 10], "nr_parties": 2},
        ],
        [
            "beaver_conv2d",
            {"a_shape": [1, 1, 28, 28], "b_shape": [5, 1, 5, 5], "nr_parties": 2},
        ],
        ["fss_comp", {"n_values": 4}],
    ],
)
def test_primitive_logging_ops(ops, get_clients) -> None:
    clients = get_clients(2)
    session = Session(parties=clients)
    SessionManager.setup_mpc(session)

    if ops[0] != "fss_comp":
        p_kwargs = {
            "a_shape": tuple(ops[1].get("a_shape")),
            "b_shape": tuple(ops[1].get("b_shape")),
        }
    else:
        p_kwargs = {}

    CryptoPrimitiveProvider.start_logging()
    CryptoPrimitiveProvider.generate_primitives(
        sessions=session.session_ptrs,
        op_str=ops[0],
        p_kwargs=p_kwargs,
        g_kwargs=ops[1],
    )
    primitive_log = CryptoPrimitiveProvider.stop_logging()
    expected_log = {ops[0]: [ops[1]]}

    assert expected_log == primitive_log

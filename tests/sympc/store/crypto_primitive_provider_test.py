# stdlib
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Tuple

# third party
import pytest

from sympc.session import Session
from sympc.store import CryptoPrimitiveProvider
from sympc.store import register_primitive_generator
from sympc.store import register_primitive_store_add
from sympc.store import register_primitive_store_get

PRIMITIVE_NR_ELEMS = 4


@register_primitive_generator("test")
def provider_test(nr_parties: int) -> List[Tuple[int]]:
    """
    This function will generate the values
    ([0, 1, ...], [0, 1, ...], [0, 1, ...], [0, 1, ...])

    And the CryptoProvider will arrange them like:

    [((0, 0, 0, 0), (1, 1, 1, 1), ...)]
    """
    primitives = (list(range(nr_parties)) for _ in range(PRIMITIVE_NR_ELEMS))
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
        """Primitives should be a list"""
        CryptoPrimitiveProvider._transfer_primitives_to_parties(
            op_str="test", primitives=50, sessions=[], p_kwargs={}
        )


def test_transfer_primitives_mismatch_len_exception() -> None:
    with pytest.raises(ValueError):
        """Primitives and Sesssions should have the same len"""
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
    Session.setup_mpc(session)

    g_kwargs = {"nr_parties": nr_parties}
    res = CryptoPrimitiveProvider.generate_primitives(
        "test",
        n_instances=nr_instances,
        sessions=session.session_ptrs,
        g_kwargs=g_kwargs,
        p_kwargs=None,
    )

    assert isinstance(res, list)
    assert len(res) == nr_instances

    for i, primitives in enumerate(res[0]):
        assert primitives == tuple(i for _ in range(PRIMITIVE_NR_ELEMS))


@pytest.mark.parametrize("nr_instances", [1, 5, 100])
@pytest.mark.parametrize("nr_parties", [2, 3, 4])
def test_generate_and_transfer_primitive(
    get_clients: Callable, nr_parties: int, nr_instances: int
) -> None:
    parties = get_clients(nr_parties)
    session = Session(parties=parties)
    Session.setup_mpc(session)

    g_kwargs = {"nr_parties": nr_parties}
    CryptoPrimitiveProvider.generate_primitives(
        "test",
        n_instances=nr_instances,
        sessions=session.session_ptrs,
        g_kwargs=g_kwargs,
        p_kwargs={},
    )

    for i in range(nr_parties):
        remote_crypto_store = session.session_ptrs[i].crypto_store
        primitives = remote_crypto_store.get_primitives_from_store("test").get()
        assert primitives == [tuple(i for _ in range(PRIMITIVE_NR_ELEMS))]

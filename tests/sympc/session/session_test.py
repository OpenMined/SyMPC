"""Tests for the Session class."""

# stdlib
import secrets

# third party
import pytest
import torch

from sympc.config import Config
from sympc.protocol import Falcon
from sympc.session import Session
from sympc.session import SessionManager
from sympc.tensor import PRIME_NUMBER
from sympc.tensor import ReplicatedSharedTensor
from sympc.tensor import ShareTensor
from sympc.utils import RING_SIZE_TO_TYPE
from sympc.utils import generate_random_element
from sympc.utils import get_new_generator
from sympc.utils import get_type_from_ring


def test_session_default_init() -> None:
    """Test correct initialisation of the Sessin class."""
    # Test default init
    session = Session()
    assert session.uuid is None
    assert session.parties == []
    assert session.trusted_third_party is None
    assert session.crypto_store is None
    assert session.protocol is not None
    assert isinstance(session.config, Config)
    assert session.przs_generators == []
    assert session.rank == -1
    assert session.session_ptrs == []
    assert session.tensor_type == get_type_from_ring(2 ** 64)
    assert session.ring_size == 2 ** 64
    assert session.min_value == -(2 ** 64) // 2
    assert session.max_value == (2 ** 64 - 1) // 2


def test_session_custom_init() -> None:
    config = Config()
    session = Session(
        parties=["alice", "bob"], ring_size=2 ** 32, config=config, ttp="TTP"
    )
    assert session.uuid is None
    assert session.parties == ["alice", "bob"]
    assert session.trusted_third_party == "TTP"
    assert session.crypto_store is None
    assert session.protocol is not None
    assert session.config == config
    assert session.przs_generators == []
    assert session.rank == -1
    assert session.session_ptrs == []
    assert session.tensor_type == get_type_from_ring(2 ** 32)
    assert session.ring_size == 2 ** 32
    assert session.min_value == -(2 ** 32) // 2
    assert session.max_value == (2 ** 32 - 1) // 2


def test_przs_share_tensor() -> None:
    """Test przs_generate_random_share method from Session for ShareTensor."""
    session = Session()  # default protocol: FSS
    SessionManager.setup_mpc(session)
    seed1 = secrets.randbits(32)
    seed2 = secrets.randbits(32)
    gen1 = get_new_generator(seed1)
    gen2 = get_new_generator(seed2)
    session.przs_generators = [gen1, gen2]
    shape = (2, 1)
    share = session.przs_generate_random_share(shape=shape)
    assert isinstance(share, ShareTensor)

    new_gen1 = get_new_generator(seed1)
    new_gen2 = get_new_generator(seed2)
    share1 = generate_random_element(
        generator=new_gen1, shape=shape, tensor_type=session.tensor_type
    )
    share2 = generate_random_element(
        generator=new_gen2, shape=shape, tensor_type=session.tensor_type
    )
    target_tensor = share1 - share2
    assert (share.tensor == target_tensor).all()


def test_przs_rs_tensor() -> None:
    """Test przs_generate_random_share method from Session for ReplicatedSharedTensor."""
    falcon = Falcon(security_type="malicious")
    session = Session(protocol=falcon)
    SessionManager.setup_mpc(session)
    seed1 = secrets.randbits(32)
    seed2 = secrets.randbits(32)
    gen1 = get_new_generator(seed1)
    gen2 = get_new_generator(seed2)
    session.przs_generators = [gen1, gen2]
    shape = (2, 1)
    share = session.przs_generate_random_share(shape=shape)
    assert isinstance(share, ReplicatedSharedTensor)

    new_gen1 = get_new_generator(seed1)
    new_gen2 = get_new_generator(seed2)
    share1 = generate_random_element(
        generator=new_gen1, shape=shape, tensor_type=session.tensor_type
    )
    share2 = generate_random_element(
        generator=new_gen2, shape=shape, tensor_type=session.tensor_type
    )
    target_tensor = share1 - share2
    assert (share.shares[0] == target_tensor).all()


def test_przs_share_tensor_pointer(get_clients) -> None:
    clients = get_clients(3)
    session = Session(parties=clients)  # default protocol: FSS
    SessionManager.setup_mpc(session)

    party1 = session.session_ptrs[0]
    share_ptr = party1.przs_generate_random_share(shape=(1, 2))

    assert share_ptr.class_name.endswith("UnionPointer")
    share = share_ptr.get()
    assert isinstance(share, ShareTensor)


def test_przs_rs_tensor_pointer(get_clients) -> None:
    clients = get_clients(3)
    falcon = Falcon(security_type="malicious")
    session = Session(protocol=falcon, parties=clients)
    SessionManager.setup_mpc(session)

    party1 = session.session_ptrs[0]
    share_ptr = party1.przs_generate_random_share(shape=(1, 2))

    assert share_ptr.class_name.endswith("UnionPointer")
    share = share_ptr.get()
    assert isinstance(share, ReplicatedSharedTensor)


def test_prrs_share_tensor() -> None:
    """Test przs_generate_random_share method from Session for ShareTensor."""
    session = Session()  # default protocol: FSS
    SessionManager.setup_mpc(session)
    seed1 = secrets.randbits(32)
    seed2 = secrets.randbits(32)
    gen1 = get_new_generator(seed1)
    gen2 = get_new_generator(seed2)
    session.przs_generators = [gen1, gen2]
    shape = (2, 1)
    share = session.prrs_generate_random_share(shape=shape)
    assert isinstance(share, ShareTensor)

    new_gen1 = get_new_generator(seed1)
    share1 = generate_random_element(
        generator=new_gen1, shape=shape, tensor_type=session.tensor_type
    )
    target_tensor = share1
    assert (share.tensor == target_tensor).all()


def test_prrs_rs_tensor() -> None:
    """Test przs_generate_random_share method from Session for ReplicatedSharedTensor."""
    falcon = Falcon(security_type="malicious")
    session = Session(protocol=falcon)
    SessionManager.setup_mpc(session)
    seed1 = secrets.randbits(32)
    seed2 = secrets.randbits(32)
    gen1 = get_new_generator(seed1)
    gen2 = get_new_generator(seed2)
    session.przs_generators = [gen1, gen2]
    shape = (2, 1)
    share = session.prrs_generate_random_share(shape=shape)
    assert isinstance(share, ReplicatedSharedTensor)

    new_gen1 = get_new_generator(seed1)
    new_gen2 = get_new_generator(seed2)
    share1 = generate_random_element(
        generator=new_gen1, shape=shape, tensor_type=session.tensor_type
    )
    share2 = generate_random_element(
        generator=new_gen2, shape=shape, tensor_type=session.tensor_type
    )
    target_tensor = [share1, share2]
    assert (torch.cat(share.shares) == torch.cat(target_tensor)).all()


def test_prrs_share_tensor_pointer(get_clients) -> None:
    clients = get_clients(3)
    session = Session(parties=clients)  # default protocol: FSS
    SessionManager.setup_mpc(session)

    party1 = session.session_ptrs[0]
    share_ptr = party1.prrs_generate_random_share(shape=(1, 2))

    assert share_ptr.class_name.endswith("UnionPointer")
    share = share_ptr.get()
    assert isinstance(share, ShareTensor)


def test_prrs_rs_tensor_pointer(get_clients) -> None:
    clients = get_clients(3)
    falcon = Falcon(security_type="malicious")
    session = Session(protocol=falcon, parties=clients)
    SessionManager.setup_mpc(session)

    party1 = session.session_ptrs[0]
    share_ptr = party1.prrs_generate_random_share(shape=(1, 2))

    assert share_ptr.class_name.endswith("UnionPointer")
    share = share_ptr.get()
    assert isinstance(share, ReplicatedSharedTensor)


def test_eq() -> None:
    """Test __eq__ for Session."""
    session = Session()
    other1 = Session()
    other2 = session

    # Test different instances:
    assert session != 1

    # Test equal sessions:
    assert session == other2

    # Test same sessions (until we call setup mpc):
    assert session == other1

    SessionManager.setup_mpc(session)

    assert session != other1


def test_copy() -> None:
    session = Session()

    copy_session = session.copy()

    assert session.nr_parties == copy_session.nr_parties
    assert session.config == copy_session.config
    assert session.protocol == copy_session.protocol


def test_invalid_protocol_exception() -> None:
    with pytest.raises(ValueError):
        Session(protocol="fs")


def test_invalid_ringsize_exception() -> None:
    with pytest.raises(ValueError):
        Session(ring_size=2 ** 63)


def test_przs_share_tensor_union_resolve(get_clients) -> None:
    clients = get_clients(3)
    session = Session(parties=clients)
    SessionManager.setup_mpc(session)

    share_pt0 = session.session_ptrs[0].przs_generate_random_share(shape=(1, 2))
    resolved_share_pt0 = share_pt0.resolve_pointer_type()
    share_pt_name = type(resolved_share_pt0).__name__

    assert share_pt_name == "ShareTensorPointer"


def test_prrs_share_tensor_union_resolve(get_clients) -> None:
    clients = get_clients(3)
    session = Session(parties=clients)
    SessionManager.setup_mpc(session)

    share_pt0 = session.session_ptrs[0].prrs_generate_random_share(shape=(1, 2))
    resolved_share_pt0 = share_pt0.resolve_pointer_type()
    share_pt_name = type(resolved_share_pt0).__name__

    assert share_pt_name == "ShareTensorPointer"


def test_przs_rst_union_resolve(get_clients) -> None:
    clients = get_clients(3)
    protocol = Falcon("semi-honest")
    session = Session(protocol=protocol, parties=clients)
    SessionManager.setup_mpc(session)

    share_pt0 = session.session_ptrs[0].przs_generate_random_share(shape=(1, 2))
    resolved_share_pt0 = share_pt0.resolve_pointer_type()
    share_pt_name = type(resolved_share_pt0).__name__

    assert share_pt_name == "ReplicatedSharedTensorPointer"


def test_prrs_rst_union_resolve(get_clients) -> None:
    clients = get_clients(3)
    protocol = Falcon("semi-honest")
    session = Session(protocol=protocol, parties=clients)
    SessionManager.setup_mpc(session)

    share_pt0 = session.session_ptrs[0].prrs_generate_random_share(shape=(1, 2))
    resolved_share_pt0 = share_pt0.resolve_pointer_type()
    share_pt_name = type(resolved_share_pt0).__name__

    assert share_pt_name == "ReplicatedSharedTensorPointer"


def test_przs_share_ring_size(get_clients) -> None:
    clients = get_clients(3)
    session = Session(parties=clients)
    SessionManager.setup_mpc(session)

    for ring_size in RING_SIZE_TO_TYPE.keys() - {2, PRIME_NUMBER}:
        share_pt0 = session.session_ptrs[0].przs_generate_random_share(
            shape=(1, 2), ring_size=str(ring_size)
        )
        share = share_pt0.get_copy()

        assert share.ring_size == ring_size
        assert share.tensor.dtype == RING_SIZE_TO_TYPE[ring_size]


def test_przs_rst_ring_size(get_clients) -> None:
    clients = get_clients(3)
    falcon = Falcon()
    session = Session(protocol=falcon, parties=clients)
    SessionManager.setup_mpc(session)

    for ring_size in RING_SIZE_TO_TYPE.keys():
        rst_pt0 = session.session_ptrs[0].przs_generate_random_share(
            shape=(1, 2), ring_size=str(ring_size)
        )
        share = rst_pt0.get_copy()

        assert share.ring_size == ring_size
        assert share.shares[0].dtype == RING_SIZE_TO_TYPE[ring_size]

        if ring_size == PRIME_NUMBER:
            assert torch.max(torch.cat(share.shares)) <= PRIME_NUMBER - 1
            assert torch.min(torch.cat(share.shares)) >= 0


def test_prrs_share_ring_size(get_clients) -> None:
    clients = get_clients(3)
    session = Session(parties=clients)
    SessionManager.setup_mpc(session)

    for ring_size in RING_SIZE_TO_TYPE.keys() - {2, PRIME_NUMBER}:
        share_pt0 = session.session_ptrs[0].prrs_generate_random_share(
            shape=(1, 2), ring_size=str(ring_size)
        )
        share = share_pt0.get_copy()

        assert share.ring_size == ring_size
        assert share.tensor.dtype == RING_SIZE_TO_TYPE[ring_size]


def test_prrs_rst_ring_size(get_clients) -> None:
    clients = get_clients(3)
    falcon = Falcon()
    session = Session(protocol=falcon, parties=clients)
    SessionManager.setup_mpc(session)

    for ring_size in RING_SIZE_TO_TYPE.keys():
        rst_pt0 = session.session_ptrs[0].prrs_generate_random_share(
            shape=(1, 2), ring_size=str(ring_size)
        )
        share = rst_pt0.get_copy()

        assert share.ring_size == ring_size
        assert share.shares[0].dtype == RING_SIZE_TO_TYPE[ring_size]
        assert share.shares[1].dtype == RING_SIZE_TO_TYPE[ring_size]

        if ring_size == PRIME_NUMBER:
            assert torch.max(torch.cat(share.shares)) <= PRIME_NUMBER - 1
            assert torch.min(torch.cat(share.shares)) >= 0

# third party
# third party
import pytest

from sympc.protocol import Falcon
from sympc.session import Session
from sympc.session import SessionManager
from sympc.store import CryptoPrimitiveProvider


def test_rst_invalid_triple(get_clients) -> None:
    parties = get_clients(3)
    falcon = Falcon("malicious")
    session = Session(parties, protocol=falcon)
    SessionManager.setup_mpc(session)
    shape_x = (1,)
    shape_y = (1,)
    # create an inconsistent sharing,invoke a prrs first
    session.session_ptrs[0].prrs_generate_random_share(shape_x)

    with pytest.raises(ValueError):

        CryptoPrimitiveProvider.generate_primitives(
            "beaver_mul",
            session=session,
            g_kwargs={
                "session": session,
                "a_shape": shape_x,
                "b_shape": shape_y,
                "nr_parties": session.nr_parties,
            },
            p_kwargs={"a_shape": shape_x, "b_shape": shape_y},
        )

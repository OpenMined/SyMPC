from .. import beaver
from ...tensor import modulo
from concurrent.futures import ThreadPoolExecutor, wait
import sympc

""" Functions that the master run """
def mul_master(x, y):

    """
    [c] = [a * b]
    [eps] = [x] - [a]
    [delta] = [y] - [b]

    Open eps and delta
    [result] = [c] + eps * [b] + delta * [a] + eps * delta
    """
    a_sh, b_sh, c_sh = beaver.build_triples("mul", x, y)
    eps = x - a_sh
    delta = y - b_sh
    nr_parties = len(x.shares)
    session = x.session

    eps_plaintext = eps.reconstruct(decode=False)
    delta_plaintext = delta.reconstruct(decode=False)

    with ThreadPoolExecutor(max_workers=nr_parties) as executor:
        args = list(zip(session.session_ptr, a_sh.shares, b_sh.shares, c_sh.shares))
        futures = [
            executor.submit(session.parties[i].sympc.protocol.spdz.mul_parties, *args[i], eps_plaintext, delta_plaintext)
            for i in range(nr_parties)
        ]

    shares = [f.result() for f in futures]
    return shares


""" Functions that run at a party """
def mul_parties(session, a_share, b_share, c_share, eps, delta):
    eps_b = modulo(eps * b_share, session)
    delta_a = modulo(delta * a_share, session)

    share = modulo(modulo(c_share + eps_b, session) + delta_a, session)
    if session.rank == 0:
        delta_eps = delta * eps
        share = modulo(share + delta_eps, session)

    return modulo(share, session)

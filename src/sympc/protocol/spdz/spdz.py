"""SPDZ Protocol.

SPDZ mechanism used for multiplication Contains functions that are run at:

* the party that orchestrates the computation
* the parties that hold the shares
"""

# stdlib
import operator
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

# third party
import torch

from sympc.session.session_utils import get_session
from sympc.store import CryptoPrimitiveProvider
from sympc.tensor import MPCTensor
from sympc.tensor import ShareTensor
from sympc.utils import count_wraps
from sympc.utils import parallel_execution

EXPECTED_OPS = {"mul", "matmul", "conv2d"}


""" Functions that are executed at the orchestrator """


def mul_master(
    x: MPCTensor, y: MPCTensor, op_str: str, kwargs_: Dict[Any, Any]
) -> MPCTensor:
    """Function that is executed by the orchestrator to multiply two secret values.

    Args:
        x (MPCTensor): First value to multiply with.
        y (MPCTensor): Second value to multiply with.
        op_str (str): Operation string.
        kwargs_ (dict): TODO:Add docstring.

    Raises:
        ValueError: If op_str not in EXPECTED_OPS.

    Returns:
        MPCTensor: Result of the multiplication.
    """
    if op_str not in EXPECTED_OPS:
        raise ValueError(f"{op_str} should be in {EXPECTED_OPS}")

    session = x.session

    shape_x = tuple(x.shape)
    shape_y = tuple(y.shape)

    CryptoPrimitiveProvider.generate_primitives(
        f"beaver_{op_str}",
        sessions=session.session_ptrs,
        g_kwargs={
            "a_shape": shape_x,
            "b_shape": shape_y,
            "nr_parties": session.nr_parties,
            **kwargs_,
        },
        p_kwargs={"a_shape": shape_x, "b_shape": shape_y},
    )

    args = [list(el) + [op_str] for el in zip(x.share_ptrs, y.share_ptrs)]

    mask = parallel_execution(spdz_mask, session.parties)(args)
    eps_shares, delta_shares = zip(*mask)

    eps = MPCTensor(shares=eps_shares, session=session)
    delta = MPCTensor(shares=delta_shares, session=session)

    eps_plaintext = eps.reconstruct(decode=False)
    delta_plaintext = delta.reconstruct(decode=False)

    # Arguments that must be sent to all parties
    common_args = [eps_plaintext, delta_plaintext, op_str]

    # Specific arguments to each party
    args = [common_args for el in session.session_ptrs]

    shares = parallel_execution(mul_parties, session.parties)(args, kwargs_)

    result = MPCTensor(shares=shares, session=session)

    return result


def public_divide(x: MPCTensor, y: Union[torch.Tensor, int]) -> MPCTensor:
    """Function that is executed by the orchestrator to divide a secret by a public value.

    Args:
        x (MPCTensor): Private numerator.
        y (Union[torch.Tensor, int]): Public denominator.

    Returns:
        MPCTensor: A new set of shares that represents the division.
    """
    session = x.session
    res_shape = x.shape

    if session.nr_parties == 2:
        shares = [operator.truediv(share, y) for share in x.share_ptrs]
        return MPCTensor(shares=shares, session=session, shape=res_shape)

    primitives = CryptoPrimitiveProvider.generate_primitives(
        "beaver_wraps",
        sessions=session.session_ptrs,
        g_kwargs={"nr_parties": session.nr_parties, "shape": res_shape},
        p_kwargs=None,
    )

    r_sh, theta_r_sh = list(zip(*list(zip(*primitives))[0]))

    r_mpc = MPCTensor(shares=r_sh, session=session, shape=x.shape)

    z = r_mpc + x
    z_shares_local = z.get_shares()

    common_args = [z_shares_local, y]
    args = zip(r_mpc.share_ptrs, theta_r_sh, x.share_ptrs)
    args = [list(el) + common_args for el in args]

    theta_x = parallel_execution(div_wraps, session.parties)(args)
    theta_x_plaintext = MPCTensor(shares=theta_x, session=session).reconstruct()

    res = x - theta_x_plaintext * 4 * ((session.ring_size // 4) // y)

    return res


""" Functions that are executed at each party that holds shares """


def div_wraps(
    r_share: ShareTensor,
    theta_r: ShareTensor,
    x_share: ShareTensor,
    z_shares: List[torch.Tensor],
    y: Union[torch.Tensor, int],
) -> ShareTensor:
    """From CrypTen Privately computes the number of wraparounds for a set a shares.

    To do so, we note that:
        [theta_x] = theta_z + [beta_xr] - [theta_r] - [eta_xr]

    Where:
        [theta_x] is the wraps for a variable x
        [beta_xr] is the differential wraps for variables x and r
        [eta_xr]  is the plaintext wraps for variables x and r

    Note: Since [eta_xr] = 0 with probability 1 - |x| / Q for modulus Q, we
    can make the assumption that [eta_xr] = 0 with high probability.

    Args:
        r_share (ShareTensor): Share.
        theta_r (ShareTensor): Share
        x_share (ShareTensor): Share
        z_shares (ShareTensor): Share
        y (Union[Torch.Tensor,y]): Public denominator.

    Returns:
        ShareTensor: Shared result of the division.
    """
    session = get_session()

    beta_xr = count_wraps([x_share.tensor, r_share.tensor])
    theta_x = ShareTensor(encoder_precision=0)
    theta_x.tensor = beta_xr - theta_r.tensor

    if session.rank == 0:
        theta_z = count_wraps(z_shares)
        theta_x.tensor += theta_z

    x_share.tensor //= y

    return theta_x


def spdz_mask(
    x_sh: ShareTensor, y_sh: ShareTensor, op_str: str
) -> Tuple[ShareTensor, ShareTensor]:
    """Spdz mask.

    Args:
        x_sh (ShareTensor): X share.
        y_sh (ShareTensor): Y share.
        op_str (str): Operator.

    Returns:
        Tuple[ShareTensor, ShareTensor]
    """
    session = get_session()

    crypto_store = session.crypto_store

    primitives = crypto_store.get_primitives_from_store(
        f"beaver_{op_str}", x_sh.shape, y_sh.shape, remove=False
    )

    a_sh, b_sh, _ = primitives

    return x_sh - a_sh, y_sh - b_sh


def mul_parties(
    eps: torch.Tensor, delta: torch.Tensor, op_str: str, **kwargs
) -> ShareTensor:
    """SPDZ Multiplication.

    Args:
        eps (torch:tensor): Epsilon value of the protocol.
        delta (torch.Tensor): Delta value of the protocol.
        op_str (str): Operator string.
        kwargs: Keywords arguments for the operator.

    Returns:
        ShareTensor: Shared result of the division.
    """
    session = get_session()

    crypto_store = session.crypto_store
    eps_shape = tuple(eps.shape)
    delta_shape = tuple(delta.shape)

    primitives = crypto_store.get_primitives_from_store(
        f"beaver_{op_str}", eps_shape, delta_shape
    )

    a_share, b_share, c_share = primitives

    if op_str == "conv2d":
        op = torch.conv2d
    else:
        op = getattr(operator, op_str)

    eps_b = op(eps, b_share.tensor, **kwargs)
    delta_a = op(a_share.tensor, delta, **kwargs)

    share_tensor = c_share.tensor + eps_b + delta_a
    if session.rank == 0:
        delta_eps = op(eps, delta, **kwargs)
        share_tensor += delta_eps

    # Convert to our tensor type
    share_tensor = share_tensor.type(session.tensor_type)

    share = ShareTensor(session=session)
    share.tensor = share_tensor

    # Ideally this should stay in the MPCTensor
    # Step 1. Do spdz_mul
    # Step 2. Divide by scale
    # This is done here to reduce one round of communication
    if session.nr_parties == 2:
        share.tensor //= share.fp_encoder.scale

    return share

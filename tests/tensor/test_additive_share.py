import pytest
import syft as sy
import torch

from syft.lib.sympc.session import SySession
from sympc.tensor.additive_shared import AdditiveSharingTensor


def clients():
    alice = sy.VirtualMachine(name="alice")
    bob = sy.VirtualMachine(name="bob")
    james = sy.VirtualMachine(name="james")

    alice_client = alice.get_client()
    bob_client = bob.get_client()
    james_client = james.get_client()

    return [alice_client, bob_client, james_client]


def test_reconstruct():
    alice_client, bob_client, james_client = clients()
    session = SySession(parties=[alice_client, bob_client, james_client])
    session.setup_mpc()

    x_secret = torch.tensor([1, 2, 3, 4])
    x = AdditiveSharingTensor(secret=x_secret, session=session)

    x = x.reconstruct()

    assert (x_secret == x).all()


def test_add():
    alice_client, bob_client, james_client = clients()
    session = SySession(parties=[alice_client, bob_client, james_client])
    session.setup_mpc()

    x_secret = torch.tensor([1, 2, 3])
    x = AdditiveSharingTensor(secret=x_secret, session=session)
    result = (x + x).reconstruct()
    assert (result == (x_secret + x_secret)).all()

    x_secret = torch.tensor([1, 2, 3])
    y_secret = torch.tensor([4, 5, 6])
    x = AdditiveSharingTensor(secret=x_secret, session=session)
    y = AdditiveSharingTensor(secret=y_secret, session=session)
    result = (x + y).reconstruct()
    assert (result == (x_secret + y_secret)).all()

    # with negative numbers
    x_secret = torch.tensor([1, -2, 3])
    x = AdditiveSharingTensor(secret=x_secret, session=session)
    result = (x + x).reconstruct()
    assert (result == (x_secret + x_secret)).all()

    x_secret = torch.tensor([1, -2, 3])
    y_secret = torch.tensor([4, 5, -6])
    x = AdditiveSharingTensor(secret=x_secret, session=session)
    y = AdditiveSharingTensor(secret=y_secret, session=session)
    result = (x + y).reconstruct()
    assert (result == (x_secret + y_secret)).all()

    # with constant integer
    x_secret = torch.tensor([1, 2, 3])
    c = 4
    x = AdditiveSharingTensor(secret=x_secret, session=session)
    result = (x + c).reconstruct()
    assert (result == (x_secret + c)).all()

    # with constant integer
    x_secret = torch.tensor([1, 2, 3])
    c = 4.6
    x = AdditiveSharingTensor(secret=x_secret, session=session)
    result = (x + c).reconstruct()
    assert (result == (x_secret + c)).all()









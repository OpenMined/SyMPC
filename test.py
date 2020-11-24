import torch
import syft as sy
from sympc.tensor.additive_shared import AdditiveSharingTensor


alice = sy.VirtualMachine(name="alice")
bob = sy.VirtualMachine(name="bob")

alice_client = alice.get_client()
bob_client = bob.get_client()
session = sy.lib.sympc.session.SySession(parties=[alice_client, bob_client])
print(session)



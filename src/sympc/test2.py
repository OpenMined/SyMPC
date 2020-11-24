from sympc.tensor import MPCTensor
import torch
import syft
from sympc.session import Session


alice = syft.VirtualMachine(name="alice")
bob = syft.VirtualMachine(name="bob")

alice_client = alice.get_client()
bob_client = bob.get_client()

session = Session(parties=[alice_client, bob_client])
Session.setup_mpc(session)


x_secret = torch.Tensor([1, 2, 3])
y_secret = torch.Tensor([4, 5, 6])
x = MPCTensor(secret=x_secret, session=session)
y = MPCTensor(secret=y_secret, session=session)
result = (x * y).reconstruct()
result_secret = x_secret * y_secret
print(result)
print(result_secret)
assert torch.allclose(result, result_secret)

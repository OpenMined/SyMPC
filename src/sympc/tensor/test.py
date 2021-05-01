import torch
import syft as sy

from sympc.session import Session
from sympc.session import SessionManager
from sympc.tensor import MPCTensor
from sympc.tensor import ShareTensor

alice = sy.VirtualMachine(name="alice")
bob = sy.VirtualMachine(name="bob")

alice_client = alice.get_client()
bob_client = bob.get_client()

session = Session(parties=[alice_client, bob_client])
SessionManager.setup_mpc(session)


x = ShareTensor(data=torch.Tensor([[1.0, 2.0, 3], [4, 5, 6]]), requires_grad=True)
y = ShareTensor(data=torch.Tensor([[2.0, 3.0], [4, 5], [6, 7]]), requires_grad=False)

z = x.t()
print(z.requires_grad)
q = z * y

"""
   ----- t ---- z (1 parent)
  /               \
x (no parent)      ---- * ---- q (2 parents)
                  /
                 /
                /
y (no parent) --
"""


print(x.parents)
print(z.parents)
print(y.parents)
print(q.parents)

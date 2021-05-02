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

# x = MPCTensor(secret=torch.Tensor([1, 2, 3]), requires_grad=True, session=session)
# print(x)


x = ShareTensor(data=torch.Tensor([6.0]), requires_grad=True)
y = ShareTensor(data=torch.Tensor([2.0]), requires_grad=False)

z = x.t()
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


# print(q.nr_out_edges)

q.backward()
print("=============================================")
print(q.decode())
print(q.grad.decode())
print(x.grad.decode())
print(z.grad.decode())
print(y.grad.decode())

import pdb

pdb.set_trace()

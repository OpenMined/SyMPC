from sympc.tensor import MPCTensor
from sympc.tensor import ShareTensor
from sympc.session import Session
from sympc.encoder import FixedPointEncoder
import torch
import syft

alice = syft.VirtualMachine(name="alice")
bob = syft.VirtualMachine(name="bob")

alice_client = alice.get_client()
bob_client = bob.get_client()

session = Session(parties=[alice_client, bob_client])
Session.setup_mpc(session)

x_secret = torch.Tensor([1, 2, 3])
c = ShareTensor(4.6)
import pdb

pdb.set_trace()
x = MPCTensor(secret=x_secret, session=session)
result = (x + c).reconstruct()
result_c = x_secret + c
print(result)

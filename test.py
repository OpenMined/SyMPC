from jax import grad
import jax.numpy as jnp
import syft as sy
from sympc.tensor import MPCTensor
from sympc.session import Session
from sympc.session import SessionManager
import torch as th

alice = sy.VirtualMachine(name="alice")
alice_client = alice.get_client()

bob = sy.VirtualMachine(name="bob")
bob_client = bob.get_client()

session = Session(parties=[alice_client, bob_client])
SessionManager.setup_mpc(session)



secret = MPCTensor(secret=th.Tensor([1, 2, 3]), session=session)
print(secret)


def f(x):  # Define a function
  y = 2 * x+1
  return y

grad_f = grad(f)
print(grad_f(jnp.array(secret)))

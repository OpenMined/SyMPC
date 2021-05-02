import torch

x = torch.tensor([6.0], requires_grad=True)
y = torch.tensor([2.0], requires_grad=False)

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

z.retain_grad()
q.retain_grad()
q.backward()
print(q)
print(q.grad)
print("=======")
print(y.grad)
print(x.grad)
print(z.grad)
import pdb

pdb.set_trace()

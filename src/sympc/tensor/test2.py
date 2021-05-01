import torch

x = torch.tensor([[1.0, 2.0, 3], [4, 5, 6]], requires_grad=True)
y = torch.tensor([[2.0, 3.0], [4, 5], [6, 7]], requires_grad=False)

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

import pdb

pdb.set_trace()

print(x.parents)
print(z.parents)
print(y.parents)
print(q.children)

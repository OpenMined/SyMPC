"""Expose various objects related to SyMPC library.

This file exposes lists of allowed modules, classes and attributes that PySyft uses to build
an Abstract Syntax Tree representing the SyMPC library. This lists could be in PySyft.
They were in the past. However, this had a problem. Whenever we wanted to add a new "operation"
to the ShareTensor list (allowed_external_attrs) and call it from a share tensor pointer, some
steps where needed to introduce that change and make all test pass.

1. Create a PR in the SyMPC (tests fail)
2. Create a PR in PySyft (tests fail)
3. Merge PySyft PR with failing tests
4. Now that SyMPC has PySyft changes, run PySyft tests and merge the PR if all is correct
5. Now that PySyft merged the PR, rerun SyMPC tests and if all is correct merge the PR

With this lists here, SyMPC has the control and this "Double PR tests error" is solved.
"""

import sympc

from . import protocol  # noqa: 401
from . import session  # noqa: 401
from . import store  # noqa: 401
from . import tensor  # noqa: 401
from . import utils  # noqa: 401

allowed_external_modules = [
    ("sympc", sympc),
    ("sympc.session", session),
    ("sympc.tensor", tensor),
    ("sympc.tensor.static", tensor.static),
    ("sympc.protocol", protocol),
    ("sympc.store", store),
    ("sympc.protocol.falcon", protocol.falcon),
    ("sympc.protocol.falcon.falcon", protocol.falcon.falcon),
    ("sympc.protocol.fss", protocol.fss),
    ("sympc.protocol.fss.fss", protocol.fss.fss),
    ("sympc.protocol.spdz", protocol.spdz),
    ("sympc.protocol.spdz.spdz", protocol.spdz.spdz),
    ("sympc.utils", utils),
    ("sympc.tensor.grads", tensor.grads),
    ("sympc.tensor.grads.grad_functions", tensor.grads.grad_functions),
    (
        "sympc.tensor.grads.grad_functions.GradConv2d",
        tensor.grads.grad_functions.GradConv2d,
    ),
]

allowed_external_classes = [
    ("sympc.session.Session", "sympc.session.Session", session.Session),
    ("sympc.store.CryptoStore", "sympc.store.CryptoStore", store.CryptoStore),
    (
        "sympc.tensor.ShareTensor",
        "sympc.tensor.ShareTensor",
        tensor.ShareTensor,
    ),
    (
        "sympc.tensor.ReplicatedSharedTensor",
        "sympc.tensor.ReplicatedSharedTensor",
        tensor.ReplicatedSharedTensor,
    ),
]

share_tensor_attrs = [
    (
        "sympc.tensor.ShareTensor.__add__",
        "sympc.tensor.ShareTensor",
    ),
    (
        "sympc.tensor.ShareTensor.__sub__",
        "sympc.tensor.ShareTensor",
    ),
    (
        "sympc.tensor.ShareTensor.__rmul__",
        "sympc.tensor.ShareTensor",
    ),
    (
        "sympc.tensor.ShareTensor.__mul__",
        "sympc.tensor.ShareTensor",
    ),
    (
        "sympc.tensor.ShareTensor.__matmul__",
        "sympc.tensor.ShareTensor",
    ),
    (
        "sympc.tensor.ShareTensor.__truediv__",
        "sympc.tensor.ShareTensor",
    ),
    (
        "sympc.tensor.ShareTensor.__rmatmul__",
        "sympc.tensor.ShareTensor",
    ),
    (
        "sympc.tensor.ShareTensor.t",
        "sympc.tensor.ShareTensor",
    ),
    (
        "sympc.tensor.ShareTensor.sum",
        "sympc.tensor.ShareTensor",
    ),
    (
        "sympc.tensor.ShareTensor.clone",
        "sympc.tensor.ShareTensor",
    ),
    (
        "sympc.tensor.ShareTensor.numel",
        "syft.lib.python.Int",  # FIXME: Can't we just return an int??
    ),
    (
        "sympc.tensor.ShareTensor.T",
        "sympc.tensor.ShareTensor",
    ),
    ("sympc.tensor.ShareTensor.squeeze", "sympc.tensor.ShareTensor"),
    ("sympc.tensor.ShareTensor.unsqueeze", "sympc.tensor.ShareTensor"),
    ("sympc.tensor.ShareTensor.reshape", "sympc.tensor.ShareTensor"),
    ("sympc.tensor.ShareTensor.view", "sympc.tensor.ShareTensor"),
    ("sympc.tensor.ShareTensor.expand", "sympc.tensor.ShareTensor"),
    ("sympc.tensor.static.stack_share_tensor", "sympc.tensor.ShareTensor"),
    ("sympc.tensor.static.cat_share_tensor", "sympc.tensor.ShareTensor"),
    ("sympc.tensor.static.helper_argmax_pairwise", "sympc.tensor.ShareTensor"),
    ("sympc.tensor.ShareTensor.reshape", "sympc.tensor.ShareTensor"),
    ("sympc.tensor.ShareTensor.repeat", "sympc.tensor.ShareTensor"),
    ("sympc.tensor.ShareTensor.narrow", "sympc.tensor.ShareTensor"),
    ("sympc.tensor.ShareTensor.dim", "syft.lib.python.Int"),
    ("sympc.tensor.ShareTensor.transpose", "sympc.tensor.ShareTensor"),
    ("sympc.tensor.ShareTensor.flatten", "sympc.tensor.ShareTensor"),
    ("sympc.tensor.ShareTensor.expand", "sympc.tensor.ShareTensor"),
    ("sympc.tensor.ShareTensor.roll", "sympc.tensor.ShareTensor"),
    ("sympc.tensor.ShareTensor.shape", "syft.lib.python.Tuple"),
]

replicated_shared_tensor_attrs = [
    (
        "sympc.tensor.ReplicatedSharedTensor.__add__",
        "sympc.tensor.ReplicatedSharedTensor",
    ),
    (
        "sympc.tensor.ReplicatedSharedTensor.__sub__",
        "sympc.tensor.ReplicatedSharedTensor",
    ),
    (
        "sympc.tensor.ReplicatedSharedTensor.__rmul__",
        "sympc.tensor.ReplicatedSharedTensor",
    ),
    (
        "sympc.tensor.ReplicatedSharedTensor.__mul__",
        "sympc.tensor.ReplicatedSharedTensor",
    ),
    (
        "sympc.tensor.ReplicatedSharedTensor.__matmul__",
        "sympc.tensor.ReplicatedSharedTensor",
    ),
    (
        "sympc.tensor.ReplicatedSharedTensor.__truediv__",
        "sympc.tensor.ReplicatedSharedTensor",
    ),
    (
        "sympc.tensor.ReplicatedSharedTensor.__rmatmul__",
        "sympc.tensor.ReplicatedSharedTensor",
    ),
    (
        "sympc.tensor.ReplicatedSharedTensor.t",
        "sympc.tensor.ReplicatedSharedTensor",
    ),
    (
        "sympc.tensor.ReplicatedSharedTensor.sum",
        "sympc.tensor.ReplicatedSharedTensor",
    ),
    (
        "sympc.tensor.ReplicatedSharedTensor.clone",
        "sympc.tensor.ReplicatedSharedTensor",
    ),
    (
        "sympc.tensor.ReplicatedSharedTensor.numel",
        "syft.lib.python.Int",  # FIXME: Can't we just return an int??
    ),
    (
        "sympc.tensor.ReplicatedSharedTensor.T",
        "sympc.tensor.ReplicatedSharedTensor",
    ),
    (
        "sympc.tensor.ReplicatedSharedTensor.unsqueeze",
        "sympc.tensor.ReplicatedSharedTensor",
    ),
    ("sympc.tensor.ReplicatedSharedTensor.view", "sympc.tensor.ReplicatedSharedTensor"),
    ("sympc.tensor.ShareTensor.repeat", "sympc.tensor.ShareTensor"),
    ("sympc.tensor.ShareTensor.narrow", "sympc.tensor.ShareTensor"),
    ("sympc.tensor.ShareTensor.dim", "syft.lib.python.Int"),
    (
        "sympc.tensor.grads.grad_functions.GradConv2d.get_grad_input_padding",
        "sympc.tensor.ShareTensor",
    ),
    ("sympc.tensor.ShareTensor.transpose", "sympc.tensor.ShareTensor"),
    ("sympc.tensor.ShareTensor.flatten", "sympc.tensor.ShareTensor"),
]

allowed_external_attrs = [
    ("sympc.store.CryptoStore.get_primitives_from_store", "syft.lib.python.List"),
    ("sympc.store.CryptoStore.store", "syft.lib.python.Dict"),
    ("sympc.session.Session.crypto_store", "sympc.store.CryptoStore"),
    ("sympc.session.Session.init_generators", "syft.lib.python._SyNone"),
    ("sympc.session.Session.przs_generators", "syft.lib.python.List"),
    ("sympc.protocol.fss.fss.mask_builder", "sympc.tensor.ShareTensor"),
    ("sympc.protocol.fss.fss.evaluate", "sympc.tensor.ShareTensor"),
    ("sympc.protocol.spdz.spdz.mul_parties", "sympc.tensor.ShareTensor"),
    ("sympc.protocol.spdz.spdz.spdz_mask", "syft.lib.python.Tuple"),
    ("sympc.protocol.spdz.spdz.div_wraps", "sympc.tensor.ShareTensor"),
    (
        "sympc.session.Session.przs_generate_random_share",
        "sympc.tensor.ShareTensor",
    ),
    (
        "sympc.store.CryptoStore.populate_store",
        "syft.lib.python._SyNone",
    ),
    (
        "sympc.utils.get_new_generator",
        "torch.Generator",
    ),
]

allowed_external_attrs.extend(replicated_shared_tensor_attrs)
allowed_external_attrs.extend(share_tensor_attrs)

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

# third party
import syft as sy

import sympc

from . import protocol  # noqa: 401
from . import session  # noqa: 401
from . import store  # noqa: 401
from . import tensor  # noqa: 401
from . import utils  # noqa: 401

from . import grads  # noqa: 401 isort: skip
from . import module  # noqa: 401 isort: skip

allowed_external_modules = [
    ("sympc", sympc),
    ("sympc.session", session),
    ("sympc.tensor", tensor),
    ("sympc.tensor.share_tensor", tensor.share_tensor),
    ("sympc.tensor.replicatedshare_tensor", tensor.replicatedshare_tensor),
    ("sympc.tensor.static", tensor.static),
    ("sympc.protocol", protocol),
    ("sympc.module", module),
    ("sympc.module.nn", module.nn),
    ("sympc.module.nn.functional", module.nn.functional),
    ("sympc.store", store),
    ("sympc.protocol.falcon", protocol.falcon),
    ("sympc.protocol.falcon.falcon", protocol.falcon.falcon),
    ("sympc.protocol.falcon.falcon.Falcon", protocol.falcon.falcon.Falcon),
    ("sympc.protocol.aby3", protocol.aby3),
    ("sympc.protocol.aby3.aby3", protocol.aby3.aby3),
    ("sympc.protocol.aby3.aby3.ABY3", protocol.aby3.aby3.ABY3),
    ("sympc.protocol.fss", protocol.fss),
    ("sympc.protocol.fss.fss", protocol.fss.fss),
    ("sympc.protocol.spdz", protocol.spdz),
    ("sympc.protocol.spdz.spdz", protocol.spdz.spdz),
    ("sympc.utils", utils),
    ("sympc.grads", grads),
]

allowed_external_classes = [
    ("sympc.session.Session", "sympc.session.Session", session.Session),
    ("sympc.store.CryptoStore", "sympc.store.CryptoStore", store.CryptoStore),
    (
        "sympc.tensor.share_tensor.ShareTensor",
        "sympc.tensor.share_tensor.ShareTensor",
        tensor.share_tensor.ShareTensor,
    ),
    (
        "sympc.tensor.replicatedshare_tensor.ReplicatedSharedTensor",
        "sympc.tensor.replicatedshare_tensor.ReplicatedSharedTensor",
        tensor.replicatedshare_tensor.ReplicatedSharedTensor,
    ),
]

share_tensor_attrs = [
    (
        "sympc.tensor.share_tensor.ShareTensor.__add__",
        "sympc.tensor.share_tensor.ShareTensor",
    ),
    (
        "sympc.tensor.share_tensor.ShareTensor.__sub__",
        "sympc.tensor.share_tensor.ShareTensor",
    ),
    (
        "sympc.tensor.share_tensor.ShareTensor.__rmul__",
        "sympc.tensor.share_tensor.ShareTensor",
    ),
    (
        "sympc.tensor.share_tensor.ShareTensor.__mul__",
        "sympc.tensor.share_tensor.ShareTensor",
    ),
    (
        "sympc.tensor.share_tensor.ShareTensor.__matmul__",
        "sympc.tensor.share_tensor.ShareTensor",
    ),
    (
        "sympc.tensor.share_tensor.ShareTensor.__truediv__",
        "sympc.tensor.share_tensor.ShareTensor",
    ),
    (
        "sympc.tensor.share_tensor.ShareTensor.__rmatmul__",
        "sympc.tensor.share_tensor.ShareTensor",
    ),
    (
        "sympc.tensor.share_tensor.ShareTensor.t",
        "sympc.tensor.share_tensor.ShareTensor",
    ),
    (
        "sympc.tensor.share_tensor.ShareTensor.sum",
        "sympc.tensor.share_tensor.ShareTensor",
    ),
    (
        "sympc.tensor.share_tensor.ShareTensor.clone",
        "sympc.tensor.share_tensor.ShareTensor",
    ),
    (
        "sympc.tensor.share_tensor.ShareTensor.numel",
        "syft.lib.python.Int",  # FIXME: Can't we just return an int??
    ),
    (
        "sympc.tensor.share_tensor.ShareTensor.T",
        "sympc.tensor.share_tensor.ShareTensor",
    ),
    (
        "sympc.tensor.share_tensor.ShareTensor.squeeze",
        "sympc.tensor.share_tensor.ShareTensor",
    ),
    (
        "sympc.tensor.share_tensor.ShareTensor.unsqueeze",
        "sympc.tensor.share_tensor.ShareTensor",
    ),
    (
        "sympc.tensor.share_tensor.ShareTensor.reshape",
        "sympc.tensor.share_tensor.ShareTensor",
    ),
    (
        "sympc.tensor.share_tensor.ShareTensor.view",
        "sympc.tensor.share_tensor.ShareTensor",
    ),
    ("sympc.tensor.static.stack_share_tensor", "sympc.tensor.share_tensor.ShareTensor"),
    ("sympc.tensor.static.cat_share_tensor", "sympc.tensor.share_tensor.ShareTensor"),
    (
        "sympc.tensor.static.helper_argmax_pairwise",
        "sympc.tensor.share_tensor.ShareTensor",
    ),
    (
        "sympc.module.nn.functional.helper_max_pool2d_reshape",
        "sympc.tensor.share_tensor.ShareTensor",
    ),
    ("sympc.tensor.share_tensor.ShareTensor.shape", "syft.lib.python.Tuple"),
    (
        "sympc.tensor.share_tensor.ShareTensor.repeat",
        "sympc.tensor.share_tensor.ShareTensor",
    ),
    (
        "sympc.tensor.share_tensor.ShareTensor.narrow",
        "sympc.tensor.share_tensor.ShareTensor",
    ),
    ("sympc.tensor.share_tensor.ShareTensor.dim", "syft.lib.python.Int"),
    (
        "sympc.tensor.share_tensor.ShareTensor.transpose",
        "sympc.tensor.share_tensor.ShareTensor",
    ),
    (
        "sympc.tensor.share_tensor.ShareTensor.flatten",
        "sympc.tensor.share_tensor.ShareTensor",
    ),
]

replicated_shared_tensor_attrs = [
    (
        "sympc.tensor.replicatedshare_tensor.ReplicatedSharedTensor.__add__",
        "sympc.tensor.replicatedshare_tensor.ReplicatedSharedTensor",
    ),
    (
        "sympc.tensor.replicatedshare_tensor.ReplicatedSharedTensor.__sub__",
        "sympc.tensor.replicatedshare_tensor.ReplicatedSharedTensor",
    ),
    (
        "sympc.tensor.replicatedshare_tensor.ReplicatedSharedTensor.__rmul__",
        "sympc.tensor.replicatedshare_tensor.ReplicatedSharedTensor",
    ),
    (
        "sympc.tensor.replicatedshare_tensor.ReplicatedSharedTensor.__mul__",
        "sympc.tensor.replicatedshare_tensor.ReplicatedSharedTensor",
    ),
    (
        "sympc.tensor.replicatedshare_tensor.ReplicatedSharedTensor.__matmul__",
        "sympc.tensor.replicatedshare_tensor.ReplicatedSharedTensor",
    ),
    (
        "sympc.tensor.replicatedshare_tensor.ReplicatedSharedTensor.__truediv__",
        "sympc.tensor.replicatedshare_tensor.ReplicatedSharedTensor",
    ),
    (
        "sympc.tensor.replicatedshare_tensor.ReplicatedSharedTensor.__rshift__",
        "sympc.tensor.replicatedshare_tensor.ReplicatedSharedTensor",
    ),
    (
        "sympc.tensor.replicatedshare_tensor.ReplicatedSharedTensor.__floordiv__",
        "sympc.tensor.replicatedshare_tensor.ReplicatedSharedTensor",
    ),
    (
        "sympc.tensor.replicatedshare_tensor.ReplicatedSharedTensor.__rmatmul__",
        "sympc.tensor.replicatedshare_tensor.ReplicatedSharedTensor",
    ),
    (
        "sympc.tensor.replicatedshare_tensor.ReplicatedSharedTensor.__xor__",
        "sympc.tensor.replicatedshare_tensor.ReplicatedSharedTensor",
    ),
    (
        "sympc.tensor.replicatedshare_tensor.ReplicatedSharedTensor.__setitem__",
        "syft.lib.python._SyNone",
    ),
    (
        "sympc.tensor.replicatedshare_tensor.ReplicatedSharedTensor.__getitem__",
        "torch.Tensor",
    ),
    (
        "sympc.tensor.replicatedshare_tensor.ReplicatedSharedTensor.t",
        "sympc.tensor.replicatedshare_tensor.ReplicatedSharedTensor",
    ),
    (
        "sympc.tensor.replicatedshare_tensor.ReplicatedSharedTensor.sum",
        "sympc.tensor.replicatedshare_tensor.ReplicatedSharedTensor",
    ),
    (
        "sympc.tensor.replicatedshare_tensor.ReplicatedSharedTensor.clone",
        "sympc.tensor.replicatedshare_tensor.ReplicatedSharedTensor",
    ),
    (
        "sympc.tensor.replicatedshare_tensor.ReplicatedSharedTensor.get_shares",
        "syft.lib.python.List",
    ),
    (
        "sympc.tensor.replicatedshare_tensor.ReplicatedSharedTensor.numel",
        "syft.lib.python.Int",  # FIXME: Can't we just return an int??
    ),
    (
        "sympc.tensor.replicatedshare_tensor.ReplicatedSharedTensor.get_ring_size",
        "syft.lib.python.String",
    ),
    (
        "sympc.tensor.replicatedshare_tensor.ReplicatedSharedTensor.get_config",
        "syft.lib.python.Dict",
    ),
    (
        "sympc.tensor.replicatedshare_tensor.ReplicatedSharedTensor.bit_extraction",
        "sympc.tensor.replicatedshare_tensor.ReplicatedSharedTensor",
    ),
    (
        "sympc.tensor.replicatedshare_tensor.ReplicatedSharedTensor.T",
        "sympc.tensor.replicatedshare_tensor.ReplicatedSharedTensor",
    ),
    (
        "sympc.tensor.replicatedshare_tensor.ReplicatedSharedTensor.unsqueeze",
        "sympc.tensor.replicatedshare_tensor.ReplicatedSharedTensor",
    ),
    (
        "sympc.tensor.replicatedshare_tensor.ReplicatedSharedTensor.view",
        "sympc.tensor.replicatedshare_tensor.ReplicatedSharedTensor",
    ),
]

allowed_external_attrs = [
    (
        "sympc.module.nn.functional.max_pool2d_backward_helper",
        "sympc.tensor.share_tensor.ShareTensor",
    ),
    ("sympc.store.CryptoStore.get_primitives_from_store", "syft.lib.python.List"),
    ("sympc.store.CryptoStore.store", "syft.lib.python.Dict"),
    ("sympc.session.Session.crypto_store", "sympc.store.CryptoStore"),
    ("sympc.session.Session.init_generators", "syft.lib.python._SyNone"),
    ("sympc.session.Session.przs_generators", "syft.lib.python.List"),
    ("sympc.protocol.fss.fss.mask_builder", "sympc.tensor.share_tensor.ShareTensor"),
    ("sympc.protocol.fss.fss.evaluate", "sympc.tensor.share_tensor.ShareTensor"),
    ("sympc.protocol.spdz.spdz.mul_parties", "sympc.tensor.share_tensor.ShareTensor"),
    ("sympc.protocol.spdz.spdz.spdz_mask", "syft.lib.python.Tuple"),
    ("sympc.protocol.spdz.spdz.div_wraps", "sympc.tensor.share_tensor.ShareTensor"),
    ("sympc.protocol.falcon.falcon.Falcon.compute_zvalue_and_add_mask", "torch.Tensor"),
    ("sympc.protocol.falcon.falcon.Falcon.falcon_mask", "syft.lib.python.Tuple"),
    (
        "sympc.protocol.falcon.falcon.Falcon.triple_verification",
        "sympc.tensor.replicatedshare_tensor.ReplicatedSharedTensor",
    ),
    ("sympc.protocol.aby3.aby3.ABY3.local_decomposition", "syft.lib.python.List"),
    (
        "sympc.session.Session.przs_generate_random_share",
        sy.lib.misc.union.UnionGenerator[
            "sympc.tensor.replicatedshare_tensor.ReplicatedSharedTensor",
            "sympc.tensor.share_tensor.ShareTensor",
        ],
    ),
    (
        "sympc.session.Session.prrs_generate_random_share",
        sy.lib.misc.union.UnionGenerator[
            "sympc.tensor.replicatedshare_tensor.ReplicatedSharedTensor",
            "sympc.tensor.share_tensor.ShareTensor",
        ],
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

import sympc

from . import approximations  # noqa: 401
from . import config  # noqa: 401
from . import encoder  # noqa: 401
from . import protocol  # noqa: 401
from . import session  # noqa: 401
from . import store  # noqa: 401
from . import tensor  # noqa: 401
from . import utils  # noqa: 401

from . import module  # noqa: 401 isort: skip

allowed_external_modules = [
    ("sympc", sympc),
    ("sympc.session", session),
    ("sympc.tensor", tensor),
    ("sympc.protocol", protocol),
    ("sympc.store", store),
    ("sympc.protocol.fss", protocol.fss),
    ("sympc.protocol.fss.fss", protocol.fss.fss),
    ("sympc.protocol.spdz", protocol.spdz),
    ("sympc.protocol.spdz.spdz", protocol.spdz.spdz),
    ("sympc.utils", utils),
]

allowed_external_classes = [
    ("sympc.session.Session", "sympc.session.Session", session.Session),
    ("sympc.store.CryptoStore", "sympc.store.CryptoStore", store.CryptoStore),
    (
        "sympc.tensor.ShareTensor",
        "sympc.tensor.ShareTensor",
        tensor.ShareTensor,
    ),
]


allowed_external_attrs = [
    ("sympc.store.CryptoStore.get_primitives_from_store", "syft.lib.python.List"),
    ("sympc.store.CryptoStore.store", "syft.lib.python.Dict"),
    ("sympc.session.Session.crypto_store", "sympc.store.CryptoStore"),
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
        "sympc.tensor.ShareTensor.numel",
        "syft.lib.python.Int",  # FIXME: Can't we just return an int??
    ),
    (
        "sympc.tensor.ShareTensor.T",
        "sympc.tensor.ShareTensor",
    ),
    ("sympc.tensor.ShareTensor.unsqueeze", "sympc.tensor.ShareTensor"),
    ("sympc.tensor.ShareTensor.view", "sympc.tensor.ShareTensor"),
]

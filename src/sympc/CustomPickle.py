# stdlib
# stdlib
import copyreg
import sys
from typing import Any
from typing import Tuple

# third party
import syft as sy
from syft.core.node.common.client import Client
from syft.core.pointer.pointer import Pointer
from syft.proto.core.node.common.client_pb2 import Client as Client_PB
from syft.proto.core.pointer.pointer_pb2 import Pointer as Pointer_PB


from .session import Session
from .session import SessionManager

SessionPointer = ""


def unpickle_pointer_data(data: Any) -> Tuple:
    pointer_pb = Pointer_PB()
    pointer_pb.ParseFromString(data)
    return Pointer._proto2object(pointer_pb)


def pickle_pointer_obj(obj: Any) -> Tuple:
    return unpickle_pointer_data, (obj._object2proto().SerializeToString(),)


def unpickle_vm_data(data: Any) -> Tuple:
    client_pb = Client_PB()
    client_pb.ParseFromString(data)
    return Client._proto2object(client_pb)


def pickle_vm_obj(obj: Any) -> Tuple:
    return unpickle_vm_data, (obj._object2proto().SerializeToString(),)


# def custom_pickle():
client_vm = sy.VirtualMachine(name="alice").get_root_client()
session_t = Session(parties=[client_vm])
SessionManager.setup_mpc(session_t)
pointer = session_t.send(client_vm)

sys.modules["syft.proxy.sympc.sessio"] = sys.modules["sympc.CustomPickle"]

copyreg.pickle(type(pointer), pickle_pointer_obj)
copyreg.pickle(type(client_vm), pickle_vm_obj)

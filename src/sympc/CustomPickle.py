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



def custom_pickle(x: Any)-> None:
    session = x.session
    sess_pointer = type(session.session_ptrs[0])
    tensor_pointer = type(x.share_ptrs[0])
    #client_vm = type(session.parties[0])

    sys.modules["syft.proxy.sympc.session"] = sys.modules["sympc.CustomPickle"]
    sys.modules["syft.proxy.sympc.tensor.replicatedshare_tensor"] = sys.modules["sympc.CustomPickle"]
    sys.modules["syft.proxy.sympc.tensor.share_tensor"] = sys.modules["sympc.CustomPickle"]

    global SessionPointer
    global ReplicatedSharedTensorPointer
    global ShareTensorPointer

    SessionPointer= sess_pointer

    tensor_name = tensor_pointer.__name__

    if(tensor_name == "ReplicatedSharedTensorPointer"):
        ReplicatedSharedTensorPointer=tensor_pointer
    elif(tensor_name == "ShareTensorPointer"):
        ShareTensorPointer = tensor_pointer
    else:
        raise ValueError(f"Invalid Pointer Type{tensor_name} for pickling.")


    copyreg.pickle(sess_pointer, pickle_pointer_obj)
    #copyreg.pickle(client_vm, pickle_vm_obj)

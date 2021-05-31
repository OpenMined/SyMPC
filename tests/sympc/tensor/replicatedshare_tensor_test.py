from sympc.tensor import ReplicatedSharedTensor
import pytest
import torch 
from sympc.session import Session
from sympc.session import SessionManager

def test_import_RSTensor():

    ReplicatedSharedTensor()

def test_send_get(get_clients, precision=12, base=4) -> None:

    client = get_clients(1)[0]
    session = Session(parties=[client])
    SessionManager.setup_mpc(session)
    share1 = torch.Tensor([1.4,2.34,3.43])
    share2 = torch.Tensor([1,2,3])
    share3 = torch.Tensor([1.4,2.34,3.43])
    x_share = ReplicatedSharedTensor(shares=[share1,share2,share3],session=session)
    x_ptr = x_share.send(client)
    result=x_ptr.get()
    
    assert result==x_share



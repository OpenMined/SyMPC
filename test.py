import syft as sy
from sympc.session import Session
from sympc.session import SessionManager
from sympc.module import SyMPCModule

alice = sy.VirtualMachine(name="alice")
bob = sy.VirtualMachine(name="bob")

clients = [alice.get_client(), bob.get_client()]

session = Session(parties=clients)
SessionManager.setup_mpc(session)

module = SyMPCModule(session)

print("Hello")

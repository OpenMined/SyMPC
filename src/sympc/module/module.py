from syft.lib.torch.module import Module

class SyMPCModule(Module):
    def __init__(self, session) -> None:
        parties = session.parties
        nr_parties = session.nr_parties
        import pdb; pdb.set_trace()

        module_ptrs = [Module(parties[i].torch) for i in range(nr_parties)]

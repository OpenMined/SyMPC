from sympc.tensor import MPCTensor


def relu(x: MPCTensor) -> MPCTensor:
    session = x.session
    protocol = session.protocol
    return protocol.relu(x)

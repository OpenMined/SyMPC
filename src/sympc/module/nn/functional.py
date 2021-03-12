def relu(x):
    session = x.session
    protocol = session.protocol
    return protocol.relu(x)

def sign(data):
    return (data > 0) + (data < 0) * (-1)


def modulus(data):
    """Calculation of modulus for a given tensor."""
    return data.signum() * data


def signum(data):
    """Calculation of signum function for a given tensor."""
    sgn = (data > 0) - (data < 0)
    return sgn

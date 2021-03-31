"""Utility functions for approximation functions."""


def sign(data):
    """Calculate sign of given tensor.

    Args:
        data: tensor whose sign has to be determined

    Returns:
        MPCTensor: tensor with the determined sign
    """
    return (data > 0) + (data < 0) * (-1)


def modulus(data):
    """Calculation of modulus for a given tensor.

    Args:
        data(MPCTensor): tensor whose modulus has to be calculated

    Returns:
        MPCTensor: the required modulus

    """
    return signum(data) * data


def signum(data):
    """Calculation of signum function for a given tensor.

    Args:
        data(MPCTensor): tensor whose ssignum has to be calculated

    Returns:
        MPCTensor: the required signum

    """
    sgn = (data > 0) - (data < 0)
    return sgn

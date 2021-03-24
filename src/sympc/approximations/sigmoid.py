# third party
import torch


def sigmoid(tensor, method: str = "exp"):
    """
    Approximates the sigmoid function using a given method
    Args:
        tensor: the fixed precision tensor
        method (str): (default = "chebyshev")
            Possible values: "exp", "maclaurin", "chebyshev"
    """

    if method == "maclaurin":
        weights = torch.tensor([0.5, 1.91204779e-01, -4.58667307e-03, 4.20690803e-05])
        degrees = [0, 1, 3, 5]

        # initiate with term of degree 0 to avoid errors with tensor ** 0
        one = tensor * 0 + 1
        result = one * weights[0]
        for i, d in enumerate(degrees[1:]):
            result += (tensor ** d) * weights[i + 1]

        return result

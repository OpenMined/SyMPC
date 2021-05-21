"""Stochastic Gradient Descent."""

# stdlib
from typing import List

from sympc.tensor import MPCTensor


class SGD:
    """The SGD Optimizer Class."""

    __slots__ = ["_lr", "_parameters"]

    def __init__(self, parameters: List[MPCTensor], lr: float) -> None:
        """Initialize a new SGD optimizer.

        Args:
            parameters (List[MPCTensor]): Parameters of the module to optimize
            lr (float): learning rate
        """
        self._lr = lr
        self._parameters = parameters

    def step(self):
        """Perform one step of gradient descent."""
        MPCTensor.AUTOGRAD_IS_ON = False
        for param in self._parameters:
            grad = param.grad

            # We need to use inplace operations such that we
            # also update the parameters from the "model"
            # Doing:
            # param -= grad * self._lr would return a new MPCTensor
            param.isub(grad * self._lr)
        MPCTensor.AUTOGRAD_IS_ON = True

    def zero_grad(self):
        """Set the gradients from the parameters to to zero/None."""
        for param in self._parameters:
            param.grad = None

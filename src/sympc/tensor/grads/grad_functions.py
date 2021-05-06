"""Definitions for the gradient functions and the forward pass logic."""

# stdlib
from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

# third party
import torch

from sympc.tensor import MPCTensor


class GradFunc(ABC):
    """Abstract class that should be implemented by the Gradient Functions."""

    @staticmethod
    @abstractmethod
    def forward(
        ctx: Dict[str, Any], *args: List[Any], **kwargs: Dict[str, Any]
    ) -> None:
        """Perform the feedforward and compute the result for the implemented operation.

        Args:
            ctx (Dict[str, Any]): Context used to save information needed in the backward pass
            *args (List[Any]): Arguments for the operation
            **kwargs (Dict[str, Any]): Named arguments for the operation

        Raises:
            NotImplementedError: The operation should be implemented
        """
        raise NotImplementedError("Forward method not implemented!")

    @staticmethod
    @abstractmethod
    def backward(
        ctx: Dict[str, Any], *args: List[Any], **kwargs: Dict[str, Any]
    ) -> None:
        """Perform the backward pass and compute the gradient for the implemented operation.

        Args:
            ctx (Dict[str, Any]): Context used to save information needed in the backward pass
            *args (List[Any]): Arguments for the operation
            **kwargs (Dict[str, Any]): Named arguments for the operation

        Raises:
            NotImplementedError: The operation should be implemented
        """
        raise NotImplementedError("Backward method not implemented!")


class GradT(GradFunc):
    """The tranpose gradient function."""

    @staticmethod
    def forward(ctx: Dict[str, Any], x: MPCTensor) -> MPCTensor:
        """Perform the feedforward and compute the result for the transpose operation.

        Args:
            ctx (Dict[str, Any]): Context used to save information needed in the backward pass
            x (MPCTensor): The operand to apply the transpose on

        Returns:
            x.t() (MPCTensor): The result after applying transpose
        """
        return x.t()

    @staticmethod
    def backward(ctx: Dict[str, Any], grad: MPCTensor) -> MPCTensor:
        """Perform the backward pass for the transpose operation.

        Args:
            ctx (Dict[str, Any]): Context used to retrieve the information for the backward pass
            grad (MPCTensor): The gradient that came from the child nodes

        Returns:
            res_grad (MPCTensor): The gradients passed to the parent node
        """
        res_grad = grad.t()
        return res_grad


class GradAdd(GradFunc):
    """The addition gradient function."""

    @staticmethod
    def forward(ctx: Dict[str, Any], x: MPCTensor, y: Any) -> MPCTensor:
        """Perform the feedforward and compute the result for the addition operation.

        Args:
            ctx (Dict[str, Any]): Context used to save information needed in the backward pass
            x (MPCTensor): 1st operand for the addition operation
            y (Any): 2nd operand for the addition operation

        Returns:
            x + y (MPCTensor): The result of the addition

        Raises:
            ValueError: The shapes for the operands are not the same
        """
        # TODO: Fix for different shapes
        if x.shape != y.shape:
            raise ValueError("X and Y should have the same shape")

        return x + y

    @staticmethod
    def backward(ctx: Dict[str, Any], grad: MPCTensor) -> Tuple[MPCTensor]:
        """Perform the backward pass for the addition operation.

        Args:
            ctx (Dict[str, Any]): Context used to retrieve the information for the backward pass
            grad (MPCTensor): The gradient that came from the child nodes

        Returns:
            (x_grad, y_grad) (Tuple[MPCTensor]): The gradients passed to the parent node
        """
        x_grad = grad
        y_grad = grad.clone()
        return x_grad, y_grad


class GradSum(GradFunc):
    """The summation gradient function."""

    @staticmethod
    def forward(ctx: Dict[str, Any], x: MPCTensor) -> MPCTensor:
        """Perform the feedforward and compute the result for the summation operation.

        Args:
            ctx (Dict[str, Any]): Context used to save information needed in the backward pass
            x (MPCTensor): The operand on which to apply the sum function

        Returns:
            sum(x) (MPCTensor): The summation of all the elements from the input
        """
        ctx["x_shape"] = x.shape
        total_sum = x.sum()
        return total_sum

    @staticmethod
    def backward(ctx: Dict[str, Any], grad: MPCTensor) -> MPCTensor:
        """Perform the backward pass for the summation operation.

        Args:
            ctx (Dict[str, Any]): Context used to retrieve the information for the backward pass
            grad (MPCTensor): The gradient that came from the child nodes

        Returns:
            res_grad (MPCTensor): The gradients passed to the parent node
        """
        x_shape = ctx["x_shape"]
        res_grad = grad * torch.ones(size=x_shape)
        return res_grad


class GradSigmoid(GradFunc):
    """The sigmoid gradient function."""

    @staticmethod
    def forward(ctx: Dict[str, Any], x: MPCTensor) -> MPCTensor:
        """Perform the feedforward and compute the result for the sigmoid operation.

        Args:
            ctx (Dict[str, Any]): Context used to save information needed in the backward pass
            x (MPCTensor): The operand on which to apply the sigmoid function

        Returns:
            sigmoid(x) (MPCTensor): The sigmoid approximation applied on the input
        """
        grad = x.sigmoid()
        ctx["probabilities"] = grad
        return grad

    @staticmethod
    def backward(ctx: Dict[str, Any], grad: MPCTensor) -> MPCTensor:
        """Perform the backward pass for the substraction operation.

        Args:
            ctx (Dict[str, Any]): Context used to retrieve the information for the backward pass
            grad (MPCTensor): The gradient that came from the child nodes

        Returns:
            res_grad (MPCTensor): The gradient passed to the parent node
        """
        probs = ctx["probabilities"]
        res_grad = grad * probs * (1 - probs)
        return res_grad


class GradSub(GradFunc):
    """The substraction gradient function."""

    @staticmethod
    def forward(ctx: Dict[str, Any], x: MPCTensor, y: Any) -> MPCTensor:
        """Perform the feedforward and compute the result for the substraction operation.

        Args:
            ctx (Dict[str, Any]): Context used to save information needed in the backward pass
            x (MPCTensor): 1st operand for the substraction operation
            y (Any): 2nd operand for the substraction operation

        Returns:
            x - y (MPCTensor): The result of the substraction

        Raises:
            ValueError: The shapes for the operands are not the same
        """
        # TODO: Fix for different shapes
        if x.shape != y.shape:
            raise ValueError("X and Y should have the same shape")

        return x - y

    @staticmethod
    def backward(ctx: Dict[str, Any], grad: MPCTensor) -> Tuple[MPCTensor]:
        """Perform the backward pass for the substraction  operation.

        Args:
            ctx (Dict[str, Any]): Context used to retrieve the information for the backward pass
            grad (MPCTensor): The gradient that came from the child nodes

        Returns:
            (x_grad, y_grad) (Tuple[MPCTensor]): The gradients passed to the X and Y nodes.
        """
        x_grad = grad
        y_grad = grad.clone()
        return x_grad, y_grad


class GradMul(GradFunc):
    """The multiplication gradient function."""

    @staticmethod
    def forward(ctx: Dict[str, Any], x: MPCTensor, y: Any) -> MPCTensor:
        """Perform the feedforward and compute the result for the multiplication operation.

        Args:
            ctx (Dict[str, Any]): Context used to save information needed in the backward pass
            x (MPCTensor): 1st operand for the multiplication operation
            y (Any): 2nd operand for the multiplication operation

        Returns:
            x * y (MPCTensor): The result of the multiplication

        Raises:
            ValueError: The shapes for the operands are not the same
        """
        # TODO: Tackle the broadcast step
        # Make sure we do not broadcast because we would need to deal with this
        # in the backward
        if x.shape != y.shape:
            raise ValueError("X and Y should have the same shape")

        ctx["x"] = x
        ctx["y"] = y
        return x * y

    @staticmethod
    def backward(ctx: Dict[str, Any], grad: MPCTensor) -> Tuple[MPCTensor]:
        """Perform the backward pass for the multiplication operation.

        Args:
            ctx (Dict[str, Any]): Context used to retrieve the information for the backward pass
            grad (MPCTensor): The gradient that came from the child nodes

        Returns:
            (y_grad, x_grad) (Tuple[MPCTensor]): The gradients passed to the X and Y nodes.

        Raises:
            ValueError: if gradient shape does not match X and Y shape
        """
        x, y = ctx["x"], ctx["y"]

        # TODO: Fix for different shapes
        if x.shape != grad.shape or y.shape != grad.shape:
            raise ValueError(
                "The gradient shape and the shape of X and Y should be the same!"
            )
        x_grad = grad * x
        y_grad = grad * y
        return y_grad, x_grad


def forward(
    _self: MPCTensor, grad_fn: GradFunc, *args: List[Any], **kwargs: Dict[str, Any]
) -> Any:
    """Perform the forward pass and construct the computational graph.

    Args:
        _self (MPCTensor): Perform
        grad_fn (GradFunc): The gradient function to use
        *args (List[Any]): Arguments for the gradient function
        **kwargs (Dict[str, Any]): Named arguments for the gradient function

    Returns:
        res (MPCTensor): The result of the computation
    """
    mpc_tensor_params = [_self] + [arg for arg in args if isinstance(arg, MPCTensor)]
    requires_grad = any(mpc_tensor.requires_grad for mpc_tensor in mpc_tensor_params)

    MPCTensor.AUTOGRAD_IS_ON = False
    res = grad_fn.forward(_self.ctx, _self, *args, **kwargs)
    MPCTensor.AUTOGRAD_IS_ON = True

    res.requires_grad = requires_grad
    res.grad_fn = grad_fn
    res.ctx = _self.ctx.copy()
    res.parents = mpc_tensor_params
    for mpc_tensor in mpc_tensor_params:
        mpc_tensor.nr_out_edges += 1
    return res


GRAD_FUNCS: Dict[str, GradFunc] = {
    "t": GradT,
    "mul": GradMul,
    "sub": GradSub,
    "add": GradAdd,
    "sum": GradSum,
    "sigmoid": GradSigmoid,
}

"""Definitions for the gradient functions and the forward pass logic.

This is how PyTorch implements new gradient function:
https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html

And also it seems that CrypTen follows the same logic
"""

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


def _reverse_broadcast(x: MPCTensor, wanted_shape: Tuple[int, ...]) -> MPCTensor:
    """Reverse the broadcast operation.

    Credits goes to the CrypTen team.

    Args:
        x (MPCTensor): The value to be "reverse broadcasted"
        wanted_shape (Tuple[int, ...]): The desired shape for the output

    Returns:
        An MPCTensor with the wanted shape
    """
    res = x

    # We know that the "ending" dimensions should match, so we look at how many
    # of the "leading" dimensions do not match
    if len(wanted_shape) < len(x.shape):
        range_dim = list(range(len(x.shape) - len(wanted_shape)))
        res = res.sum(range_dim)

    for dim in range(len(res.shape)):
        if wanted_shape[dim] == 1 and res.shape[dim] > 1:
            res = res.sum(dim, keepdim=True)

    return res


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
        """
        ctx["x_shape"] = x.shape
        ctx["y_shape"] = y.shape

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
        x_shape, y_shape = ctx["x_shape"], ctx["y_shape"]

        x_grad = _reverse_broadcast(grad, x_shape)
        y_grad = _reverse_broadcast(grad, y_shape)

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
        """
        ctx["x_shape"] = x.shape
        ctx["y_shape"] = y.shape

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
        x_shape, y_shape = ctx["x_shape"], ctx["y_shape"]

        x_grad = _reverse_broadcast(grad, x_shape)
        y_grad = _reverse_broadcast(grad, y_shape)
        return x_grad, -1 * y_grad


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
        """
        if not hasattr(x, "shape"):
            x = torch.tensor(x)
        if not hasattr(y, "shape"):
            y = torch.tensor(y)

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
            (x_grad, y_grad) (Tuple[MPCTensor]): The gradients passed to the X and Y nodes.
        """
        x, y = ctx["x"], ctx["y"]

        x_grad = _reverse_broadcast(grad * y, x.shape)
        y_grad = _reverse_broadcast(grad * x, y.shape)
        return x_grad, y_grad


class GradPow(GradFunc):
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
        """
        ctx["x"] = x
        ctx["y"] = y
        return x ** y

    @staticmethod
    def backward(ctx: Dict[str, Any], grad: MPCTensor) -> Tuple[MPCTensor]:
        """Perform the backward pass for the multiplication operation.

        Args:
            ctx (Dict[str, Any]): Context used to retrieve the information for the backward pass
            grad (MPCTensor): The gradient that came from the child nodes

        Returns:
            (x_grad, y_grad) (Tuple[MPCTensor]): The gradients passed to the X and Y nodes.
        """
        x, y = ctx["x"], ctx["y"]

        return x ** (y - 1) * y * grad


class GradMatMul(GradFunc):
    """The multiplication gradient function."""

    @staticmethod
    def forward(ctx: Dict[str, Any], x: MPCTensor, y: Any) -> MPCTensor:
        """Perform the feedforward and compute the result for the matrix multiplication operation.

        Args:
            ctx (Dict[str, Any]): Context used to save information needed in the backward pass
            x (MPCTensor): 1st operand for the matrix multiplication operation
            y (Any): 2nd operand for the matrix multiplication operation

        Returns:
            x @ y (MPCTensor): The result of the matrix multiplication
        """
        ctx["x"] = x
        ctx["y"] = y
        return x @ y

    @staticmethod
    def backward(ctx: Dict[str, Any], grad: MPCTensor) -> Tuple[MPCTensor]:
        """Perform the backward pass for the matrix multiplication operation.

        Args:
            ctx (Dict[str, Any]): Context used to retrieve the information for the backward pass
            grad (MPCTensor): The gradient that came from the child nodes

        Returns:
            (x_grad, y_grad) (Tuple[MPCTensor]): The gradients passed to the X and Y nodes.

        Raises:
            ValueError: if gradient shape does not match X and Y shape
        """
        x, y = ctx["x"], ctx["y"]

        x_grad = grad.clone()
        y_grad = grad.clone()

        if len(x.shape) < 2:
            x = x.unsqueeze(0)
            x_grad = x_grad.unsqueeze(0)

        if len(y.shape) < 2:
            y = y.unsqueeze(1)
            y_grad = y_grad.unsqueeze(1)

        x_grad = x_grad @ y.t()
        y_grad = x.t() @ y_grad

        if x.shape != x_grad.shape or y.shape != y_grad.shape:
            raise ValueError(
                "The gradient shape and the shape of X and Y should be the same!"
            )

        return x_grad, y_grad


class GradFlatten(GradFunc):
    """The Flatten gradient function."""

    @staticmethod
    def forward(
        ctx: Dict[str, Any], x: MPCTensor, start: int = 0, end: int = -1
    ) -> MPCTensor:
        """Perform the feedforward and compute the result for the multiplication operation.

        Args:
            ctx (Dict[str, Any]): Context used to save information needed in the backward pass
            x (MPCTensor): 1st operand for the multiplication operation
            start (int): Start dimension for the flatten operation
            end (int): Final dimension for the flatten operation

        Returns:
            res (MPCTensor): The result of the flatten operation
        """
        ctx["x_shape"] = x.shape
        return x.flatten(start_dim=start, end_dim=end)

    @staticmethod
    def backward(ctx: Dict[str, Any], grad: MPCTensor) -> MPCTensor:
        """Perform the backward pass for the flaten operation.

        Args:
            ctx (Dict[str, Any]): Context used to retrieve the information for the backward pass
            grad (MPCTensor): The gradient that came from the child nodes

        Returns:
            grad (MPCTensor): The gradients passed to the X node.
        """
        shape = tuple(ctx["x_shape"])
        return grad.reshape(shape)


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

    _self.session.autograd_active = False
    ctx = {}
    res = grad_fn.forward(ctx, _self, *args, **kwargs)
    _self.session.autograd_active = True

    res.requires_grad = requires_grad
    res.grad_fn = grad_fn
    res.ctx = ctx
    res.parents = mpc_tensor_params
    for mpc_tensor in mpc_tensor_params:
        mpc_tensor.nr_out_edges += 1
    return res


GRAD_FUNCS: Dict[str, GradFunc] = {
    "t": GradT,
    "mul": GradMul,
    "pow": GradPow,
    "matmul": GradMatMul,
    "sub": GradSub,
    "add": GradAdd,
    "sum": GradSum,
    "sigmoid": GradSigmoid,
    "flatten": GradFlatten,
}

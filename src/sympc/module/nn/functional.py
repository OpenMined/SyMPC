"""Implementations for torch.nn.functional equivalent for MPC."""

# stdlib
from typing import Optional
from typing import Tuple
from typing import Union

# third party
import numpy as np
import torch

from sympc.grads import GRAD_FUNCS
from sympc.grads import forward
from sympc.session import get_session
from sympc.tensor import MPCTensor
from sympc.tensor import ShareTensor
from sympc.utils import parallel_execution


def sigmoid(x: MPCTensor) -> MPCTensor:
    """Sigmoid function.

    Args:
        x (MPCTensor): The tensor on which we apply the function

    Returns:
        An MPCTensor which represents the sigmoid applied on the input tensor
    """
    from sympc.approximations.sigmoid import sigmoid

    sigmoid_forward = GRAD_FUNCS.get("sigmoid", None)
    if sigmoid_forward and x.session.autograd_active:
        return forward(x, sigmoid_forward)
    res = sigmoid(x)
    return res


def relu(x: MPCTensor) -> MPCTensor:
    """Rectified linear unit function.

    Args:
        x (MPCTensor): The tensor on which we apply the function

    Returns:
        An MPCTensor which represents the ReLu applied on the input tensor
    """
    relu_forward = GRAD_FUNCS.get("relu", None)
    if relu_forward and x.session.autograd_active:
        return forward(x, relu_forward)
    res = x * (x >= 0)
    return res


def mse_loss(pred: MPCTensor, target: MPCTensor, reduction: str = "mean") -> MPCTensor:
    """Mean Squared Error loss.

    Args:
        pred (MPCTensor): The predictions obtained
        target (MPCTensor): The target values
        reduction (str): the reduction method, default is `mean`

    Returns:
        The loss

    Raises:
        ValueError: If `reduction` not in supported methods
    """
    if reduction == "mean":
        result = (pred - target).pow(2).sum() / pred.shape[0]
    elif reduction == "sum":
        result = (pred - target).pow(2).sum()
    else:
        raise ValueError("do not support reduction method: %s" % reduction)
    return result


Kernel2D = Tuple[int, int]
Stride2D = Tuple[int, int]
Padding2D = Tuple[int, int]
Dilation2D = Tuple[int, int]
MaxPool2DArgs = Tuple[Kernel2D, Stride2D, Padding2D, Dilation2D]


def _sanity_check_max_pool2d(
    kernel_size: Union[int, Tuple[int, int]],
    stride: Optional[Union[int, Tuple[int, int]]] = None,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
) -> MaxPool2DArgs:
    """Sanity check the parameters required for max_pool2d (backward and forward pass).

    Args:
        kernel_size (Union[int, Tuple[int, int]]): the kernel size
            in case it is passed as an integer then that specific value is used for height and width
        stride (Union[int, Tuple[int, int]]): the stride size
            in case it is passed as an integer then that specific value is used for height and width
        padding (Union[int, Tuple[int, int]]): the padding size
            in case it is passed as an integer then that specific value is used for height and width
        dilation (Union[int, Tuple[int, int]]): the dilation size
            in case it is passed as an integer then that specific value is used for height and width

    Returns:
        A 4 element type with types Tuple[int, int] representing the converted parameters.

    Raises:
        ValueError: if the parameters are not passing the sanity check
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if len(kernel_size) != 2:
        raise ValueError("Kernel_size should have only 2 dimensions")

    if stride is None:
        stride = kernel_size

    if isinstance(stride, int):
        stride = (stride, stride)

    if len(stride) != 2:
        raise ValueError("Stride should have only 2 dimensions")

    if isinstance(padding, int):
        padding = (padding, padding)

    if padding[0] > kernel_size[0] or padding[1] > kernel_size[1]:
        raise ValueError("Padding should be <= kernel_size / 2")

    if len(padding) != 2:
        raise ValueError("Padding should have only 2 dimensions")

    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    if len(dilation) != 2:
        raise ValueError("Dilation should have only 2 dimensions")

    if dilation[0] != 1 or dilation[1] != 1:
        raise ValueError("Supported only dilation == 1")

    return kernel_size, stride, padding, dilation


def _reshape_max_pool2d(
    x: MPCTensor,
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int],
) -> MPCTensor:
    """Prepare the share tensors by calling the reshape function in parallel at each party.

    Args:
        x (MPCTensor): the MPCTensor on which to apply the reshape operation
        kernel_size (Tuple[int, int]): the kernel size
        stride (Tuple[int, int]): the stride size
        padding (Tuple[int, int]): the padding size
        dilation (Tuple[int, int]): the dilation size

    Returns:
        The reshaped MPCTensor.
    """
    session = x.session

    args = [[share, kernel_size, stride, padding, dilation] for share in x.share_ptrs]
    shares = parallel_execution(helper_max_pool2d_reshape, session.parties)(args)

    res_shape = shares[0].shape.get()
    res = MPCTensor(shares=shares, session=session, shape=res_shape)
    return res


def helper_max_pool2d_reshape(
    x: ShareTensor,
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int],
) -> ShareTensor:
    """Function that runs at each party for preparing the share.

    Reshape each share tensor to prepare it for calling 'argmax'.
    The new share would have "each element" as the input on which we
    will run the max_pool2d kernel.

    Args:
        x (ShareTensor): the ShareTensor on which to apply the reshaping
        kernel_size (Tuple[int, int]): the kernel size
        stride (Tuple[int, int]): the stride size
        padding (Tuple[int, int]): the padding size
        dilation (Tuple[int, int]): the dilation size

    Returns:
        The prepared share tensor (reshaped)
    """
    session = get_session(x.session_uuid)
    tensor = x.tensor.numpy()

    padding = [(0, 0)] * len(tensor.shape[:-2]) + [
        (padding[0], padding[0]),
        (padding[1], padding[1]),
    ]
    tensor_type = session.tensor_type

    padding_value = 0
    if session.rank == 0:
        # ATTENTION: Min value for max_pool2d that works -25
        padding_value = -25

    tensor = np.pad(tensor, padding, mode="constant", constant_values=padding_value)

    output_shape = tensor.shape[:-2]
    output_shape += (
        (tensor.shape[-2] - kernel_size[0]) // stride[0] + 1,
        (tensor.shape[-1] - kernel_size[1]) // stride[1] + 1,
    )
    output_shape += kernel_size

    output_strides = tensor.strides[:-2]
    output_strides += (stride[0] * tensor.strides[-2], stride[1] * tensor.strides[-1])
    output_strides += tensor.strides[-2:]

    window_view_share = torch.tensor(
        np.lib.stride_tricks.as_strided(
            tensor, shape=output_shape, strides=output_strides
        ),
        dtype=tensor_type,
    )

    window_view_share = window_view_share.reshape(-1, *kernel_size)

    res_share = ShareTensor(config=x.config)
    res_share.tensor = window_view_share
    return res_share


def max_pool2d(
    x: MPCTensor,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Optional[Union[int, Tuple[int, int]]] = None,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    return_indices: bool = False,
) -> Union[MPCTensor, Tuple[MPCTensor, MPCTensor]]:
    """Compute the max pool for a tensor with 2 dimension.

    Args:
        x (MPCTensor): the MPCTensor on which to apply the operation
        kernel_size (Union[int, Tuple[int, int]]): the kernel size
            in case it is passed as an integer then that specific value is used for height and width
        stride (Union[int, Tuple[int, int]]): the stride size
            in case it is passed as an integer then that specific value is used for height and width
        padding (Union[int, Tuple[int, int]]): the padding size
            in case it is passed as an integer then that specific value is used for height and width
        dilation (Union[int, Tuple[int, int]]): the dilation size
            in case it is passed as an integer then that specific value is used for height and width
        return_indices (bool): to return the indices of the max values

    Returns:
        A tuple representing maximum values and the indices (as a one hot encoding

    Raises:
        ValueError: if the kernel size is bigger than the input
    """
    if x.session.nr_parties != 2:
        raise ValueError("Maxpool currently has support for only two parties.")

    max_pool2d_forward = GRAD_FUNCS.get("max_pool2d", None)
    if max_pool2d_forward and x.session.autograd_active:
        return forward(
            x,
            max_pool2d_forward,
            kernel_size,
            stride,
            padding,
            dilation,
            return_indices,
        )

    kernel_size, stride, padding, dilation = _sanity_check_max_pool2d(
        kernel_size, stride, padding, dilation
    )

    if (
        x.shape[-2] + 2 * padding[0] < kernel_size[0]
        or x.shape[-1] + 2 * padding[1] < kernel_size[1]
    ):
        raise ValueError(
            f"Kernel size ({kernel_size}) has more elements on an axis than "
            f"input shape ({x.shape}) considering padding of {padding}"
        )

    x_reshaped = _reshape_max_pool2d(
        x, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation
    )

    res_max_columns, columns = x_reshaped.max(dim=-1, one_hot=True)
    res_max, rows = res_max_columns.max(dim=-1, one_hot=True)

    output_shape = x.shape[:-2] + (
        (x.shape[-2] - kernel_size[0] + 2 * padding[0]) // stride[0] + 1,
        (x.shape[-1] - kernel_size[1] + 2 * padding[1]) // stride[1] + 1,
    )

    res = res_max.reshape(*output_shape)
    if return_indices:
        indices = columns * rows.unsqueeze(-1)
        res = (res, indices.reshape(output_shape + kernel_size))

    return res


def max_pool2d_backward_helper(
    input_shape: Tuple[int],
    grads_share: ShareTensor,
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
) -> ShareTensor:
    """Helper function to compute the gradient needed to be passed to the parent node.

    Args:
        input_shape (Tuple[int]): the size of the input tensor when running max_pool2d
        grads_share (ShareTensor): the share for the output gradient specific to this party
        kernel_size (Tuple[int, int]): the kernel size
        stride (Tuple[int, int]): the stride size
        padding (Tuple[int, int]): the padding size

    Returns:
        A ShareTensor specific for the computed gradient

    Raises:
        ValueError: if the input shape (taken into consideration the padding) is smaller than the
            kernel shape
    """
    session = get_session(str(grads_share.session_uuid))

    res_shape = input_shape[:-2]
    res_shape += (input_shape[-2] + 2 * padding[0], input_shape[-1] + 2 * padding[1])

    if res_shape[-2] < kernel_size[0] or res_shape[-1] < kernel_size[1]:
        raise ValueError(
            f"Kernel size ({kernel_size}) has more elements on an axis than "
            f"input shape ({res_shape}) considering padding of {padding}"
        )

    tensor_type = session.tensor_type
    tensor = torch.zeros(res_shape, dtype=tensor_type)

    for i in range((res_shape[-2] - kernel_size[0]) // stride[0] + 1):
        row_idx = i * stride[0]
        for j in range((res_shape[-1] - kernel_size[1]) // stride[1] + 1):
            col_idx = j * stride[1]
            if len(res_shape) == 4:
                tensor[
                    :,
                    :,
                    row_idx : row_idx + kernel_size[0],
                    col_idx : col_idx + kernel_size[1],
                ] += grads_share.tensor[:, :, i, j]
            else:
                tensor[
                    :,
                    row_idx : row_idx + kernel_size[0],
                    col_idx : col_idx + kernel_size[1],
                ] += grads_share.tensor[:, i, j]

    if len(res_shape) == 4:
        tensor = tensor[
            :, :, padding[0] : input_shape[-2], padding[1] : input_shape[-1]
        ]
    else:
        tensor = tensor[
            :,
            padding[0] : res_shape[-2] - padding[0],
            padding[1] : res_shape[-1] - padding[1],
        ]
    res = ShareTensor(config=grads_share.config)
    res.tensor = tensor

    return res


def max_pool2d_backward(
    grad: MPCTensor,
    input_shape: Tuple[int],
    indices: MPCTensor,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Optional[Union[int, Tuple[int, int]]] = None,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
) -> MPCTensor:
    """Helper function for the backwards step for max_pool2d.

    Credits goes to the CrypTen team.

    Args:
        grad (MPCTensor): gradient that comes from the child node
        input_shape (Tuple[int]): the shape of the input when the max_pool2d was run
        indices (MPCTensor): the indices where the maximum value was found in the input
        kernel_size (Union[int, Tuple[int, int]]): the kernel size
            in case it is passed as an integer then that specific value is used for height and width
        stride (Union[int, Tuple[int, int]]): the stride size
            in case it is passed as an integer then that specific value is used for height and width
        padding (Union[int, Tuple[int, int]]): the padding size
            in case it is passed as an integer then that specific value is used for height and width
        dilation (Union[int, Tuple[int, int]]): the dilation size
            in case it is passed as an integer then that specific value is used for height and width

    Returns:
        The gradient that should be backpropagated (MPCTensor)

    Raises:
        ValueError: In case some of the values for the parameters are not supported
    """
    kernel_size, stride, padding, dilation = _sanity_check_max_pool2d(
        kernel_size, stride, padding, dilation
    )
    if len(grad.shape) != 4 and len(grad.shape) != 3:
        raise ValueError(
            f"Expected gradient to have 3/4 dimensions (4 with batch). Found {len(grad.shape)}"
        )

    if len(indices.shape) != len(grad.shape) + 2:
        raise ValueError(
            "Expected indices shape to have 2 extra dimensions because of "
            f"(kernel_size, kernel_size), but has {len(indices.shape)}"
        )

    session = grad.session

    mappings = grad.view(grad.shape + (1, 1)) * indices
    args = [
        [tuple(input_shape), grads_share, kernel_size, stride, padding]
        for grads_share in mappings.share_ptrs
    ]
    shares = parallel_execution(max_pool2d_backward_helper, session.parties)(args)

    res = MPCTensor(shares=shares, shape=input_shape, session=session)
    return res

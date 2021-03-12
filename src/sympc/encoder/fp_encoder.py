"""The implementation for the FixedPrecisionTensor.

The implementation is taken from the Facebook Research Project: CrypTen
Website: https://crypten.ai/
GitHub: https://github.com/facebookresearch/CrypTen
"""


# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Part of the code is from the CrypTen Facebook Project

# stdlib
from typing import Union

# third party
import torch


class FixedPointEncoder:
    """Class for encoding/decoding a tensor to/from a fixed precision
    representation.

    This class was inspired from the Facebook Research - CrypTen project

    Arguments:
        base (int): the base for the encoder
        precision (int): the precision for the encoder

    Attributes:
        _precision (int): the precision for the encoder
        _base (int): the base for the encoder
        _scale (int): the scale used for encoding/decoding
    """

    __slots__ = {"_precision", "_base", "_scale"}

    def __init__(self, base: int = 2, precision: int = 16):
        """Initializer for the FP Encoder."""
        self._precision = precision
        self._base = base
        self._scale = base ** precision

    def encode(self, value: Union[torch.Tensor, float, int]) -> torch.LongTensor:
        """Encode a value using the FixedPoint Encoder.

        :return: the encoded value by FPEncoder
        :rtype: a long tensor
        """

        if not isinstance(value, torch.Tensor):
            value = torch.tensor(data=[value])

        # Use the largest type
        long_value = (value * self._scale).long()
        return long_value

    def decode(self, value: Union[int, torch.Tensor]) -> torch.Tensor:
        """Decode a value using the FixedPoint Encoder.

        :return: the decoded value for a tensor
        :rtype: a float tensor
        """

        if isinstance(value, torch.Tensor) and value.dtype.is_floating_point:
            raise ValueError(f"{value} should be converted to long format")

        if isinstance(value, int):
            value = torch.LongTensor([value])

        tensor = value
        if self._precision == 0:
            return tensor

        correction = (tensor < 0).long()
        dividend = tensor // self._scale - correction
        remainder = tensor % self._scale
        remainder += (remainder == 0).long() * self._scale * correction

        tensor = dividend.float() + remainder.float() / self._scale
        return tensor

    @property
    def precision(self):
        """Get the precision for the FixedPrecision Encoder."""
        return self._precision

    @precision.setter
    def precision(self, precision: int) -> None:
        """Set the precision for the FixedPoint Encoder By changing it, there
        will also be changed the "_scale" attribute."""
        self._precision = precision
        self._scale = self._base ** precision

    @property
    def base(self) -> int:
        """Get the base for the FixedPrecision Encoder."""
        return self._base

    @base.setter
    def base(self, base: int) -> None:
        """Set the base for the FixedPoint Encoder By changing it, there will
        also be changed the "_scale" attribute."""
        self._base = base
        self._scale = base ** self._precision

    @property
    def scale(self) -> int:
        """Get the scale for the FixedPrecision Encoder."""
        return self._scale

    def __str__(self) -> str:
        """Get the string representation."""
        type_name = type(self).__name__
        out = f"[{type_name}]: precision: {self._precision}, base: {self._base}"
        return out

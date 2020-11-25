# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Part of the code is from the CrypTen Facebook Project

from typing import Union

import torch


class FixedPointEncoder:
    __slots__ = {"_precision", "_base", "_scale"}

    def __init__(self, base: int = 10, precision: int = 4):
        self._precision = precision
        self._base = base
        self._scale = base ** precision

    def encode(self, value: Union[torch.Tensor, float, int]):
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(data=[value], dtype=torch.long)

        # Use the largest type
        long_value = (value * self._scale).long()
        return long_value

    def decode(self, tensor: torch.Tensor):
        if tensor.dtype.is_floating_point:
            raise ValueError(f"{tensor} should be converted to long format")

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
        return self._precision

    @property
    def base(self):
        return self._base

    @property
    def scale(self):
        return self._scale

    @precision.setter
    def precision(self, precision):
        self._precision = precision
        self._scale = self._base ** precision

    @base.setter
    def base(self, base):
        self._base = base
        self._scale = base ** self._precision

    def __str__(self):
        type_name = type(self).__name__
        out = f"[{type_name}]: precision: {self._precision}, base: {self._base}"
        return out

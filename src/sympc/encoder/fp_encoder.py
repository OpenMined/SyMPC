# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


class FixedPointEncoder:
    def __init__(self, base=10, precision=4):
        self._precision = precision
        self._base = base
        self._scale = base ** precision

    def encode(self, value):
        return (value * self._scale).long()

    def decode(self, tensor):
        if tensor.dtype.is_floating_point:
            raise ValueError(f"{tensor} should be converted to long format")

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

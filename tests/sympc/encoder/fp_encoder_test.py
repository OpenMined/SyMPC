"""
Tests for the Fixed Precision Encoder.
"""
import torch

from sympc.encoder import FixedPointEncoder


def test_fp_encoder_init():
    """
    Test correct FixedPointEncoder initialisation.
    """
    fp_encoder = FixedPointEncoder(
        base=3,
        precision=8,
    )
    assert fp_encoder.base == 3
    assert fp_encoder.precision == 8
    assert fp_encoder.scale == 3 ** 8


def test_fp_encoding():
    """
    Test correct encoding with FixedPointEncoder.
    """
    # Test encoding with tensors.
    fp_encoder = FixedPointEncoder()
    tensor = torch.Tensor([1, 2, 3])
    encoded_tensor = fp_encoder.encode(tensor)
    target_tensor = torch.LongTensor([1, 2, 3]) * fp_encoder.scale
    assert (encoded_tensor == target_tensor).all
    # Test encoding with foats.
    fp_encoder = FixedPointEncoder()
    test_float = 42.0
    encoded_float = fp_encoder.encode(test_float)
    target_float = torch.LongTensor([42.0]) * fp_encoder.scale
    assert (encoded_float == target_float).all
    # Test encoding with ints.
    fp_encoder = FixedPointEncoder()
    test_int = 2
    encoded_int = fp_encoder.encode(test_int)
    target_int = torch.LongTensor([2]) * fp_encoder.scale
    assert (encoded_int == target_int).all

"""
Tests for the Fixed Precision Encoder.
"""
import pytest
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
    fp_encoder = FixedPointEncoder()
    # Test encoding with tensors.
    tensor = torch.Tensor([1, 2, 3])
    encoded_tensor = fp_encoder.encode(tensor)
    target_tensor = torch.LongTensor([1, 2, 3]) * fp_encoder.scale
    assert (encoded_tensor == target_tensor).all()
    # Test encoding with foats.
    test_float = 42.0
    encoded_float = fp_encoder.encode(test_float)
    target_float = torch.LongTensor([42]) * fp_encoder.scale
    assert (encoded_float == target_float).all()
    # Test encoding with ints.
    test_int = 2
    encoded_int = fp_encoder.encode(test_int)
    target_int = torch.LongTensor([2]) * fp_encoder.scale
    assert (encoded_int == target_int).all()


def test_fp_decoding():
    """
    Test correct decoding with FixedPointEncoder.
    """
    fp_encoder = FixedPointEncoder()
    # Test decoding with tensors.
    # CaseA: throw a ValueError with floating point tensors.
    tensor = torch.Tensor([1.00, 2.00, 3.00])
    with pytest.raises(ValueError):
        fp_encoder.decode(tensor)
    # Test decoding with ints.
    # Case A (precision = 0):
    fp_encoder = FixedPointEncoder(precision=0)
    test_int = 2
    decoded_int = fp_encoder.decode(test_int)
    target_int = torch.LongTensor([2])
    assert (decoded_int == target_int).all()
    # Case B (precision != 0):
    fp_encoder = FixedPointEncoder()
    test_int = 2 * fp_encoder.base ** fp_encoder.precision
    decoded_int = fp_encoder.decode(test_int)
    target_int = torch.LongTensor([2])
    assert (decoded_int == target_int).all()


def test_fp_precision_setter():
    """
    Test the precision setter for the FixedPointEncoder.
    """
    fp_encoder = FixedPointEncoder()
    fp_encoder.precision = 3
    assert fp_encoder.precision == 3
    assert fp_encoder.scale == fp_encoder.base ** 3


def test_fp_base_setter():
    """
    Test the base setter for the FixedPointEncoder.
    """
    fp_encoder = FixedPointEncoder()
    fp_encoder.base = 3
    assert fp_encoder.base == 3
    assert fp_encoder.scale == 3 ** fp_encoder.precision


def test_fp_string_representation():
    """
    Test the string representation of the FixedPointEncoder.
    """
    fp_encoder = FixedPointEncoder()
    assert str(fp_encoder) == "[FixedPointEncoder]: precision: 16, base: 2"
